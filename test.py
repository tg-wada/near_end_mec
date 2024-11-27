import cv2
import os
from multiprocessing import Process, Queue, cpu_count
from mmtrack.apis import inference_mot, init_model
import json
from collections import deque
import numpy as np
import time
import signal

# タイムアウト例外
class TimeoutException(Exception):
    pass

# タイムアウトハンドラ
def timeout_handler(signum, frame):
    raise TimeoutException("Inference timed out")

# RTSP URLと出力ディレクトリ
rtsp_url = "rtsp://root:root@192.168.5.93/axis-media/media.amp?videocodec=h264"
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# バッチサイズ設定
BATCH_SIZE = 20
# カメラ位置設定
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
camera_position = (IMG_WIDTH / 2, IMG_HEIGHT)
# 接近検知閾値
DISTANCE_THRESHOLD = 20
WINDOW_SIZE = 5
MIN_BBOX_AREA = 20000


def calculate_distance(center_x, center_y):
    """カメラ位置と人物中心点の距離を計算"""
    return ((center_x - camera_position[0]) ** 2 + (center_y - camera_position[1]) ** 2) ** 0.5


def smooth_distance_change(distance_changes, window_size=5):
    """距離変化を平滑化"""
    if len(distance_changes) < window_size:
        return np.mean(distance_changes)
    return np.mean(list(distance_changes)[-window_size:])


def frame_capture(rtsp_url, frame_queue):
    """RTSPストリームからフレームを取得して保存し、キューに送信します。"""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[ERROR] Cannot connect to the RTSP stream.")
        return

    frame_id = 0
    print("[INFO] Starting frame capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No frame received. Exiting...")
            break

        # キューが満杯の時は待機
        while frame_queue.full():
            print("[INFO] Frame queue is full. Waiting...")
            time.sleep(0.1)

        # フレーム保存とキュー送信
        frame_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_queue.put((frame_id, frame_path))
        print(f"[INFO] Captured frame {frame_id}")
        frame_id += 1

    # 終了を通知
    frame_queue.put(None)
    cap.release()
    print("[INFO] Frame capture complete.")


def initialize_model():
    """モデルの初期化を行います。"""
    try:
        mot_config = "mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
        mot_checkpoint = "checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
        model = init_model(mot_config, mot_checkpoint, device="cpu")
        model.CLASSES = ('person',)
        print("[INFO] Model initialized.")
        return model
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {str(e)}")
        return None


def batch_processing_worker(batch_queue, model, batch_output_dir):
    """バッチ処理を実行し、接近検知結果を含むJSONファイルを保存します。"""
    if model is None:
        print("[ERROR] Model is not initialized. Exiting worker.")
        return

    previous_distances = {}
    distance_changes = {}

    while True:
        batch_data = batch_queue.get()
        if batch_data is None:
            print("[INFO] Worker received termination signal. Exiting.")
            break

        try:
            print(f"[INFO] Processing batch with {len(batch_data)} frames...")
            results = []
            for frame_id, frame_path in batch_data:
                try:
                    # フレームを読み込む
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"[ERROR] Failed to read frame: {frame_path}. Skipping.")
                        continue

                    # 推論を実行
                    print(f"[DEBUG] Running inference for frame {frame_id}...")
                    result = inference_mot(model, frame, frame_id=frame_id)

                    # 推論結果のデバッグログ
                    if result:
                        print(f"[DEBUG] Inference result for frame {frame_id}: {result}")
                    else:
                        print(f"[WARNING] No result returned for frame {frame_id}. Skipping frame.")
                        continue

                    # 推論結果の解析
                    frame_results = {"frame_id": frame_id, "persons": []}
                    if 'track_bboxes' in result:
                        for track in result['track_bboxes']:
                            if len(track.shape) == 2:
                                for t in track:
                                    if len(t) < 6:
                                        continue
                                    track_id, x1, y1, x2, y2, score = t
                                    if score < 0.5:
                                        continue

                                    # バウンディングボックスの面積
                                    bbox_area = (x2 - x1) * (y2 - y1)
                                    if bbox_area < MIN_BBOX_AREA:
                                        continue

                                    # 中心座標と距離計算
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    current_distance = calculate_distance(center_x, center_y)

                                    # 接近状態の判定
                                    approaching = False
                                    if track_id in previous_distances:
                                        prev_distance = previous_distances[track_id]
                                        distance_change = prev_distance - current_distance
                                        if track_id not in distance_changes:
                                            distance_changes[track_id] = deque(maxlen=WINDOW_SIZE)
                                        distance_changes[track_id].append(distance_change)

                                        avg_distance_change = smooth_distance_change(
                                            distance_changes[track_id], window_size=WINDOW_SIZE
                                        )

                                        if avg_distance_change > DISTANCE_THRESHOLD:
                                            approaching = True

                                    # 距離の記録
                                    previous_distances[track_id] = current_distance

                                    # 結果に追加
                                    frame_results["persons"].append({
                                        "track_id": int(track_id),
                                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                        "approaching": approaching
                                    })

                    # フレーム結果を追加
                    results.append(frame_results)
                    print(f"[DEBUG] Processed frame {frame_id}, persons detected: {len(frame_results['persons'])}")

                except Exception as e:
                    print(f"[ERROR] Error during processing frame {frame_id}: {str(e)}")

            # JSONファイルの保存
            if results:
                batch_id = batch_data[0][0] // BATCH_SIZE
                json_path = os.path.join(batch_output_dir, f"batch_{batch_id:03d}.json")
                try:
                    with open(json_path, "w") as json_file:
                        json.dump(results, json_file, indent=4)
                    print(f"[INFO] Batch {batch_id} processed and saved to {json_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save JSON for batch {batch_id}: {str(e)}")
            else:
                print("[WARNING] No results to save for this batch.")

        except Exception as e:
            print(f"[ERROR] Error processing batch: {str(e)}")


def create_directories():
    """必要なディレクトリを作成します。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_output_dir = os.path.join(output_dir, "batch_results")
    os.makedirs(batch_output_dir, exist_ok=True)
    return batch_output_dir


def main():
    """メインの実行関数"""
    create_directories()

    frame_queue = Queue(maxsize=100)
    batch_queue = Queue()

    # モデル初期化
    model = initialize_model()

    # フレームキャプチャプロセス
    capture_process = Process(target=frame_capture, args=(rtsp_url, frame_queue))
    capture_process.start()

    # バッチ処理プロセス
    batch_output_dir = os.path.join(output_dir, "batch_results")
    num_workers = cpu_count()
    workers = []
    for _ in range(num_workers):
        worker = Process(target=batch_processing_worker, args=(batch_queue, model, batch_output_dir))
        worker.start()
        workers.append(worker)

    # フレームキューからバッチ作成
    batch = []
    while True:
        try:
            item = frame_queue.get(timeout=5)  # 5秒間データを待つ
            if item is None:  # 終了通知
                print("[INFO] Frame capture finished. Finalizing batches...")
                if batch:  # 残ったバッチを処理
                    batch_queue.put(batch)
                break

            batch.append(item)

            # バッチが完成したら送信
            if len(batch) == BATCH_SIZE:
                batch_queue.put(batch)
                print(f"[INFO] Batch of {BATCH_SIZE} frames queued for processing.")
                batch = []

        except Exception:
            print("[INFO] Frame queue is empty or timed out. Finalizing batches...")
            if batch:
                batch_queue.put(batch)
                batch = []
            break

    # ワーカー終了処理
    capture_process.join()
    for _ in workers:
        batch_queue.put(None)  # ワーカー終了通知
    for worker in workers:
        worker.join()

    print("[INFO] Processing complete. Exiting.")


if __name__ == "__main__":
    main()