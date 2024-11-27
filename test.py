import os
import cv2
import mmcv
from multiprocessing import Process, Queue, cpu_count
import json
from mmtrack.apis import inference_mot, init_model

# 接近検知を行うための設定
def calculate_distance(center_x, center_y, camera_position):
    return ((center_x - camera_position[0]) ** 2 + (center_y - camera_position[1]) ** 2) ** 0.5

def smooth_distance_change(distance_changes, window_size=5):
    if len(distance_changes) < window_size:
        return sum(distance_changes) / len(distance_changes)
    return sum(distance_changes[-window_size:]) / window_size

# バッチ処理ワーカー
def batch_processing_worker(worker_id, frame_queue, output_dir, model, distance_threshold, min_bbox_area, window_size):
    print(f"[Worker {worker_id}] Starting batch processing worker...")
    camera_position = (1920 / 2, 1080)  # カメラの仮想中心
    while True:
        try:
            batch_data = frame_queue.get(timeout=10)  # キューが空なら10秒待機して終了
        except Exception as e:
            print(f"[Worker {worker_id}] Queue timeout or empty: {e}")
            break

        if batch_data is None:
            print(f"[Worker {worker_id}] Received termination signal.")
            break

        batch_id, frames = batch_data
        print(f"[Worker {worker_id}] Processing batch {batch_id} with {len(frames)} frames.")
        batch_results = []

        for frame_id, frame_path in frames:
            try:
                img = mmcv.imread(frame_path)
                result = inference_mot(model, img, frame_id=frame_id)

                # フレームごとの結果を解析
                frame_result = {"frame_id": frame_id, "persons": []}
                if isinstance(result, dict) and 'track_bboxes' in result:
                    for track in result['track_bboxes']:
                        if len(track.shape) == 2:
                            for t in track:
                                if len(t) < 6:
                                    continue
                                track_id, x1, y1, x2, y2, score = t
                                if score < 0.5:
                                    continue
                                # バウンディングボックスの解析
                                bbox_area = (x2 - x1) * (y2 - y1)
                                if bbox_area < min_bbox_area:
                                    continue
                                
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                current_distance = calculate_distance(center_x, center_y, camera_position)

                                approaching = current_distance < distance_threshold

                                person_info = {
                                    "track_id": int(track_id),
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "approaching": approaching
                                }
                                frame_result["persons"].append(person_info)
                batch_results.append(frame_result)
            except Exception as e:
                print(f"[Worker {worker_id}] Error processing frame {frame_id}: {e}")

        # バッチ結果をJSONとして保存
        batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        json_path = os.path.join(batch_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(batch_results, f, indent=4)

        # フレームを移動
        for frame_id, frame_path in frames:
            target_path = os.path.join(batch_dir, os.path.basename(frame_path))
            os.rename(frame_path, target_path)
        print(f"[Worker {worker_id}] Finished processing batch {batch_id}. Results saved to {json_path}")


def main(rtsp_url, output_dir, batch_size, distance_threshold, min_bbox_area, window_size):
    print("[INFO] Starting frame capture...")
    os.makedirs(output_dir, exist_ok=True)

    # モデルの初期化
    print("[INFO] Initializing model...")
    mot_config = 'mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
    mot_checkpoint = 'checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    model = init_model(mot_config, mot_checkpoint, device='cpu')
    print("[INFO] Model initialized.")

    # RTSPストリームからフレームを取得
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[ERROR] Unable to open RTSP stream.")
        return

    frame_queue = Queue()
    frame_list = []
    frame_id = 0
    batch_id = 0

    # ワーカープロセスの開始
    num_workers = cpu_count()
    print(f"[INFO] Starting {num_workers} worker processes...")
    workers = []
    for worker_id in range(num_workers):
        p = Process(target=batch_processing_worker, args=(
            worker_id, frame_queue, output_dir, model, distance_threshold, min_bbox_area, window_size))
        p.start()
        workers.append(p)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of RTSP stream or connection lost.")
            break

        # フレームの保存
        frame_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_list.append((frame_id, frame_path))
        frame_id += 1

        # バッチ処理の準備
        if len(frame_list) >= batch_size:
            print(f"[INFO] Initiating batch {batch_id}...")
            frame_queue.put((batch_id, frame_list))
            frame_list = []
            batch_id += 1

    # 残りのフレームをバッチ処理
    if frame_list:
        print(f"[INFO] Initiating final batch {batch_id}...")
        frame_queue.put((batch_id, frame_list))

    # 終了シグナルを送信
    for _ in range(num_workers):
        frame_queue.put(None)

    # ワーカーの終了を待機
    for p in workers:
        p.join()

    cap.release()
    print("[INFO] All processing completed.")


if __name__ == "__main__":
    parser.add_argument("--rtsp_url", type=str, default="rtsp://root:root@192.168.5.93/axis-media/media.amp?videocodec=h264", help="RTSP URL of the camera.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save JSON files and frames.")
    parser.add_argument("--batch_size", type=int, default=30, help="Number of frames per batch.")
    parser.add_argument("--distance_threshold", type=float, default=1, help="Threshold for detecting approaching objects.")
    parser.add_argument("--min_bbox_area", type=int, default=20000, help="Minimum bounding box area to consider.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for smoothing distance change.")
    args = parser.parse_args()

    main(args.rtsp_url, args.output_dir, args.batch_size, args.distance_threshold, args.min_bbox_area, args.window_size)