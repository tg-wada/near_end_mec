import os
import cv2
import mmcv
import numpy as np
import logging
import torch
import json
from collections import deque
from multiprocessing import Process, Queue
from tqdm import tqdm
from mmtrack.apis import inference_mot, init_model
import argparse

# カメラの位置設定
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
camera_position = (IMG_WIDTH / 2, IMG_HEIGHT)

# ログの設定
def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'tracking.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

# 距離計算関数
def calculate_distance(center_x, center_y):
    distance = ((center_x - camera_position[0]) ** 2 + (center_y - camera_position[1]) ** 2) ** 0.5
    return distance

# 平滑化
def smooth_distance_change(distance_changes, window_size=5):
    if len(distance_changes) < window_size:
        return np.mean(distance_changes)
    return np.mean(list(distance_changes)[-window_size:])

# 接近検知とJSON保存を行うワーカー
def process_worker(frame_q: Queue, distance_threshold, min_bbox_area, window_size, display_duration, json_file_path):
    previous_distances = {}
    distance_changes = {}
    approaching_frames = {}
    tracking_data = []  # フレームごとの情報を保存するリスト

    while True:
        item = frame_q.get()
        if item is None:
            # Noneを受け取ったら終了
            break
        i, frame, result = item
        frame_info = {
            "frame_id": i,
            "persons": []  # 人物情報を保存するためのリスト
        }

        if isinstance(result, dict) and 'track_bboxes' in result:
            for track in result['track_bboxes']:
                if len(track.shape) == 2:
                    for t in track:
                        if len(t) < 6:
                            continue
                        track_id, x1, y1, x2, y2, score = t
                        if score < 0.5:
                            continue

                        # バウンディングボックスの面積計算
                        bbox_area = (x2 - x1) * (y2 - y1)
                        if bbox_area < min_bbox_area:
                            continue

                        # 中心座標と距離を計算
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        current_distance = calculate_distance(center_x, center_y)

                        # 前フレームとの比較
                        approaching = False  # 初期化
                        if track_id in previous_distances:
                            prev_distance = previous_distances[track_id]
                            distance_change = prev_distance - current_distance
                            if track_id not in distance_changes:
                                distance_changes[track_id] = deque(maxlen=window_size)
                            distance_changes[track_id].append(distance_change)

                            # 平滑化された距離変化の平均を計算
                            avg_distance_change = smooth_distance_change(distance_changes[track_id], window_size=window_size)

                            # 接近を検出
                            if avg_distance_change > distance_threshold:
                                approaching_frames[track_id] = display_duration
                                logging.info(f"Frame {i}: Person Approaching detected for track_id: {track_id}")
                                approaching = True

                        # 距離記録
                        previous_distances[track_id] = current_distance

                        # フレーム情報に追加
                        person_info = {
                            "track_id": int(track_id),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "approaching": approaching
                        }
                        frame_info["persons"].append(person_info)

        tracking_data.append(frame_info)  # トラッキングデータにフレーム情報を追加

    # JSONファイルに保存
    with open(json_file_path, 'w') as json_file:
        json.dump(tracking_data, json_file, indent=4)

# メイン処理
def main(rtsp_url, output_dir, distance_threshold, min_bbox_area, window_size, display_duration):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    setup_logging(output_dir)

    # モデルの設定ファイルとチェックポイント
    mot_config = 'mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
    checkpoints_dir = 'checkpoints'
    checkpoint_filename = 'bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    mot_checkpoint = os.path.join(checkpoints_dir, checkpoint_filename)

    print("Initializing model...")
    mot_model = init_model(mot_config, None, device='cpu')
    checkpoint = torch.load(mot_checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ema_')}
    mot_model.load_state_dict(new_state_dict, strict=False)
    mot_model.CLASSES = ('person',)
    print("Model initialized.")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("IPカメラに接続できませんでした")
        return

    frame_queue = Queue()
    json_file_path = os.path.join(output_dir, 'tracking_data.json')
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ワーカー開始
    process = Process(target=process_worker, args=(
        frame_queue, distance_threshold, min_bbox_area, window_size, display_duration, json_file_path))
    process.start()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました")
            break

        # フレームを保存
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # トラッキングの実行
        result = inference_mot(mot_model, frame, frame_id=frame_count)
        frame_queue.put((frame_count, frame, result))
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 処理終了
    frame_queue.put(None)
    process.join()
    cap.release()
    cv2.destroyAllWindows()

    print(f'接近検知の結果が {json_file_path} に保存されました')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracking on an IP camera stream.")
    parser.add_argument("--rtsp_url", type=str, required=True, help="RTSP URL for the IP camera.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSON and frames.")
    parser.add_argument("--distance_threshold", type=float, default=1, help="Threshold for detecting approaching objects.")
    parser.add_argument("--min_bbox_area", type=int, default=20000, help="Minimum bounding box area to consider.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for smoothing the distance change.")
    parser.add_argument("--display_duration", type=int, default=30, help="Number of frames to display 'Person Approaching'.")
    args = parser.parse_args()
    main(args.rtsp_url, args.output_dir, args.distance_threshold, args.min_bbox_area, args.window_size, args.display_duration)