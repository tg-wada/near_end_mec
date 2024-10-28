import argparse
import os
import time
import mmcv
import numpy as np
import logging
import torch
from collections import deque
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm
import json
import shutil  # 追加
from mmtrack.apis import inference_mot, init_model

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

# 接近検知を行い、フレームごとの情報をJSON形式で保存するワーカー
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
        i, img_path, result = item
        img = mmcv.imread(img_path)
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
                                print(f"Frame {i}: Person Approaching detected for track_id: {track_id}")
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

def main(input_video, output_dir, distance_threshold, min_bbox_area, window_size, display_duration):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    setup_logging(output_dir)

    # モデルの設定ファイルとチェックポイント
    mot_config = 'mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # チェックポイントのファイル名とパス
    checkpoint_filename = 'bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    mot_checkpoint = os.path.join(checkpoints_dir, checkpoint_filename)

    # チェックポイントの存在確認とダウンロード
    if not os.path.exists(mot_checkpoint):
        print("Checkpoint not found locally")
    else:
        print(f"Loading checkpoint from local file: {mot_checkpoint}")

    print("Initializing model...")
    mot_model = init_model(mot_config, None, device='cpu')  # チェックポイントを指定せずにモデルを初期化

    # チェックポイントをロードし、EMA パラメータを除外
    checkpoint = torch.load(mot_checkpoint, map_location='cpu')

    # 'ema_' で始まるキーを除外
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('ema_'):
            new_state_dict[k] = v

    # モデルにパラメータをロード
    mot_model.load_state_dict(new_state_dict, strict=False)

    # クラス名を設定
    mot_model.CLASSES = ('person',)
    print("Model initialized.")

    imgs = mmcv.VideoReader(input_video)

    # 一時的にフレームを保存
    temp_dir = os.path.join(output_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    for i, img in enumerate(imgs):
        img_path = os.path.join(temp_dir, f'frame_{i}.jpg')
        mmcv.imwrite(img, img_path)
        frame_paths.append(img_path)

    print("Starting person tracking...")
    tracking_results = []
    tracking_times = []

    # トラッキング処理（シングルプロセス）
    for i in tqdm(range(len(frame_paths)), desc="Tracking Progress"):
        img_path = frame_paths[i]
        img = mmcv.imread(img_path)
        start_time = time.time()
        result = inference_mot(mot_model, img, frame_id=i)
        end_time = time.time()
        processing_time = end_time - start_time
        tracking_results.append((i, img_path, result))
        tracking_times.append(processing_time)
    print("Person tracking done")

    # プロセス間通信のためのキューを作成
    frame_queue = Queue()

    num_processes = 32  # 32コアCPUを前提としてプロセス数を設定
    json_file_path = os.path.join(output_dir, 'tracking_data.json')

    # プロセスの作成と開始
    processes = []
    for _ in range(num_processes):
        p = Process(target=process_worker, args=(
            frame_queue, distance_threshold, min_bbox_area, window_size, display_duration, json_file_path))
        p.start()
        processes.append(p)

    # フレームデータをキューに送信
    for item in tracking_results:
        frame_queue.put(item)

    # ワーカーに終了を通知
    for _ in range(num_processes):
        frame_queue.put(None)

    # 全てのプロセスが終了するのを待機
    for p in processes:
        p.join()

    logging.info(f'Results have been saved to {json_file_path}')
    print(f'Results have been saved to {json_file_path}')

    # 平均処理時間を出力
    total_frames = len(frame_paths)
    average_tracking_time = sum(tracking_times) / total_frames
    print(f'Average tracking time per frame: {average_tracking_time:.4f} seconds')

    # 一時ファイルの削除
    for img_path in frame_paths:
        os.remove(img_path)
    shutil.rmtree(temp_dir)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracking on a video file.")
    parser.add_argument("--input_video", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output video and log file.")
    parser.add_argument("--distance_threshold", type=float, default=1, help="Threshold for detecting approaching objects.")
    parser.add_argument("--min_bbox_area", type=int, default=20000, help="Minimum bounding box area to consider.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for smoothing the distance change.")
    parser.add_argument("--display_duration", type=int, default=30, help="Number of frames to display 'Person Approaching'.")
    args = parser.parse_args()
    main(args.input_video, args.output_dir, args.distance_threshold, args.min_bbox_area, args.window_size, args.display_duration)