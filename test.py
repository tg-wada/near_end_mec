import argparse
import os
import cv2
import time
import mmcv
import numpy as np
import logging
from collections import deque
from multiprocessing import Process, Queue
from tqdm import tqdm
import json
import shutil

# カメラの設定
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
camera_position = (IMG_WIDTH / 2, IMG_HEIGHT)

# ログ設定
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 距離計算関数
def calculate_distance(center_x, center_y):
    return ((center_x - camera_position[0]) ** 2 + (center_y - camera_position[1]) ** 2) ** 0.5

# 平滑化
def smooth_distance_change(distance_changes, window_size=5):
    if len(distance_changes) < window_size:
        return np.mean(distance_changes)
    return np.mean(list(distance_changes)[-window_size:])

# フレーム取得
def frame_capture(rtsp_url, frame_queue, batch_size, output_dir):
    logging.info("Starting frame capture...")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error("Failed to connect to camera.")
        return
    
    frame_id = 0
    batch_id = 0
    batch_folder = os.path.join(output_dir, f"batch_{batch_id:03d}")
    os.makedirs(batch_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame capture failed. Exiting...")
            break

        frame_path = os.path.join(batch_folder, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_queue.put({"frame_id": frame_id, "frame_path": frame_path})

        frame_id += 1
        if frame_id % batch_size == 0:
            batch_id += 1
            batch_folder = os.path.join(output_dir, f"batch_{batch_id:03d}")
            os.makedirs(batch_folder, exist_ok=True)
            logging.info(f"Initiating batch {batch_id}...")

    cap.release()

# バッチ処理
def batch_processing_worker(frame_queue, output_dir, model, distance_threshold, min_bbox_area, window_size):
    logging.info("Starting batch processing...")
    previous_distances = {}
    distance_changes = {}

    while True:
        if frame_queue.empty():
            time.sleep(1)
            continue

        batch = []
        while not frame_queue.empty():
            batch.append(frame_queue.get())
            if len(batch) >= 10:  # バッチサイズ
                break

        if not batch:
            continue

        logging.info(f"Processing batch with {len(batch)} frames...")
        batch_data = []
        for frame_data in batch:
            try:
                frame_id = frame_data["frame_id"]
                frame_path = frame_data["frame_path"]
                img = mmcv.imread(frame_path)

                result = inference_mot(model, img, frame_id=frame_id)
                logging.debug(f"Processed frame {frame_id}")

                frame_info = {"frame_id": frame_id, "persons": []}

                if isinstance(result, dict) and "track_bboxes" in result:
                    for track in result["track_bboxes"]:
                        if len(track.shape) == 2:
                            for t in track:
                                if len(t) < 6:
                                    continue
                                track_id, x1, y1, x2, y2, score = t
                                if score < 0.5:
                                    continue

                                bbox_area = (x2 - x1) * (y2 - y1)
                                if bbox_area < min_bbox_area:
                                    continue

                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                current_distance = calculate_distance(center_x, center_y)

                                approaching = False
                                if track_id in previous_distances:
                                    prev_distance = previous_distances[track_id]
                                    distance_change = prev_distance - current_distance
                                    if track_id not in distance_changes:
                                        distance_changes[track_id] = deque(maxlen=window_size)
                                    distance_changes[track_id].append(distance_change)

                                    avg_distance_change = smooth_distance_change(distance_changes[track_id])
                                    if avg_distance_change > distance_threshold:
                                        approaching = True

                                previous_distances[track_id] = current_distance

                                frame_info["persons"].append({
                                    "track_id": int(track_id),
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "approaching": approaching
                                })
                batch_data.append(frame_info)

            except Exception as e:
                logging.error(f"Error processing frame {frame_data['frame_id']}: {e}")

        batch_json_path = os.path.join(output_dir, f"batch_{batch[0]['frame_id']}_data.json")
        with open(batch_json_path, "w") as f:
            json.dump(batch_data, f, indent=4)
        logging.info(f"Batch JSON saved: {batch_json_path}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run real-time person tracking with batch processing.")
    parser.add_argument("--rtsp_url", type=str, default="rtsp://root:root@192.168.5.93/axis-media/media.amp?videocodec=h264", help="RTSP URL of the camera.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save JSON files and frames.")
    parser.add_argument("--distance_threshold", type=float, default=1.0, help="Threshold for detecting approaching persons.")
    parser.add_argument("--min_bbox_area", type=int, default=20000, help="Minimum bounding box area to consider.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for smoothing distance changes.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of frames per batch.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # モデル初期化
    mot_config = "mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
    mot_checkpoint = "checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
    model = init_model(mot_config, mot_checkpoint, device="cpu")

    frame_queue = Queue()
    Process(target=frame_capture, args=(args.rtsp_url, frame_queue, args.batch_size, args.output_dir)).start()
    Process(target=batch_processing_worker, args=(
        frame_queue, args.output_dir, model, args.distance_threshold, args.min_bbox_area, args.window_size
    )).start()

if __name__ == "__main__":
    main()