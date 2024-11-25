import os
import time
import json
import cv2
import mmcv
import numpy as np
import torch
from multiprocessing import Process, Queue
from mmtrack.apis import inference_mot, init_model
from collections import deque

# バッチごとに処理したフレームとJSONデータを保存する関数
def save_json_and_frames(batch_json_data, processed_frames_dir, batch_id):
    try:
        os.makedirs(processed_frames_dir, exist_ok=True)

        # JSONファイルの保存
        json_file_path = os.path.join(processed_frames_dir, f"batch_{batch_id}_data.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(batch_json_data, json_file, indent=4)
        print(f"[INFO] Saved JSON file: {json_file_path}")

        # フレームの保存
        for frame_info in batch_json_data:
            frame_id = frame_info["frame_id"]
            frame_data = frame_info.get("image", None)
            if frame_data is not None:
                frame_path = os.path.join(processed_frames_dir, f"frame_{frame_id:04d}.jpg")
                cv2.imwrite(frame_path, frame_data)
                print(f"[INFO] Saved frame: {frame_path}")

    except Exception as e:
        print(f"[ERROR] Error saving JSON and frames: {e}")

# 人物の接近を検知
def calculate_distance_change(distance_queue, current_distance, window_size):
    if len(distance_queue) >= window_size:
        distance_queue.popleft()
    distance_queue.append(current_distance)

    if len(distance_queue) > 1:
        distance_change = [distance_queue[i] - distance_queue[i - 1] for i in range(1, len(distance_queue))]
        avg_distance_change = np.mean(distance_change)
        return avg_distance_change
    return 0

# バッチ処理ワーカー
def batch_processing_worker(frame_queue, result_queue, model, distance_threshold, window_size, min_bbox_area, batch_id):
    tracking_data = []
    previous_distances = {}
    distance_queues = {}

    print(f"[INFO] Starting batch processing for batch {batch_id}")

    while not frame_queue.empty():
        try:
            frame_data = frame_queue.get_nowait()
            frame_id, image = frame_data["frame_id"], frame_data["image"]

            print(f"[DEBUG] Processing frame ID: {frame_id}")

            # 推論実行
            result = inference_mot(model, image, frame_id=frame_id)

            frame_info = {"frame_id": frame_id, "persons": [], "image": image}

            if "track_bboxes" in result:
                for bbox_data in result["track_bboxes"]:
                    if bbox_data[4] < min_bbox_area:
                        continue

                    track_id = int(bbox_data[0])
                    x1, y1, x2, y2 = bbox_data[1:5]

                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    current_distance = np.sqrt(center_x ** 2 + center_y ** 2)

                    if track_id not in distance_queues:
                        distance_queues[track_id] = deque(maxlen=window_size)

                    avg_distance_change = calculate_distance_change(distance_queues[track_id], current_distance, window_size)
                    approaching = avg_distance_change > distance_threshold

                    person_info = {
                        "track_id": track_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "approaching": approaching,
                    }
                    frame_info["persons"].append(person_info)

            tracking_data.append(frame_info)

        except Exception as e:
            print(f"[ERROR] Error in batch_processing_worker: {e}")

    print(f"[INFO] Batch {batch_id} processing completed.")
    result_queue.put((batch_id, tracking_data))

# メイン処理
def main(rtsp_url, output_dir, batch_size, distance_threshold, window_size, min_bbox_area):
    os.makedirs(output_dir, exist_ok=True)

    # モデル初期化
    config_file = "mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
    checkpoint_file = "checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
    model = init_model(config_file, checkpoint_file, device="cpu")

    frame_queue = Queue()
    result_queue = Queue()
    cap = cv2.VideoCapture(rtsp_url)

    batch_id = 0
    processed_frames_dir = os.path.join(output_dir, "processed_frames")

    try:
        print("[INFO] Starting frame capture...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame.")
                break

            # フレームをキューに追加
            frame_queue.put({"frame_id": batch_id, "image": frame})
            batch_id += 1

            # バッチ処理
            if frame_queue.qsize() >= batch_size:
                print(f"[INFO] Initiating batch {batch_id}...")
                process = Process(
                    target=batch_processing_worker,
                    args=(
                        frame_queue,
                        result_queue,
                        model,
                        distance_threshold,
                        window_size,
                        min_bbox_area,
                        batch_id,
                    ),
                )
                process.start()
                process.join()

                # 結果を取得して保存
                while not result_queue.empty():
                    batch_id, batch_json_data = result_queue.get()
                    print(f"[INFO] Received batch {batch_id} result.")
                    save_json_and_frames(batch_json_data, processed_frames_dir, batch_id)

    except Exception as e:
        print(f"[ERROR] Main process encountered an error: {e}")

    finally:
        cap.release()
        print("[INFO] Processing complete.")

if __name__ == "__main__":
    rtsp_url = "rtsp://root:root@192.168.5.93/axis-media/media.amp?videocodec=h264"
    output_dir = "output"
    batch_size = 30
    distance_threshold = 5
    window_size = 5
    min_bbox_area = 20000
    main(rtsp_url, output_dir, batch_size, distance_threshold, window_size, min_bbox_area)