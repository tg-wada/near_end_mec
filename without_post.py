import os
import cv2
import time
import json
import requests
import torch
import numpy as np
from collections import deque
from multiprocessing import Process, Queue, cpu_count
from mmtrack.apis import inference_mot, init_model

# グローバル設定
rtsp_url = "rtsp://root:root@192.168.5.93/axis-media/media.amp?videocodec=h264"
output_dir = "output"
batch_size = 10
post_url = "http://your-server-endpoint/api"
distance_threshold = 1.0
min_bbox_area = 20000
window_size = 5
display_duration = 30
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
camera_position = (IMG_WIDTH / 2, IMG_HEIGHT)

# モデル初期化
def initialize_model():
    mot_config = 'mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
    checkpoint_path = 'checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    model = init_model(mot_config, None, device='cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('ema_')}
    model.load_state_dict(state_dict, strict=False)
    model.CLASSES = ('person',)
    return model

# 距離計算関数
def calculate_distance(center_x, center_y):
    return ((center_x - camera_position[0]) ** 2 + (center_y - camera_position[1]) ** 2) ** 0.5

# バッチ処理ワーカー
def batch_processing_worker(frame_queue: Queue, model):
    previous_distances, distance_changes, approaching_frames = {}, {}, {}
    batch_count = 0

    while True:
        batch = []
        while len(batch) < batch_size:
            frame_data = frame_queue.get()
            if frame_data is None:
                return
            batch.append(frame_data)

        tracking_data = []
        batch_folder = os.path.join(output_dir, f"batch_{batch_count:04d}")
        os.makedirs(batch_folder, exist_ok=True)

        for frame_id, img_path in batch:
            img = cv2.imread(img_path)
            result = inference_mot(model, img, frame_id=frame_id)
            frame_info = {"frame_id": frame_id, "persons": []}

            if isinstance(result, dict) and 'track_bboxes' in result:
                for track in result['track_bboxes']:
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
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            current_distance = calculate_distance(center_x, center_y)

                            approaching = False
                            if track_id in previous_distances:
                                prev_distance = previous_distances[track_id]
                                distance_change = prev_distance - current_distance
                                if track_id not in distance_changes:
                                    distance_changes[track_id] = deque(maxlen=window_size)
                                distance_changes[track_id].append(distance_change)

                                avg_distance_change = np.mean(list(distance_changes[track_id])[-window_size:])
                                if avg_distance_change > distance_threshold:
                                    approaching_frames[track_id] = display_duration
                                    approaching = True
                            previous_distances[track_id] = current_distance
                            frame_info["persons"].append({
                                "track_id": int(track_id),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "approaching": approaching
                            })

            tracking_data.append(frame_info)
            # 処理済みフレームをバッチフォルダに移動
            os.rename(img_path, os.path.join(batch_folder, os.path.basename(img_path)))

        # バッチごとにJSONファイルを保存
        json_data = json.dumps(tracking_data, indent=4)
        json_filename = os.path.join(batch_folder, f"tracking_data_batch_{batch_count:04d}.json")
        with open(json_filename, 'w') as json_file:
            json_file.write(json_data)

        # JSONファイルをサーバーにPOST送信
        try:
            response = requests.post(post_url, data=json_data)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error posting JSON: {e}")

        batch_count += 1

# メイン処理
def main():
    os.makedirs(output_dir, exist_ok=True)
    model = initialize_model()

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Failed to connect to the camera.")
        return

    frame_queue = Queue()
    num_workers = cpu_count()
    processes = [Process(target=batch_processing_worker, args=(frame_queue, model)) for _ in range(num_workers)]
    for p in processes:
        p.start()

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを保存し、キューに入れる
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_queue.put((frame_count, frame_filename))
            frame_count += 1

            if frame_count % batch_size == 0:
                time.sleep(1)  # 次のバッチ処理待機
    finally:
        cap.release()
        for _ in range(num_workers):
            frame_queue.put(None)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()