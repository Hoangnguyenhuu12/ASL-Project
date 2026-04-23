import cv2
import os
import glob
import numpy as np
import mediapipe as mp
import concurrent.futures
import multiprocessing
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Tất cả đường dẫn đều tương đối với vị trí của file này
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")
INPUT_DIR  = str(BASE_DIR / "data" / "frames")          # Thư mục chứa ảnh gốc (output của cutvideo.py)
OUTPUT_DIR = str(BASE_DIR / "data" / "frames_normalized") # Thư mục lưu ảnh đã chuẩn hóa

# Kích thước ảnh chuẩn mà mọi camera/khoảng cách đều phải quy về
STANDARD_SIZE = 256
# Tỷ lệ viền đắp thêm (1.5 nghĩa là khung cắt rộng gấp rưỡi bàn tay)
PADDING_RATIO = 1.5

def setup_landmarker():
    """Khởi tạo bộ nhận diện tay của Mediapipe."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)

def get_dynamic_crop_info(hand_landmarks, img_width, img_height):
    """Đo đạc kích thước tay để tính ra cái khung cắt vừa vặn nhất."""
    x_coords = [landmark.x * img_width for landmark in hand_landmarks[0]]
    y_coords = [landmark.y * img_height for landmark in hand_landmarks[0]]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    center_x = int((min_x + max_x) / 2)
    center_y = int((min_y + max_y) / 2)
    
    hand_w = max_x - min_x
    hand_h = max_y - min_y
    
    dynamic_size = int(max(hand_w, hand_h) * PADDING_RATIO)
    
    return center_x, center_y, dynamic_size

def crop_and_pad(image, center_x, center_y, crop_size):
    """Cắt ảnh theo kích thước động, đắp viền đen nếu lẹm ra ngoài."""
    img_height, img_width = image.shape[:2]
    half_size = crop_size // 2

    start_y = center_y - half_size
    end_y   = center_y + half_size
    start_x = center_x - half_size
    end_x   = center_x + half_size

    pad_top    = max(0, -start_y)
    pad_bottom = max(0, end_y - img_height)
    pad_left   = max(0, -start_x)
    pad_right  = max(0, end_x - img_width)

    padded_img = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    new_start_y = start_y + pad_top
    new_end_y   = end_y   + pad_top
    new_start_x = start_x + pad_left
    new_end_x   = end_x   + pad_left

    return padded_img[new_start_y:new_end_y, new_start_x:new_end_x]

worker_detector = None

def init_worker():
    global worker_detector
    worker_detector = setup_landmarker()

def process_single_image(task_args):
    img_path, out_folder_path, folder_name = task_args
    
    image = cv2.imread(img_path)
    if image is None:
        return None
        
    img_height, img_width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    detection_result = worker_detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        cx, cy, dynamic_size = get_dynamic_crop_info(
            detection_result.hand_landmarks, img_width, img_height
        )
        cropped_img     = crop_and_pad(image, cx, cy, dynamic_size)
        standardized_img = cv2.resize(cropped_img, (STANDARD_SIZE, STANDARD_SIZE))
        
        base_filename   = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        img_save_path   = os.path.join(out_folder_path, f"{name_without_ext}.jpg")
        cv2.imwrite(img_save_path, standardized_img)
        return True
    return False

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy file model tại: {MODEL_PATH}")
        print("Hãy tải về bằng lệnh:")
        print("  wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    if not os.path.exists(INPUT_DIR):
        print(f"[LỖI] Không tìm thấy thư mục đầu vào: {INPUT_DIR}")
        print(f"Hãy chạy cutvideo.py trước để tạo thư mục này.")
        return

    print(f"[INFO] Thư mục đầu vào : {INPUT_DIR}")
    print(f"[INFO] Thư mục đầu ra  : {OUTPUT_DIR}")
    print("Đang nạp năng lượng cho hệ thống chuẩn hóa...")
    
    all_tasks = []
    for folder_path in glob.glob(os.path.join(INPUT_DIR, '*')):
        if not os.path.isdir(folder_path):
            continue
            
        folder_name     = os.path.basename(folder_path)
        out_folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(out_folder_path, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]*')))
        for img_path in image_files:
            all_tasks.append((img_path, out_folder_path, folder_name))

    if not all_tasks:
        print("[CẢNH BÁO] Trống không! Chưa có ảnh để xử lý.")
        return

    max_workers = multiprocessing.cpu_count()
    print(f"Quét được {len(all_tasks)} ảnh. Chạy với {max_workers} nhân CPU!")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker
    ) as executor:
        results = list(executor.map(process_single_image, all_tasks))
        
    success_count = sum(1 for r in results if r)
    print(f"\nHoàn tất! Đã chuẩn hóa {success_count}/{len(all_tasks)} khung hình.")
    print(f"Ảnh đã lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()