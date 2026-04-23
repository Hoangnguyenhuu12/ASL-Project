import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import joblib
import csv
from pathlib import Path
import time

# Xác định đường dẫn gốc để gọi file
BASE_DIR = Path(__file__).resolve().parent

# 1. Tải mô hình và bộ giải mã nhãn
# Lưu ý: Đảm bảo tên file .pkl chính xác với file bạn đang có
print("Đang khởi động hội đồng chuyên gia Random Forest...")
model_rf = joblib.load(BASE_DIR / "mo_hinh_randomforest_cu_chi.pkl")
label_encoder = joblib.load(BASE_DIR / "bo_giai_ma_nhan_randomforest.pkl")

# 2. Thiết lập bộ dò tìm bàn tay (Hand Landmarker Task)
base_options = python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

STANDARD_SIZE = 256
PADDING_RATIO = 1.5
DEBUG_SAVE_FEATURES = False
DEBUG_FEATURES_CSV = BASE_DIR / "production_features_debug.csv"
MIRROR_CAMERA = True
WINDOW_NAME = "Sign Language - Production"
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Danh sách kết nối để vẽ xương bàn tay bằng OpenCV
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# Tên 60 cột đặc trưng phải khớp hoàn toàn với file CSV (x1, y1, z1... x20, y20, z20)
ten_cot = []
for i in range(1, 21):
    ten_cot.extend([f'x{i}', f'y{i}', f'z{i}'])


def init_debug_csv():
    if not DEBUG_SAVE_FEATURES:
        return
    if DEBUG_FEATURES_CSV.exists():
        return
    with open(DEBUG_FEATURES_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + ten_cot)


def append_debug_features(features):
    if not DEBUG_SAVE_FEATURES:
        return
    with open(DEBUG_FEATURES_CSV, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([f"{time.time():.6f}"] + features.tolist())


def normalize_landmarks_for_model(landmarks):
    """Đồng bộ chuẩn hóa landmark theo label.py rồi lấy 60 chiều (landmark 1..20)."""
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Wrist-center
    points = points - points[0]

    # Palm-size normalization theo vector wrist -> middle_mcp (landmark 9)
    palm_size = np.linalg.norm(points[9])
    if palm_size > 0:
        points = points / palm_size

    # Rotation alignment giống label.py
    current_vector = points[9]
    angle = np.arctan2(current_vector[0], current_vector[1])
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    normalized = points.dot(rotation_matrix)

    # Giữ đúng schema 60 chiều x1..z20 đang dùng cho model production
    return normalized[1:21].flatten()


def crop_and_pad(image, center_x, center_y, crop_size):
    """Cắt theo tâm tay và đắp viền đen nếu vượt biên ảnh."""
    img_height, img_width = image.shape[:2]
    half_size = crop_size // 2

    start_y = center_y - half_size
    end_y = center_y + half_size
    start_x = center_x - half_size
    end_x = center_x + half_size

    pad_top = max(0, -start_y)
    pad_bottom = max(0, end_y - img_height)
    pad_left = max(0, -start_x)
    pad_right = max(0, end_x - img_width)

    padded_img = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    new_start_y = start_y + pad_top
    new_end_y = end_y + pad_top
    new_start_x = start_x + pad_left
    new_end_x = end_x + pad_left

    return padded_img[new_start_y:new_end_y, new_start_x:new_end_x]


def draw_hand_skeleton(image, landmarks, width, height):
    for s, e in HAND_CONNECTIONS:
        p1 = (int(landmarks[s].x * width), int(landmarks[s].y * height))
        p2 = (int(landmarks[e].x * width), int(landmarks[e].y * height))
        cv2.line(image, p1, p2, (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 5, (255, 0, 255), -1)

# 3. Mở Camera (Độ phân giải 1080x1920 như dữ liệu đầu vào)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Khoa kich thuoc cua so de khong bi co gian theo tung frame
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("Camera đã mở! Chế độ nhận diện đồng bộ tọa độ chuẩn hóa đang chạy.")
init_debug_csv()

pTime = 0

while True:
    thanh_cong, frame = cap.read()
    if not thanh_cong:
        continue

    if MIRROR_CAMERA:
        frame = cv2.flip(frame, 1)

    # Tính FPS để kiểm tra độ mượt
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # --- Pipeline 1: Xu ly nhan dien/crop/model (khong hien thi truc tiep) ---
    img_h, img_w, _ = frame.shape
    display_frame = frame.copy()
    render_text = "Tim tay..."
    render_color = (0, 0, 255)
    model_time_text = f"FPS: {int(fps)}"
    
    # Bước 1: Tìm vị trí bàn tay trên toàn khung hình để xác định tâm
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_full = detector.detect(mp_image)

    if detection_full.hand_landmarks:
        landmarks_full = detection_full.hand_landmarks[0]
        draw_hand_skeleton(display_frame, landmarks_full, img_w, img_h)
        
        # Tính tâm bàn tay dựa trên min/max tọa độ (y hệt file cropanh.py)
        x_coords = [lm.x * img_w for lm in landmarks_full]
        y_coords = [lm.y * img_h for lm in landmarks_full]
        center_x = int((min(x_coords) + max(x_coords)) / 2)
        center_y = int((min(y_coords) + max(y_coords)) / 2)
        
        # Bước 2: Crop động + pad giống cropanh.py
        hand_w = max(x_coords) - min(x_coords)
        hand_h = max(y_coords) - min(y_coords)
        crop_size = max(32, int(max(hand_w, hand_h) * PADDING_RATIO))
        cropped = crop_and_pad(frame, center_x, center_y, crop_size)

        # Bước 3: Ép về 256x256 giống môi trường testing trong cropanh.py
        frame_256 = cv2.resize(cropped, (STANDARD_SIZE, STANDARD_SIZE))

        # Bước 4: Detect lần 2 trên ảnh chuẩn hóa 256x256
        rgb_256 = cv2.cvtColor(frame_256, cv2.COLOR_BGR2RGB)
        mp_image_256 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_256)
        detection_crop = detector.detect(mp_image_256)

        if detection_crop.hand_landmarks:
            ban_tay = detection_crop.hand_landmarks[0]

            dac_trung = normalize_landmarks_for_model(ban_tay)
            append_debug_features(dac_trung)

            df_input = pd.DataFrame([dac_trung], columns=ten_cot)
            
            # Dự đoán và tính toán độ tin tưởng (%)
            t_start = time.perf_counter()
            probs = model_rf.predict_proba(df_input)
            confidence = np.max(probs) * 100
            nhan_so = np.argmax(probs)
            t_ms = (time.perf_counter() - t_start) * 1000
            
            chu_cai = label_encoder.inverse_transform([nhan_so])[0]
            
            # Pipeline 2: Chi hien thi tren khung goc, khong zoom khung cua so
            hien_thi = chu_cai if confidence > 15 else "?"
            render_text = f"Chu: {hien_thi} ({confidence:.1f}%)"
            render_color = (0, 255, 0) if confidence > 50 else (0, 165, 255)
            model_time_text = f"Model: {t_ms:.1f}ms | FPS: {int(fps)}"

    # Pipeline 2: Hien thi on dinh kich thuoc cua so
    cv2.putText(display_frame, render_text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, render_color, 2)
    cv2.putText(display_frame, model_time_text, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    frame_display = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow(WINDOW_NAME, frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()