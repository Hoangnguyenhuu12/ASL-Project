import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import re
import urllib.request
import threading
import zipfile
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import drive

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Định nghĩa các đường nối khớp tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def download_model(model_path='/content/hand_landmarker.task'):
    if os.path.exists(model_path) and os.path.getsize(model_path) < 1000:
        os.remove(model_path)
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            print(f"Lỗi tải model: {e}")
            exit()

class ColabHandCollector:
    def __init__(self, output_file='/content/hand_data_v2.csv', model_path='/content/hand_landmarker.task'):
        self.output_file = output_file
        self.model_path = model_path
        self._thread_local = threading.local()
        download_model(model_path)

        self.detector = self._create_detector(model_path)

        # Khởi tạo file CSV với header
        if not os.path.exists(self.output_file):
            with open(self.output_file, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                header = ['label']
                for i in range(21):
                    header.extend([f'x{i}', f'y{i}', f'z{i}'])
                writer.writerow(header)

    def _create_detector(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        return vision.HandLandmarker.create_from_options(options)

    def _get_thread_detector(self):
        detector = getattr(self._thread_local, 'detector', None)
        if detector is None:
            detector = self._create_detector(self.model_path)
            self._thread_local.detector = detector
        return detector

    def normalize_landmarks(self, landmarks):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        wrist = points[0]
        points = points - wrist
        palm_size = np.linalg.norm(points[9])
        if palm_size > 0: points = points / palm_size
        current_vector = points[9]
        angle = np.arctan2(current_vector[0], current_vector[1])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,             0,              1]
        ])
        return np.round(points.dot(rotation_matrix).flatten(), 3)

    def adjust_gamma(self, image_bgr, gamma=1.0):
        gamma = max(gamma, 1e-6)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image_bgr, table)

    def preprocess_for_detection(self, frame_bgr, mode='full'):
        variants = []
        base = frame_bgr.copy()
        h, w = base.shape[:2]

        variants.append(("original", base))

        if mode == 'fast':
            lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            clahe_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
            variants.append(("clahe", clahe_bgr))

            gamma_bright = self.adjust_gamma(base, gamma=1.3)
            variants.append(("gamma_bright", gamma_bright))
            return variants

        denoise = cv2.fastNlMeansDenoisingColored(base, None, 4, 4, 7, 21)
        variants.append(("denoise", denoise))

        bilateral = cv2.bilateralFilter(base, d=9, sigmaColor=50, sigmaSpace=50)
        variants.append(("bilateral", bilateral))

        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        clahe_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        variants.append(("clahe", clahe_bgr))

        lab2 = cv2.cvtColor(denoise, cv2.COLOR_BGR2LAB)
        l_d, a_d, b_d = cv2.split(lab2)
        l_d2 = clahe.apply(l_d)
        denoise_clahe = cv2.cvtColor(cv2.merge((l_d2, a_d, b_d)), cv2.COLOR_LAB2BGR)
        variants.append(("denoise_clahe", denoise_clahe))

        gamma_bright = self.adjust_gamma(base, gamma=1.4)
        gamma_dark = self.adjust_gamma(base, gamma=0.75)
        variants.append(("gamma_bright", gamma_bright))
        variants.append(("gamma_dark", gamma_dark))

        try:
            detailed = cv2.detailEnhance(base, sigma_s=10, sigma_r=0.15)
            variants.append(("detail_enhance", detailed))
        except Exception:
            pass

        up = cv2.resize(base, (int(w * 1.6), int(h * 1.6)), interpolation=cv2.INTER_CUBIC)
        up_back = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
        variants.append(("multiscale_up", up_back))

        down = cv2.resize(base, (max(32, int(w * 0.6)), max(32, int(h * 0.6))), interpolation=cv2.INTER_AREA)
        down_back = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        variants.append(("multiscale_down", down_back))

        return variants

    def _detect_from_single_image(self, frame_bgr, detector=None):
        if detector is None:
            detector = self.detector
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return None, None
        landmarks = result.hand_landmarks[0]
        static_coords = self.normalize_landmarks(landmarks)
        return landmarks, static_coords

    def detect_landmarks_from_frame(self, frame_bgr, mode='full', detector=None):
        if detector is None:
            detector = self.detector
        variants = self.preprocess_for_detection(frame_bgr, mode=mode)
        for variant_name, candidate in variants:
            landmarks, static_coords = self._detect_from_single_image(candidate, detector=detector)
            if landmarks is not None:
                return landmarks, static_coords, variant_name
        return None, None, "none"

    def append_many_to_csv(self, rows):
        if not rows:
            return
        with open(self.output_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def infer_label(self, image_path):
        # Lùi lại một bước để lấy đúng tên của cái kẹp hồ sơ (thư mục) chứa ảnh
        # Ví dụ: "...\A\frame_0000.jpg" sẽ bóc ra ngay lập tức chữ "A"
        parent_folder = os.path.basename(os.path.dirname(image_path)).strip()
        
        # Trả về kết quả và viết hoa toàn bộ để file CSV trông thật chuyên nghiệp
        return parent_folder.upper()

    def get_image_files(self, folder_path):
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in valid_ext:
                    image_paths.append(os.path.join(root, name))
        return image_paths

    def analyze_non_detection(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        h, w = gray.shape

        reasons = []
        if brightness < 55: reasons.append("Ảnh quá tối")
        elif brightness > 210: reasons.append("Ảnh quá sáng/cháy")
        if blur_score < 80: reasons.append("Ảnh bị mờ hoặc out-focus")
        if min(h, w) < 160: reasons.append("Ảnh có độ phân giải thấp")
        if not reasons: reasons.append("Pose tay bị che khuất/ngoài khung hoặc nền quá nhiễu")

        return {'brightness': round(brightness, 1), 'blur_score': round(blur_score, 1), 'width': int(w), 'height': int(h), 'reasons': reasons}

    def process_single_image_parallel(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            return {'status': 'skip', 'path': image_path, 'reason': 'Không đọc được ảnh'}

        detector = self._get_thread_detector()
        _, static_coords, detected_by = self.detect_landmarks_from_frame(frame, mode='fast', detector=detector)
        if static_coords is None:
            _, static_coords, detected_by = self.detect_landmarks_from_frame(frame, mode='full', detector=detector)

        if static_coords is None:
            diagnostics = self.analyze_non_detection(frame)
            reason_text = "; ".join(diagnostics['reasons'])
            return {'status': 'skip', 'path': image_path, 'reason': f"Không detect được tay | {reason_text}"}

        auto_label = self.infer_label(image_path)
        if not auto_label:
            return {'status': 'skip', 'path': image_path, 'reason': 'Không parse được label từ tên file/folder'}

        return {'status': 'ok', 'path': image_path, 'row': [auto_label] + static_coords.tolist()}

    def label_data_from_folder(self, folder_path):
        image_files = self.get_image_files(folder_path)
        if not image_files:
            print("-> KHÔNG TÌM THẤY ẢNH HỢP LỆ TRONG FOLDER NÀY.")
            return

        print(f"\n=== ĐANG XỬ LÝ {len(image_files)} ẢNH TỪ FOLDER ===")

        # Tối ưu hóa triệt để cho Colab Pro 44 vCPU
        cpu_threads = os.cpu_count() or 44
        max_workers = cpu_threads * 2 # Nhân đôi số luồng để tận dụng I/O
        print(f"-> Khởi chạy sức mạnh Colab Pro: Sử dụng {max_workers} luồng xử lý song song!")

        total_saved = 0
        rows_to_append = []

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_image_parallel, p) for p in image_files]

            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result['status'] == 'ok':
                    rows_to_append.append(result['row'])
                    total_saved += 1

                # Log tiến độ mỗi 500 ảnh để tránh rác console
                if idx % 500 == 0 or idx == len(image_files):
                    print(f"-> Tiến độ: {idx}/{len(image_files)} | Thành công: {total_saved}")

        self.append_many_to_csv(rows_to_append)

        elapsed_time = time.time() - start_time
        print(f"\n=== HOÀN TẤT! Đã lưu {total_saved} mẫu trong {elapsed_time:.2f} giây ===")

# ==============================================================================
# QUY TRÌNH CHẠY TRÊN COLAB
# ==============================================================================
def run_colab_pipeline(zip_drive_path, output_drive_folder, csv_filename="hand_data.csv"):
    # 1. Mount Google Drive
    print("Đang kết nối với Google Drive...")
    drive.mount('/content/drive')

    if not os.path.exists(zip_drive_path):
        print(f"LỖI: Không tìm thấy file zip tại: {zip_drive_path}")
        return

    # 2. Tạo đường dẫn tạm trên Colab để tăng tốc I/O
    local_extract_dir = '/content/temp_images'
    local_csv_path = f'/content/{csv_filename}'

    # Xóa folder cũ nếu có để tránh trùng lặp
    if os.path.exists(local_extract_dir):
        shutil.rmtree(local_extract_dir)
    if os.path.exists(local_csv_path):
        os.remove(local_csv_path)

    os.makedirs(local_extract_dir, exist_ok=True)

    # 3. Giải nén file ảnh
    print(f"\nĐang giải nén file: {zip_drive_path}...")
    try:
        with zipfile.ZipFile(zip_drive_path, 'r') as zip_ref:
            zip_ref.extractall(local_extract_dir)
        print("Giải nén hoàn tất vào môi trường cục bộ của Colab!")
    except Exception as e:
        print(f"LỖI giải nén: {e}")
        return

    # 4. Chạy quá trình trích xuất Landmark
    app = ColabHandCollector(output_file=local_csv_path)
    app.label_data_from_folder(local_extract_dir)

    # 5. Lưu kết quả về Google Drive
    if os.path.exists(local_csv_path):
        print("\nĐang lưu file CSV về Google Drive...")
        os.makedirs(output_drive_folder, exist_ok=True)
        final_drive_path = os.path.join(output_drive_folder, csv_filename)
        shutil.copy2(local_csv_path, final_drive_path)
        print(f"-> THÀNH CÔNG! File CSV đã được lưu an toàn tại: {final_drive_path}")
    else:
        print("-> LỖI: Không tìm thấy file CSV đầu ra để copy vào Drive.")

# --- ĐIỀN ĐƯỜNG DẪN CỦA BẠN VÀO ĐÂY VÀ CHẠY ---
if __name__ == "__main__":
    # Thay đổi các đường dẫn này cho phù hợp với Drive của bạn
    # Ví dụ: '/content/drive/MyDrive/Data/images.zip'
    FILE_ZIP_TREN_DRIVE = '/content/drive/MyDrive/Colab Notebooks/framesNguyen_Normalized.zip'

    # Thư mục trên Drive để lưu file CSV đầu ra
    THU_MUC_LUU_CSV = '/content/drive/MyDrive/Colab Notebooks'

    # Tên file muốn lưu
    TEN_FILE_CSV = 'hand_landmarks_v2.csv'

    run_colab_pipeline(FILE_ZIP_TREN_DRIVE, THU_MUC_LUU_CSV, TEN_FILE_CSV)