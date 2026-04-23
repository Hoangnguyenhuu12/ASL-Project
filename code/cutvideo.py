import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter

import cv2

# ==========================================
# 1. PHẦN CẤU HÌNH (Bạn chỉ cần sửa ở đây)
# ==========================================

# Tất cả đường dẫn đều tương đối với vị trí của file này
BASE_DIR = Path(__file__).resolve().parent

# Thư mục chứa các file video của bạn (đặt video vào đây trước khi chạy)
THU_MUC_CHUA_VIDEO = str(BASE_DIR / "data" / "videos")

# Thư mục lưu các frame ảnh được tách ra (sẽ được tạo tự động)
THU_MUC_LUU_ANH = str(BASE_DIR / "data" / "frames")

# Có thể tăng/giảm các giá trị này nếu muốn ưu tiên tốc độ hoặc RAM
SO_CPU        = os.cpu_count() or 1
DINH_DANG_VIDEO = (".mp4", ".avi", ".mov", ".mkv")
CHAT_LUONG_JPEG = 95

# ==========================================
# 2. PHẦN MÁY MÓC HOẠT ĐỘNG (Không cần đụng tới)
# ==========================================

def _ghi_anh(ten_anh, khung_hinh):
    thanh_cong = cv2.imwrite(ten_anh, khung_hinh, [cv2.IMWRITE_JPEG_QUALITY, CHAT_LUONG_JPEG])
    if not thanh_cong:
        raise RuntimeError(f"Không thể lưu ảnh: {ten_anh}")


def _thu_don_cong_viec(danh_sach_future, cho_den_khi_nho_hon=None):
    while danh_sach_future and (
        cho_den_khi_nho_hon is None or len(danh_sach_future) >= cho_den_khi_nho_hon
    ):
        da_xong, dang_cho = wait(danh_sach_future, return_when=FIRST_COMPLETED)
        danh_sach_future  = list(dang_cho)
        for future in da_xong:
            future.result()
        if cho_den_khi_nho_hon is not None and len(danh_sach_future) < cho_den_khi_nho_hon:
            break
    return danh_sach_future


def cat_video_thanh_anh(duong_dan, thu_muc, so_luong_luong_ghi, gioi_han_hang_doi):
    os.makedirs(thu_muc, exist_ok=True)
    cv2.setNumThreads(1)

    video = cv2.VideoCapture(duong_dan)
    if not video.isOpened():
        raise RuntimeError(f"Không thể mở video: {duong_dan}")

    dem           = 0
    bat_dau       = perf_counter()
    danh_sach_future = []

    with ThreadPoolExecutor(max_workers=so_luong_luong_ghi) as executor:
        while True:
            thanh_cong, khung_hinh = video.read()
            if not thanh_cong:
                break

            ten_anh = os.path.join(thu_muc, f"frame_{dem:04d}.jpg")
            danh_sach_future.append(executor.submit(_ghi_anh, ten_anh, khung_hinh.copy()))
            dem += 1

            if len(danh_sach_future) >= gioi_han_hang_doi:
                danh_sach_future = _thu_don_cong_viec(
                    danh_sach_future,
                    cho_den_khi_nho_hon=max(1, gioi_han_hang_doi // 2),
                )

        _thu_don_cong_viec(danh_sach_future)

    video.release()
    thoi_gian = perf_counter() - bat_dau
    toc_do    = dem / thoi_gian if thoi_gian > 0 else 0
    print(
        f"Xong {os.path.basename(duong_dan)}: {dem} ảnh trong {thoi_gian:.2f}s "
        f"({toc_do:.2f} frame/s, {so_luong_luong_ghi} luồng ghi)"
    )
    return dem


def _xu_ly_mot_video(cong_viec):
    duong_dan_video, thu_muc_dich_goc, so_luong_luong_ghi, gioi_han_hang_doi = cong_viec
    ten_file        = os.path.basename(duong_dan_video)
    ten_khong_duoi  = os.path.splitext(ten_file)[0]
    thu_muc_con     = os.path.join(thu_muc_dich_goc, ten_khong_duoi)

    print(f"\n--- Đang xử lý: {ten_file} ---")
    so_anh = cat_video_thanh_anh(
        duong_dan_video,
        thu_muc_con,
        so_luong_luong_ghi=so_luong_luong_ghi,
        gioi_han_hang_doi=gioi_han_hang_doi,
    )
    return ten_file, so_anh


def xu_ly_toan_bo_thu_muc(thu_muc_nguon, thu_muc_dich):
    if not os.path.exists(thu_muc_nguon):
        print(f"[LỖI] Không tìm thấy thư mục video: {thu_muc_nguon}")
        print(f"Hãy tạo thư mục '{thu_muc_nguon}' và đặt video vào đó.")
        return

    os.makedirs(thu_muc_dich, exist_ok=True)

    danh_sach_video = [
        os.path.join(thu_muc_nguon, ten_file)
        for ten_file in os.listdir(thu_muc_nguon)
        if ten_file.lower().endswith(DINH_DANG_VIDEO)
    ]

    if not danh_sach_video:
        print(f"[CẢNH BÁO] Không tìm thấy video nào trong: {thu_muc_nguon}")
        print(f"Định dạng được hỗ trợ: {DINH_DANG_VIDEO}")
        return

    so_tien_trinh    = min(SO_CPU, len(danh_sach_video))
    so_luong_luong_ghi = max(1, SO_CPU // so_tien_trinh)
    gioi_han_hang_doi  = max(so_luong_luong_ghi * 4, 16)

    print(f"[INFO] Thư mục video  : {thu_muc_nguon}")
    print(f"[INFO] Thư mục lưu ảnh: {thu_muc_dich}")
    print(
        f"Tìm thấy {len(danh_sach_video)} video. "
        f"Chạy với {so_tien_trinh} tiến trình, {so_luong_luong_ghi} luồng ghi mỗi video."
    )

    cong_viec = [
        (duong_dan_video, thu_muc_dich, so_luong_luong_ghi, gioi_han_hang_doi)
        for duong_dan_video in danh_sach_video
    ]

    tong_anh = 0
    bat_dau  = perf_counter()

    with ProcessPoolExecutor(max_workers=so_tien_trinh) as executor:
        for ten_file, so_anh in executor.map(_xu_ly_mot_video, cong_viec):
            tong_anh += so_anh
            print(f"Hoàn tất {ten_file}: {so_anh} ảnh")

    thoi_gian = perf_counter() - bat_dau
    print(f"\nTổng cộng đã tách {tong_anh} ảnh trong {thoi_gian:.2f}s.")
    print(f"Ảnh đã lưu tại: {thu_muc_dich}")


if __name__ == "__main__":
    xu_ly_toan_bo_thu_muc(THU_MUC_CHUA_VIDEO, THU_MUC_LUU_ANH)