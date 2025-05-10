import os
import pandas as pd
import shutil

print("--- Bắt đầu Tách Ảnh ---")

# --- Cấu hình ---
CSV_PATH = "csv/image_birads 4.csv"     # File CSV chứa danh sách hình ảnh
IMAGE_DIR = "file_train_tach"       # Thư mục chứa ảnh gốc
OUTPUT_DIR = "ảnh_đã_tách_4"       # Thư mục lưu các ảnh được tách ra
IMAGE_EXTENSION = ".png"         # Định dạng ảnh

# --- Kiểm tra Input ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Không tìm thấy file CSV: {CSV_PATH}")
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}")

# --- Đọc file CSV ---
df = pd.read_csv(CSV_PATH)
if "image_id" not in df.columns:
    raise ValueError("File CSV không có cột 'image_id'.")

# --- Tạo thư mục đầu ra ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Tách ảnh ---
copied = 0
missing = 0

for idx, row in df.iterrows():
    image_name = str(row["image_id"]).strip()

    if not image_name:
        print(f"[Dòng {idx}] Thiếu tên ảnh. Bỏ qua.")
        continue

    # Đảm bảo có đuôi ảnh
    if not image_name.lower().endswith(IMAGE_EXTENSION):
        image_name += IMAGE_EXTENSION

    source_path = os.path.join(IMAGE_DIR, image_name)
    dest_path = os.path.join(OUTPUT_DIR, image_name)

    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, dest_path)
            print(f"✅ Đã sao chép: {image_name}")
            copied += 1
        except Exception as e:
            print(f"❌ Lỗi khi sao chép ảnh {image_name}: {e}")
    else:
        print(f"⚠️ Ảnh không tồn tại: {image_name}")
        missing += 1

# --- Tóm tắt ---
print("\n--- Kết quả ---")
print(f"Tổng ảnh cần xử lý: {len(df)}")
print(f"✔️ Ảnh đã sao chép: {copied}")
print(f"❌ Ảnh bị thiếu: {missing}")
print(f"📁 Ảnh đã lưu tại: {os.path.abspath(OUTPUT_DIR)}")
print("--- Kết thúc ---")
