# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
# Sử dụng API cốt lõi của XGBoost
import xgboost as xgb
import time
import warnings
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns


# Ghi lại thời gian bắt đầu tổng thể của script
script_start_time_main = time.time()


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


print("-------------------------------------------------------------")
print("--- Script Kết hợp Annotations và Phân loại BI-RADS v5 (Objective softprob) ---")
print("-------------------------------------------------------------")


# =============================================================
# ===                1. THIẾT LẬP CẤU HÌNH                 ===
# =============================================================
# --- Đường dẫn Annotations ---
breast_level_csv = 'them data/breast-level_annotations.csv'
finding_level_csv = 'them data/finding_annotations.csv'
# --- Đường dẫn Ảnh ---
image_base_dir = 'orr'# Thư mục CHA
train_dir_name = "train"
test_dir_name = "test"
IMAGE_EXTENSION = ".png" # <<<--- !!! KIỂM TRA VÀ CẬP NHẬT ĐUÔI FILE !!!
# --- Đường dẫn Output ---
output_dir = "classification_output_v7_softprob" # Thư mục output
os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu chưa có
output_processed_csv_path = os.path.join(output_dir, "final_processed_annotations.csv")
output_model_path = os.path.join(output_dir, "xgboost_birads_model_core_softprob.json")
output_report_path = os.path.join(output_dir, "classification_results.txt")
output_training_plot_path = os.path.join(output_dir, "training_performance_curve.png")
output_cm_plot_path = os.path.join(output_dir, "confusion_matrix_heatmap.png")
output_roc_plot_path = os.path.join(output_dir, "roc_curves_ovr.png")
output_pr_plot_path = os.path.join(output_dir, "precision_recall_curves_ovr1.png")
# --- Cấu hình Kết hợp & Phân loại ---
KEY_COLUMNS = ['image_id']; IMAGE_PATH_COL = 'image_path'
MODEL_NAME = 'resnet50'; FEATURE_VECTOR_SIZE = 2048; IMAGE_SIZE = (224, 224)
# Params cho xgb.train (API cốt lõi) - SỬ DỤNG SOFTPROB
XGB_PARAMS = {
    'max_depth': 7, 'learning_rate': 0.1,
    'objective': 'multi:softprob', # <<<--- ĐỔI SANG SOFTPROB
    'eval_metric': 'mlogloss',
    'random_state': 42, 'nthread': -1
}
NUM_BOOST_ROUND = 400 # Số cây tối đa
EARLY_STOPPING_ROUNDS = 60 # Số vòng dừng sớm
TEST_SIZE = 0.2; RANDOM_STATE = 42
METADATA_COLS_FOR_MODEL = ['view_position', 'breast_density', 'laterality']; TARGET_COL = 'breast_birads'
print(f"📂 Thư mục Output: {os.path.abspath(output_dir)}")




# =============================================================
# ===          2. ĐỌC, LỌC VÀ KẾT HỢP ANNOTATIONS          ===
# =============================================================
print("\n[Bước 1/8] Đọc, lọc và kết hợp annotations...")
start_step_time = time.time()
# Kiểm tra file
if not os.path.exists(breast_level_csv): print(f"❌ Lỗi: Không tìm thấy file '{breast_level_csv}'"); exit()
if not os.path.exists(finding_level_csv): print(f"❌ Lỗi: Không tìm thấy file '{finding_level_csv}'"); exit()
print(f"   - Tìm thấy '{os.path.basename(breast_level_csv)}'.")
print(f"   - Tìm thấy '{os.path.basename(finding_level_csv)}'.")
# Đọc file
try: df_breast = pd.read_csv(breast_level_csv); print(f"   - Đọc {len(df_breast)} dòng từ '{os.path.basename(breast_level_csv)}'.")
except Exception as e: print(f"❌ Lỗi khi đọc file CSV '{breast_level_csv}': {e}"); exit()
try: df_finding = pd.read_csv(finding_level_csv); print(f"   - Đọc {len(df_finding)} dòng từ '{os.path.basename(finding_level_csv)}'.")
except Exception as e: print(f"❌ Lỗi khi đọc file CSV '{finding_level_csv}': {e}"); exit()
# Kiểm tra cột khóa
missing_keys_breast = [col for col in KEY_COLUMNS if col not in df_breast.columns]; missing_keys_finding = [col for col in KEY_COLUMNS if col not in df_finding.columns]
if missing_keys_breast: print(f"❌ Lỗi: File '{os.path.basename(breast_level_csv)}' thiếu cột khóa: {missing_keys_breast}"); exit()
if missing_keys_finding: print(f"❌ Lỗi: File '{os.path.basename(finding_level_csv)}' thiếu cột khóa: {missing_keys_finding}"); exit()
print(f"   - Cột khóa {KEY_COLUMNS} tồn tại.")
# Xác định cột chung và lọc
common_columns = list(set(df_breast.columns) & set(df_finding.columns)); print(f"   - Xác định {len(common_columns)} cột chung.")
df_breast_common = df_breast[common_columns].copy(); df_finding_common = df_finding[common_columns].copy()
# Merge
print(f"   - Thực hiện 'inner' join trên {KEY_COLUMNS}...")
merged_df = pd.merge(df_breast_common, df_finding_common, on=KEY_COLUMNS, how='inner', suffixes=('_breast', '_finding'))
# Xử lý cột trùng tên (ưu tiên _breast)
processed_cols = set(); final_cols = []
for col in merged_df.columns:
    base_col = col.replace('_breast', '').replace('_finding', '')
    if base_col not in processed_cols:
        if f"{base_col}_breast" in merged_df.columns: merged_df[base_col] = merged_df[f"{base_col}_breast"]
        elif f"{base_col}_finding" in merged_df.columns: merged_df[base_col] = merged_df[f"{base_col}_finding"]
        elif base_col in merged_df.columns: pass
        else: continue
        final_cols.append(base_col); processed_cols.add(base_col)
merged_df = merged_df[final_cols]
print(f"   - Số dòng sau khi 'inner' join: {len(merged_df)}")
# Xử lý trùng lặp
initial_merged_rows = len(merged_df); merged_df.drop_duplicates(subset=KEY_COLUMNS, keep='first', inplace=True)
duplicates_dropped = initial_merged_rows - len(merged_df)
if duplicates_dropped > 0: print(f"   - Đã loại bỏ {duplicates_dropped} bản ghi trùng lặp.")
print(f"   - Số dòng cuối cùng trong DataFrame kết hợp: {len(merged_df)}")
if merged_df.empty: print("❌ Lỗi: Không có dữ liệu chung. Kết thúc."); exit()
end_step_time = time.time(); print(f"✅ Kết hợp annotations hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
del df_breast, df_finding, df_breast_common, df_finding_common # Giải phóng bộ nhớ
gc.collect()




# =============================================================
# ===     3. TẠO ĐƯỜNG DẪN ẢNH VÀ KIỂM TRA DỮ LIỆU        ===
# =============================================================
print("\n[Bước 2/8] Tạo đường dẫn ảnh và kiểm tra dữ liệu kết hợp...")
start_step_time = time.time()
# Kiểm tra cột cần thiết
if 'split' not in merged_df.columns: print(f"❌ Lỗi: Thiếu cột 'split'."); exit()
if 'image_id' not in merged_df.columns: print(f"❌ Lỗi: Thiếu cột 'image_id'."); exit()
# Hàm tạo đường dẫn
def create_image_path(row):
    """Tạo đường dẫn tương đối dựa trên split và image_id."""
    # !!! ĐIỀU CHỈNH LOGIC NÀY NẾU CÓ THƯ MỤC CON (patient_id/study_id) !!!
    split_folder = train_dir_name if row['split'] == 'training' else test_dir_name
    image_filename = f"{row['image_id']}{IMAGE_EXTENSION}"
    # Ví dụ không có thư mục con:
    path = os.path.join(split_folder, image_filename)
    return path
print(f"   - Tạo cột '{IMAGE_PATH_COL}' với đuôi file '{IMAGE_EXTENSION}'...")
merged_df[IMAGE_PATH_COL] = merged_df.apply(create_image_path, axis=1)
if merged_df[IMAGE_PATH_COL].isnull().any(): print("   ⚠️ Cảnh báo: Có lỗi khi tạo đường dẫn ảnh."); merged_df.dropna(subset=[IMAGE_PATH_COL], inplace=True)
print(f"     Ví dụ đường dẫn: {merged_df[IMAGE_PATH_COL].iloc[0]}" if not merged_df.empty else "     (Không có dữ liệu)")
# Kiểm tra cột cần thiết cho model
required_cols_model = [IMAGE_PATH_COL] + METADATA_COLS_FOR_MODEL + [TARGET_COL]
missing_cols_model = [col for col in required_cols_model if col not in merged_df.columns]
if missing_cols_model: print(f"❌ Lỗi: DataFrame thiếu cột cho mô hình: {missing_cols_model}"); exit()
print(f"   - Các cột cần thiết ({required_cols_model}) đã tồn tại.")
# Xử lý NaN
initial_rows = len(merged_df); merged_df.dropna(subset=required_cols_model, inplace=True)
dropped_rows = initial_rows - len(merged_df)
if dropped_rows > 0: print(f"   - Đã loại bỏ {dropped_rows} dòng chứa giá trị thiếu.")
print(f"   - Số dòng hợp lệ sau khi loại bỏ NaN: {len(merged_df)}")
if merged_df.empty: print("❌ Lỗi: Không còn dữ liệu hợp lệ."); exit()
# Mã hóa
print("   - Mã hóa metadata và nhãn...")
le_view = LabelEncoder(); le_laterality = LabelEncoder(); le_density = LabelEncoder(); le_birads = LabelEncoder()
try:
    merged_df['view_position_encoded'] = le_view.fit_transform(merged_df['view_position'])
    merged_df['laterality_encoded'] = le_laterality.fit_transform(merged_df['laterality'])
    if not pd.api.types.is_numeric_dtype(merged_df['breast_density']): merged_df['breast_density_encoded'] = le_density.fit_transform(merged_df['breast_density'])
    else: merged_df['breast_density_encoded'] = merged_df['breast_density']
    merged_df['birads_encoded'] = le_birads.fit_transform(merged_df[TARGET_COL])
    num_classes = len(le_birads.classes_); target_names = [str(cls) for cls in le_birads.classes_]
    XGB_PARAMS['num_class'] = num_classes # Cập nhật tham số XGBoost
    print(f"     - Đã mã hóa các cột. Số lớp BI-RADS: {num_classes}")
except KeyError as ke: print(f"❌ Lỗi KeyError khi mã hóa cột: {ke}."); exit()
except Exception as e: print(f"❌ Lỗi không xác định khi mã hóa: {e}"); exit()
end_step_time = time.time(); print(f"✅ Tiền xử lý dữ liệu kết hợp hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")




# =============================================================
# ===          4. THIẾT LẬP TRÍCH XUẤT ĐẶC TRƯNG           ===
# =============================================================
print("\n[Bước 3/8] Thiết lập mô hình trích xuất đặc trưng ảnh...")
start_step_time = time.time()
image_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"   - Sử dụng thiết bị: {str(device).upper()}")
print(f"   - Tải mô hình {MODEL_NAME} pretrained...");
try:
    cnn_model = None
    if MODEL_NAME == 'resnet50': cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); cnn_model.fc = nn.Identity()
    elif MODEL_NAME == 'resnet101': cnn_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2); cnn_model.fc = nn.Identity()
    else: print(f"❌ Lỗi: Model '{MODEL_NAME}' không được hỗ trợ."); exit()
    if cnn_model: cnn_model = cnn_model.to(device); cnn_model.eval()
    else: print("❌ Lỗi: Không thể khởi tạo cnn_model."); exit()
    end_step_time = time.time(); print(f"✅ Tải và cấu hình {MODEL_NAME} thành công (Thời gian: {end_step_time - start_step_time:.2f}s).")
except Exception as e: print(f"❌ Lỗi khi tải/cấu hình CNN: {e}"); exit()




# =============================================================
# ===             5. TRÍCH XUẤT ĐẶC TRƯNG ẢNH              ===
# =============================================================
# ==== Định nghĩa LABEL_COL ====
# ==== Chuẩn bị thư mục lưu ảnh augment ====
# ==== Chuẩn bị thư mục lưu ảnh augment ====
import os
import time
import gc
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

augmented_images_dir = "augmented_images2"
os.makedirs(augmented_images_dir, exist_ok=True)

LABEL_COL = 'birads_encoded'

print("\n[Bước 4/8] Bắt đầu trích xuất đặc trưng ảnh (augment nhãn 2,3,4 để cân bằng với nhãn 1)...")
start_step_time = time.time()

# ==== Augmentation nhẹ cho nhãn 2,3 ====
light_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
])

# ==== Augmentation mạnh cho nhãn 4 ====
strong_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
])

# ==== Hàm extract feature từ PIL Image ====
def extract_feature_from_pil(img):
    img_tensor = image_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = cnn_model(img_tensor).squeeze().cpu().numpy()
    if np.isnan(feature_vector).any():
        return None
    return feature_vector

# ==== Các biến lưu trữ trung gian ====
features_list = []
valid_indices_processed = []
augmentation_flags = []
skipped_files_count = 0
augment_image_count = 0

# ==== Xác định số lượng nhãn hiện tại ====
label_counts = merged_df[LABEL_COL].value_counts()
print(f"🔎 Số lượng ban đầu: {dict(label_counts)}")

augment_labels = [2, 3, 4]  
target_per_label = 8000  # Mỗi nhãn augment tới 5000 ảnh (tính cả gốc và augment)

current_augmented_counts = {label: label_counts.get(label, 0) for label in augment_labels}
print(f"🎯 Mục tiêu số lượng cho mỗi nhãn: {target_per_label}")

# ==== Xử lý từng ảnh ====
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="   Trích xuất", unit="ảnh"):
    img_rel_path = row['image_path']
    img_full_path = os.path.join(image_base_dir, img_rel_path)
    label = row[LABEL_COL]

    try:
        img = Image.open(img_full_path).convert('RGB')

        # Trích đặc trưng từ ảnh gốc
        feat = extract_feature_from_pil(img)
        if feat is not None and feat.shape == (FEATURE_VECTOR_SIZE,):
            features_list.append(feat)
            valid_indices_processed.append(idx)
            augmentation_flags.append(False)

            # Nếu cần augment thêm
            if label in augment_labels:
                while current_augmented_counts[label] < target_per_label:
                    aug = strong_augment if label == 4 else light_augment
                    aug_img = aug(img)

                    aug_feat = extract_feature_from_pil(aug_img)
                    if aug_feat is not None and aug_feat.shape == (FEATURE_VECTOR_SIZE,):
                        features_list.append(aug_feat)
                        valid_indices_processed.append(idx)
                        augmentation_flags.append(True)
                        augment_image_count += 1
                        current_augmented_counts[label] += 1

                        # Lưu ảnh augment
                        orig_filename = os.path.splitext(os.path.basename(img_rel_path))[0]
                        aug_filename = f"{orig_filename}_aug{augment_image_count}.jpg"
                        aug_save_path = os.path.join(augmented_images_dir, aug_filename)
                        aug_img.save(aug_save_path)
                    else:
                        break  # Nếu lỗi feature thì dừng augment cho ảnh này

        else:
            skipped_files_count += 1

    except (FileNotFoundError, UnidentifiedImageError):
        skipped_files_count += 1
    except Exception as e:
        skipped_files_count += 1
        # print(f"⚠️ Lỗi xử lý ảnh {img_full_path}: {e}")

end_step_time = time.time()

# ==== Thống kê kết quả ====
print(f"✅ Trích xuất hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
print(f"   - Số lượng ảnh gốc thành công: {len(merged_df) - skipped_files_count}")
print(f"   - Số lượng ảnh augment tạo thêm: {augment_image_count}")
print(f"   - Tổng số feature vector (gốc + augment): {len(features_list)}")
print(f"   - Số lượng mới sau augment: {current_augmented_counts}")
if skipped_files_count > 0:
    print(f"   - Ảnh bị lỗi hoặc bỏ qua: {skipped_files_count}")
if not features_list:
    print("❌ Lỗi: Không trích xuất được bất kỳ đặc trưng nào!")
    exit()

# ==== Tạo DataFrame sau xử lý ====
df_final = merged_df.loc[valid_indices_processed].reset_index(drop=True)
df_final['is_augmented'] = augmentation_flags
image_features = np.stack(features_list)
print(f"   - Kích thước mảng đặc trưng: {image_features.shape}")

# ==== Lưu file CSV kết quả ====
try:
    df_final.to_csv(output_processed_csv_path, index=False)
    print(f"💾 Đã lưu DataFrame đã xử lý vào: {output_processed_csv_path}")
except Exception as e:
    print(f"⚠️ Lỗi khi lưu file CSV: {e}")

# ==== Dọn bộ nhớ ====
del merged_df, features_list
gc.collect()
torch.cuda.empty_cache() if device == 'cuda' else None

# ==== Vẽ biểu đồ ====
print("\n📊 Vẽ biểu đồ phân phối label (gốc + augment)...")

df_plot = df_final.copy()
df_plot['label'] = df_plot[LABEL_COL]

plt.figure(figsize=(10,6))
df_plot.groupby(['label', 'is_augmented']).size().unstack(fill_value=0).plot(
    kind='bar', stacked=True, ax=plt.gca(), color=['skyblue', 'salmon']
)
plt.title('Số lượng ảnh gốc và augment theo từng label')
plt.xlabel('Label')
plt.ylabel('Số lượng ảnh')
plt.legend(['Ảnh gốc', 'Ảnh augment'])
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================
# ===         6. KẾT HỢP ĐẶC TRƯNG CUỐI & CHIA DỮ LIỆU       ===
# =============================================================
print("\n[Bước 5/8] Kết hợp đặc trưng ảnh và metadata cuối cùng...")
start_step_time = time.time()

try:
    metadata_cols = ['view_position_encoded', 'breast_density_encoded', 'laterality_encoded']
    metadata_features = df_final[metadata_cols].values
    metadata_scaled = StandardScaler().fit_transform(metadata_features)
    print("   - Đã áp dụng StandardScaler cho metadata.")

    X = np.hstack([image_features, metadata_scaled])
    y = df_final['birads_encoded'].values
    print(f"   - Kích thước X: {X.shape}, y: {y.shape}")

except KeyError as ke:
    print(f"❌ Lỗi KeyError: {ke}. Kiểm tra lại METADATA_COLS_FOR_MODEL."); exit()
except Exception as e:
    print(f"❌ Lỗi khi kết hợp đặc trưng: {e}"); exit()

print(f"✅ Hoàn tất (Thời gian: {time.time() - start_step_time:.2f}s).")

# =============================================================
# ===                  6.1. CHIA DẮ LIỆU                   ===
# =============================================================
print("\n[Bước 6/8] Chia dữ liệu thành tập huấn luyện và kiểm tra...")
start_step_time = time.time()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"   - Tập Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   - Tập Test : X={X_test.shape}, y={y_test.shape}")
except Exception as e:
    print(f"❌ Lỗi khi chia dữ liệu: {e}"); exit()

print(f"✅ Hoàn tất (Thời gian: {time.time() - start_step_time:.2f}s).")

# =============================================================
# ===              7. HUẤN LUYỆN & LƯU MÔ HÌNH (SVM)        ===
# =============================================================
from sklearn.svm import SVC
import pickle

print("\n[Bước 7/8] Huấn luyện mô hình SVM và lưu model...")
start_step_time = time.time()

try:
    clf = SVC(probability=True, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print("   - Huấn luyện SVM thành công.")

    # Lưu model
    with open(output_model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"     📅 Lưu mô hình vào: {output_model_path}")
except Exception as e:
    print(f"❌ Lỗi huấn luyện/lưu SVM: {e}")

print(f"✅ Hoàn tất (Thời gian: {time.time() - start_step_time:.2f}s).")

# =============================================================
# ===             8. ĐÁNH GIÁ & LƯU KẾT QUẢ                ===
# =============================================================
print("\n[Bước 8/8] Đánh giá mô hình và lưu kết quả...")
start_step_time = time.time()

metrics_ready = False

try:
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    y_true = y_test.astype(int)
    metrics_ready = True
except Exception as e:
    print(f"❌ Lỗi dự đoán: {e}")

accuracy = -1.0
if metrics_ready:
    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        cm_df = pd.DataFrame(cm, index=le_birads.classes_, columns=le_birads.classes_)
    except Exception as e:
        print(f"❌ Lỗi khi tính toán metrics: {e}")
        metrics_ready = False

if metrics_ready:
    print(f"\n   - Độ chính xác: {accuracy:.4f}")
    print("\n   - Classification Report:\n", report)
    print("\n   - Confusion Matrix:\n", cm_df)

    # Heatmap CM
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True'); ax_cm.set_title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(output_cm_plot_path); plt.close()
    print(f"     📅 Đã lưu heatmap CM: {output_cm_plot_path}")

    # ROC Curve
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'{target_names[i]} (AUC={roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--'); ax_roc.legend(); ax_roc.grid(True)
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        plt.tight_layout(); plt.savefig(output_roc_plot_path); plt.close()
        print(f"     📅 Đã lưu ROC Curve: {output_roc_plot_path}")
    except Exception as e:
        print(f"⚠️ Lỗi vẽ ROC: {e}")

    # PR Curve
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            ax_pr.plot(recall, precision, label=f'{target_names[i]} (AP={ap:.2f})')
        ax_pr.legend(); ax_pr.grid(True)
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        plt.tight_layout(); plt.savefig(output_pr_plot_path); plt.close()
        print(f"     📅 Đã lưu PR Curve: {output_pr_plot_path}")
    except Exception as e:
        print(f"⚠️ Lỗi vẽ PR: {e}")

# Lưu report txt
print(f"\n   - Lưu kết quả vào: {output_report_path}")
try:
    total_duration = time.time() - script_start_time_main
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("="*53 + "\n")
        f.write("===       KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI BI-RADS       ===\n")
        f.write("="*53 + "\n\n")
        f.write(f"Tổng thời gian: {total_duration:.2f} giây ({total_duration/60:.2f} phút)\n\n")
        f.write(f"Cấu hình:\n  - Model: {MODEL_NAME}\n  - IMAGE_SIZE: {IMAGE_SIZE}\n")
        f.write(f"  - Classifier: SVM\n  - Test size: {TEST_SIZE:.2%}\n  - Num classes: {num_classes}\n\n")
        if metrics_ready:
            f.write(f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}\n\n")
            f.write(f"Confusion Matrix:\n{cm_df.to_string()}\n\n")
        else:
            f.write("Lỗi khi tính toán metrics.\n")
        f.write("="*53 + "\n")
    print("     📅 Đã lưu báo cáo.")
except Exception as e:
    print(f"⚠️ Lỗi khi lưu báo cáo: {e}")

print("\n📅 Hoàn tất đánh giá và lưu kết quả (Thời gian: {:.2f}s).".format(time.time() - start_step_time))
print("\n-------------------------------------------------------------")
print("--- Quy trình Kết hợp và Phân loại BI-RADS Hoàn tất ---")
print("-------------------------------------------------------------")

 

