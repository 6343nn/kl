# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
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
import xgboost as xgb
import time
import warnings
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
image_base_dir = 'orr'  # Thư mục CHA
train_dir_name = "train"
test_dir_name = "test"
IMAGE_EXTENSION = ".png"  # <<<--- !!! KIỂM TRA VÀ CẬP NHẬT ĐUÔI FILE !!!
# --- Đường dẫn Output ---
output_dir = "classification_output_v001_softprob_lime_shap"  # Thư mục output
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
output_processed_csv_path = os.path.join(output_dir, "final_processed_annotations1.csv")
output_model_path = os.path.join(output_dir, "xgboost_birads_model_core_softprob1.json")
output_report_path = os.path.join(output_dir, "classification_results1.txt")
output_training_plot_path = os.path.join(output_dir, "training_performance_curve1.png")
output_cm_plot_path = os.path.join(output_dir, "confusion_matrix_heatmap1.png")
output_roc_plot_path = os.path.join(output_dir, "roc_curves_ovr1.png")
output_pr_plot_path = os.path.join(output_dir, "precision_recall_curves_ovr1.png")
# --- Cấu hình Kết hợp & Phân loại ---
KEY_COLUMNS = ['image_id']
IMAGE_PATH_COL = 'image_path'
MODEL_NAME = 'resnet50'
FEATURE_VECTOR_SIZE = 2048
IMAGE_SIZE = (224, 224)
# Params cho xgb.train (API cốt lõi) - SỬ DỤNG SOFTPROB
XGB_PARAMS = {
    'max_depth': 7,
    'learning_rate': 0.1,
    'objective': 'multi:softprob',  # <<<--- ĐỔI SANG SOFTPROB
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'nthread': -1
}
NUM_BOOST_ROUND = 400  # Số cây tối đa
EARLY_STOPPING_ROUNDS = 50  # Số vòng dừng sớm
TEST_SIZE = 0.2
RANDOM_STATE = 42
METADATA_COLS_FOR_MODEL = ['view_position', 'breast_density', 'laterality']
TARGET_COL = 'breast_birads'
print(f"📂 Thư mục Output: {os.path.abspath(output_dir)}")

# =============================================================
# ===          2. ĐỌC, LỌC VÀ KẾT HỢP ANNOTATIONS          ===
# =============================================================
print("\n[Bước 1/10] Đọc, lọc và kết hợp annotations...")
start_step_time = time.time()
# Kiểm tra file
if not os.path.exists(breast_level_csv):
    print(f"❌ Lỗi: Không tìm thấy file '{breast_level_csv}'")
    exit()
if not os.path.exists(finding_level_csv):
    print(f"❌ Lỗi: Không tìm thấy file '{finding_level_csv}'")
    exit()
print(f"   - Tìm thấy '{os.path.basename(breast_level_csv)}'.")
print(f"   - Tìm thấy '{os.path.basename(finding_level_csv)}'.")
# Đọc file
try:
    df_breast = pd.read_csv(breast_level_csv)
    print(f"   - Đọc {len(df_breast)} dòng từ '{os.path.basename(breast_level_csv)}'.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file CSV '{breast_level_csv}': {e}")
    exit()
try:
    df_finding = pd.read_csv(finding_level_csv)
    print(f"   - Đọc {len(df_finding)} dòng từ '{os.path.basename(finding_level_csv)}'.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file CSV '{finding_level_csv}': {e}")
    exit()
# Kiểm tra cột khóa
missing_keys_breast = [col for col in KEY_COLUMNS if col not in df_breast.columns]
missing_keys_finding = [col for col in KEY_COLUMNS if col not in df_finding.columns]
if missing_keys_breast:
    print(f"❌ Lỗi: File '{os.path.basename(breast_level_csv)}' thiếu cột khóa: {missing_keys_breast}")
    exit()
if missing_keys_finding:
    print(f"❌ Lỗi: File '{os.path.basename(finding_level_csv)}' thiếu cột khóa: {missing_keys_finding}")
    exit()
print(f"   - Cột khóa {KEY_COLUMNS} tồn tại.")
# Xác định cột chung và lọc
common_columns = list(set(df_breast.columns) & set(df_finding.columns))
print(f"   - Xác định {len(common_columns)} cột chung.")
df_breast_common = df_breast[common_columns].copy()
df_finding_common = df_finding[common_columns].copy()
# Merge
print(f"   - Thực hiện 'inner' join trên {KEY_COLUMNS}...")
merged_df = pd.merge(df_breast_common, df_finding_common, on=KEY_COLUMNS, how='inner', suffixes=('_breast', '_finding'))
# Xử lý cột trùng tên (ưu tiên _breast)
processed_cols = set()
final_cols = []
for col in merged_df.columns:
    base_col = col.replace('_breast', '').replace('_finding', '')
    if base_col not in processed_cols:
        if f"{base_col}_breast" in merged_df.columns:
            merged_df[base_col] = merged_df[f"{base_col}_breast"]
        elif f"{base_col}_finding" in merged_df.columns:
            merged_df[base_col] = merged_df[f"{base_col}_finding"]
        elif base_col in merged_df.columns:
            pass
        else:
            continue
        final_cols.append(base_col)
        processed_cols.add(base_col)
merged_df = merged_df[final_cols]
print(f"   - Số dòng sau khi 'inner' join: {len(merged_df)}")
# Xử lý trùng lặp
initial_merged_rows = len(merged_df)
merged_df.drop_duplicates(subset=KEY_COLUMNS, keep='first', inplace=True)
duplicates_dropped = initial_merged_rows - len(merged_df)
if duplicates_dropped > 0:
    print(f"   - Đã loại bỏ {duplicates_dropped} bản ghi trùng lặp.")
print(f"   - Số dòng cuối cùng trong DataFrame kết hợp: {len(merged_df)}")
if merged_df.empty:
    print("❌ Lỗi: Không có dữ liệu chung. Kết thúc.")
    exit()
end_step_time = time.time()
print(f"✅ Kết hợp annotations hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
del df_breast, df_finding, df_breast_common, df_finding_common  # Giải phóng bộ nhớ
gc.collect()

# =============================================================
# ===     3. TẠO ĐƯỜNG DẪN ẢNH VÀ KIỂM TRA DỮ LIỆU        ===
# =============================================================
print("\n[Bước 2/10] Tạo đường dẫn ảnh và kiểm tra dữ liệu kết hợp...")
start_step_time = time.time()
# Kiểm tra cột cần thiết
if 'split' not in merged_df.columns:
    print(f"❌ Lỗi: Thiếu cột 'split'.")
    exit()
if 'image_id' not in merged_df.columns:
    print(f"❌ Lỗi: Thiếu cột 'image_id'.")
    exit()
# Hàm tạo đường dẫn
def create_image_path(row):
    """Tạo đường dẫn tương đối dựa trên split và image_id."""
    split_folder = train_dir_name if row['split'] == 'training' else test_dir_name
    image_filename = f"{row['image_id']}{IMAGE_EXTENSION}"
    path = os.path.join(split_folder, image_filename)
    return path
print(f"   - Tạo cột '{IMAGE_PATH_COL}' với đuôi file '{IMAGE_EXTENSION}'...")
merged_df[IMAGE_PATH_COL] = merged_df.apply(create_image_path, axis=1)
if merged_df[IMAGE_PATH_COL].isnull().any():
    print("   ⚠️ Cảnh báo: Có lỗi khi tạo đường dẫn ảnh.")
    merged_df.dropna(subset=[IMAGE_PATH_COL], inplace=True)
print(f"     Ví dụ đường dẫn: {merged_df[IMAGE_PATH_COL].iloc[0]}" if not merged_df.empty else "     (Không có dữ liệu)")
# Kiểm tra cột cần thiết cho model
required_cols_model = [IMAGE_PATH_COL] + METADATA_COLS_FOR_MODEL + [TARGET_COL]
missing_cols_model = [col for col in required_cols_model if col not in merged_df.columns]
if missing_cols_model:
    print(f"❌ Lỗi: DataFrame thiếu cột cho mô hình: {missing_cols_model}")
    exit()
print(f"   - Các cột cần thiết ({required_cols_model}) đã tồn tại.")
# Xử lý NaN chỉ trong các cột breast_birads và image_id
required_cols_model = ['breast_birads', 'image_id']  # Các cột bắt buộc
initial_rows = len(merged_df)
merged_df.dropna(subset=required_cols_model, inplace=True)
dropped_rows = initial_rows - len(merged_df)
if dropped_rows > 0:
    print(f"   - Đã loại bỏ {dropped_rows} dòng chứa giá trị thiếu trong các cột {required_cols_model}.")
print(f"   - Số dòng hợp lệ sau khi loại bỏ NaN: {len(merged_df)}")
if merged_df.empty:
    print("❌ Lỗi: Không còn dữ liệu hợp lệ.")
    exit()
# Mã hóa
print("   - Mã hóa metadata và nhãn...")
le_view = LabelEncoder()
le_laterality = LabelEncoder()
le_density = LabelEncoder()
le_birads = LabelEncoder()
try:
    merged_df['view_position_encoded'] = le_view.fit_transform(merged_df['view_position'])
    merged_df['laterality_encoded'] = le_laterality.fit_transform(merged_df['laterality'])
    if not pd.api.types.is_numeric_dtype(merged_df['breast_density']):
        merged_df['breast_density_encoded'] = le_density.fit_transform(merged_df['breast_density'])
    else:
        merged_df['breast_density_encoded'] = merged_df['breast_density']
    merged_df['birads_encoded'] = le_birads.fit_transform(merged_df[TARGET_COL])
    num_classes = len(le_birads.classes_)
    target_names = [str(cls) for cls in le_birads.classes_]
    XGB_PARAMS['num_class'] = num_classes  # Cập nhật tham số XGBoost
    print(f"     - Đã mã hóa các cột. Số lớp BI-RADS: {num_classes}")
except KeyError as ke:
    print(f"❌ Lỗi KeyError khi mã hóa cột: {ke}.")
    exit()
except Exception as e:
    print(f"❌ Lỗi không xác định khi mã hóa: {e}")
    exit()
end_step_time = time.time()
print(f"✅ Tiền xử lý dữ liệu kết hợp hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# =============================================================
# ===          4. THIẾT LẬP TRÍCH XUẤT ĐẶC TRƯNG           ===
# =============================================================
print("\n[Bước 3/10] Thiết lập mô hình trích xuất đặc trưng ảnh...")
start_step_time = time.time()
image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   - Sử dụng thiết bị: {str(device).upper()}")
print(f"   - Tải mô hình {MODEL_NAME} pretrained...")
try:
    cnn_model = None
    if MODEL_NAME == 'resnet50':
        cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        cnn_model.fc = nn.Identity()
    elif MODEL_NAME == 'resnet101':
        cnn_model = models.resnet101(weights=models.ResNet101_Weights.IMAGEN1K_V2)
        cnn_model.fc = nn.Identity()
    else:
        print(f"❌ Lỗi: Model '{MODEL_NAME}' không được hỗ trợ.")
        exit()
    if cnn_model:
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
    else:
        print("❌ Lỗi: Không thể khởi tạo cnn_model.")
        exit()
    end_step_time = time.time()
    print(f"✅ Tải và cấu hình {MODEL_NAME} thành công (Thời gian: {end_step_time - start_step_time:.2f}s).")
except Exception as e:
    print(f"❌ Lỗi khi tải/cấu hình CNN: {e}")
    exit()

# =============================================================
# ===             5. TRÍCH XUẤT ĐẶC TRƯNG ẢNH              ===
# =============================================================
augmented_images_dir = "augmented_images_xg" 
os.makedirs(augmented_images_dir, exist_ok=True)
LABEL_COL = 'birads_encoded'
print("\n[Bước 4/10] Bắt đầu trích xuất đặc trưng ảnh (augment nhãn 2,3,4 để cân bằng với nhãn 1)...")
start_step_time = time.time()
light_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
])
strong_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
])
def extract_feature_from_pil(img):
    img_tensor = image_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = cnn_model(img_tensor).squeeze().cpu().numpy()
    if np.isnan(feature_vector).any():
        return None
    return feature_vector
features_list = []
valid_indices_processed = []
augmentation_flags = []
skipped_files_count = 0
augment_image_count = 0
label_counts = merged_df[LABEL_COL].value_counts()
print(f"🔎 Số lượng ban đầu: {dict(label_counts)}")
augment_labels = [2, 3, 4]
target_per_label =8000
current_augmented_counts = {label: label_counts.get(label, 0) for label in augment_labels}
print(f"🎯 Mục tiêu số lượng cho mỗi nhãn: {target_per_label}")
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="   Trích xuất", unit="ảnh"):
    img_rel_path = row['image_path']
    img_full_path = os.path.join(image_base_dir, img_rel_path)
    label = row[LABEL_COL]
    try:
        img = Image.open(img_full_path).convert('RGB')
        feat = extract_feature_from_pil(img)
        if feat is not None and feat.shape == (FEATURE_VECTOR_SIZE,):
            features_list.append(feat)
            valid_indices_processed.append(idx)
            augmentation_flags.append(False)
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
                        orig_filename = os.path.splitext(os.path.basename(img_rel_path))[0]
                        aug_filename = f"{orig_filename}_aug{augment_image_count}.jpg"
                        aug_save_path = os.path.join(augmented_images_dir, aug_filename)
                        aug_img.save(aug_save_path)
                    else:
                        break
        else:
            skipped_files_count += 1
    except (FileNotFoundError, UnidentifiedImageError):
        skipped_files_count += 1
    except Exception as e:
        skipped_files_count += 1
end_step_time = time.time()
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
df_final = merged_df.loc[valid_indices_processed].reset_index(drop=True)
image_features = np.stack(features_list)
print(f"   - Kích thước mảng đặc trưng ảnh cuối cùng: {image_features.shape}")
try:
    df_final.to_csv(output_processed_csv_path, index=False)
    print(f"💾 Đã lưu DataFrame đã xử lý vào: {output_processed_csv_path}")
except Exception as e:
    print(f"⚠️ Lỗi khi lưu file CSV đã xử lý: {e}")
del merged_df, features_list
gc.collect()
torch.cuda.empty_cache() if device == 'cuda' else None

# =============================================================
# ===       6. KẾT HỢP ĐẶC TRƯNG CUỐI & CHIA DỮ LIỆU       ===
# =============================================================
print("\n[Bước 5/10] Kết hợp đặc trưng ảnh và metadata cuối cùng...")
start_step_time = time.time()
try:
    metadata_features_final = df_final[['view_position_encoded', 'breast_density_encoded', 'laterality_encoded']].values
    scaler = StandardScaler()
    metadata_scaled_final = scaler.fit_transform(metadata_features_final)
    print("   - Đã áp dụng StandardScaler cho metadata.")
    X = np.hstack([image_features, metadata_scaled_final])
    y = df_final['birads_encoded'].values
    print(f"   - Kích thước mảng đặc trưng kết hợp (X): {X.shape}")
    print(f"   - Kích thước mảng nhãn (y): {y.shape}")
except KeyError as ke:
    print(f"❌ Lỗi KeyError khi lấy cột metadata: {ke}. Kiểm tra lại METADATA_COLS_FOR_MODEL.")
    exit()
except Exception as e:
    print(f"❌ Lỗi khi kết hợp đặc trưng: {e}")
    exit()
end_step_time = time.time()
print(f"✅ Kết hợp đặc trưng hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
print(f"\n[Bước 6/10] Chia dữ liệu thành tập huấn luyện và kiểm tra...")
start_step_time = time.time()
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"   - Tập huấn luyện: X={X_train.shape}, y={y_train.shape}")
    print(f"   - Tập kiểm tra:  X={X_test.shape}, y={y_test.shape}")
except Exception as e:
    print(f"❌ Lỗi khi chia dữ liệu: {e}")
    exit()
end_step_time = time.time()
print(f"✅ Chia dữ liệu hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
print("   - Tạo DMatrix cho XGBoost...")
try:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    print("   - Tạo DMatrix thành công.")
except Exception as e:
    print(f"❌ Lỗi khi tạo DMatrix: {e}")
    exit()

# =============================================================
# ===            7. HUẤN LUYỆN VÀ LƯU MÔ HÌNH              ===
# =============================================================
print("\n[Bước 7/10] Huấn luyện mô hình XGBoost (Core API) và Lưu mô hình...")
start_step_time = time.time()
print("   - Chuẩn bị tham số và huấn luyện bằng xgb.train...")
params = XGB_PARAMS.copy()  # Đã cập nhật num_class ở Bước 2
evals_result = {}
bst = None  # Khởi tạo bst
try:
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=False
    )
except xgb.core.XGBoostError as xgb_err:
    print(f"❌ Lỗi XGBoost khi huấn luyện: {xgb_err}")
    exit()
except Exception as e:
    print(f"❌ Lỗi không xác định khi huấn luyện: {e}")
    exit()
end_step_time = time.time()
print(f"✅ Huấn luyện XGBoost hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
# Lấy số iteration tốt nhất một cách an toàn
best_iteration = getattr(bst, 'best_iteration', -1)  # Dùng getattr để tránh AttributeError
optimal_num_trees = NUM_BOOST_ROUND
if best_iteration >= 0:
    optimal_num_trees = best_iteration + 1
    print(f"   - Số lượng cây tối ưu (best_iteration + 1): {optimal_num_trees} / {NUM_BOOST_ROUND}")
else:
    print(f"   - Early stopping không kích hoạt/lỗi, dùng {NUM_BOOST_ROUND} cây.")
# Vẽ Biểu đồ Huấn luyện
print(f"   - Vẽ và lưu biểu đồ hiệu suất huấn luyện: {output_training_plot_path}")
fig_train = None
try:
    metric_name = params['eval_metric']
    if evals_result and 'train' in evals_result and 'test' in evals_result and \
       metric_name in evals_result['train'] and metric_name in evals_result['test']:
        train_metric_history = evals_result['train'][metric_name]
        test_metric_history = evals_result['test'][metric_name]
        epochs = len(train_metric_history)
        x_axis = range(0, epochs)
        fig_train, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, train_metric_history, label=f'Train {metric_name}')
        ax.plot(x_axis, test_metric_history, label=f'Test {metric_name}')
        if best_iteration >= 0:
            ax.axvline(best_iteration, color='r', linestyle='--', label=f'Best Iteration ({optimal_num_trees})')
        ax.legend()
        plt.ylabel(metric_name.capitalize())
        plt.xlabel('Boosting Iterations (Trees)')
        plt.title(f'XGBoost {metric_name.capitalize()} during Training')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_training_plot_path)
        print("     💾 Biểu đồ huấn luyện đã lưu.")
    else:
        print(f"     ⚠️ Không đủ dữ liệu trong evals_result cho metric '{metric_name}' để vẽ biểu đồ.")
except KeyError as ke:
    print(f"     ⚠️ Lỗi KeyError khi truy cập evals_result (metric '{metric_name}'): {ke}")
except Exception as e:
    print(f"     ⚠️ Lỗi khi vẽ/lưu biểu đồ huấn luyện: {e}")
finally:
    plt.close(fig_train) if fig_train else None
# Lưu Model
print(f"   - Lưu mô hình (Booster) vào: {output_model_path}")
try:
    bst.save_model(output_model_path)
    print("     💾 Lưu mô hình thành công.")
except Exception as e:
    print(f"     ⚠️ Lỗi khi lưu mô hình Booster: {e}")

# =============================================================
# ===          8. ĐÁNH GIÁ CHI TIẾT MÔ HÌNH                ===
# =============================================================
print("\n[Bước 8/10] Đánh giá chi tiết mô hình...")
start_step_time = time.time()
# Dự đoán bằng Booster
y_pred = None
y_pred_proba = None
metrics_calculated = False
if bst is None:  # Kiểm tra nếu huấn luyện thất bại
    print("❌ Lỗi: Mô hình (bst) không được huấn luyện thành công, không thể đánh giá.")
else:
    try:
        limit = optimal_num_trees  # Dùng số cây đã xác định
        print(f"   - Dự đoán xác suất bằng Booster với giới hạn cây: {limit}")
        y_pred_proba = bst.predict(dtest, iteration_range=(0, limit))  # Dùng iteration_range thay vì ntree_limit
        if y_pred_proba is not None and y_pred_proba.ndim == 2 and y_pred_proba.shape[0] == dtest.num_row() and y_pred_proba.shape[1] == num_classes:
            y_pred = np.argmax(y_pred_proba, axis=1)
            print("   - Đã lấy nhãn lớp dự đoán từ xác suất.")
            metrics_calculated = True
        else:
            print(f"   ❌ Lỗi: Output dự đoán xác suất có shape không mong muốn: {y_pred_proba.shape if y_pred_proba is not None else 'None'}")
            print(f"     (Kỳ vọng: ({dtest.num_row()}, {num_classes}))")
    except xgb.core.XGBoostError as pred_e:
        print(f"❌ Lỗi XGBoost khi dự đoán: {pred_e}")
    except Exception as pred_e:
        print(f"❌ Lỗi không xác định khi dự đoán: {pred_e}")
# Tính Metrics nếu dự đoán thành công
accuracy = -1.0
report_str = "Lỗi dự đoán/lấy nhãn"
cm = np.array([[-1]])
cm_df = pd.DataFrame()
if metrics_calculated:
    try:
        y_true_labels = dtest.get_label().astype(int)
        accuracy = accuracy_score(y_true_labels, y_pred)
        report_str = classification_report(y_true_labels, y_pred, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_true_labels, y_pred, labels=range(num_classes))
        cm_df = pd.DataFrame(cm, index=le_birads.classes_, columns=le_birads.classes_)
    except Exception as metric_e:
        print(f"❌ Lỗi khi tính toán metrics: {metric_e}")
        metrics_calculated = False
        accuracy = -1.0
        report_str = "Lỗi khi tính metrics."
        cm_df = pd.DataFrame()
# In Kết quả
print(f"\n   --- Kết quả đánh giá trên tập kiểm tra ---")
print(f"   - Độ chính xác (Accuracy): {accuracy:.4f}" if metrics_calculated else "   - Độ chính xác (Accuracy): Lỗi")
if metrics_calculated:
    print("\n   - Báo cáo phân loại chi tiết (Classification Report):")
    print(report_str)
    print("\n   - Ma trận nhầm lẫn (Confusion Matrix):")
    print(cm_df)
else:
    print("\n   - Không có báo cáo và ma trận nhầm lẫn do lỗi.")
# Vẽ Biểu đồ Đánh giá chỉ khi metrics được tính VÀ có xác suất
plot_evaluation_charts = metrics_calculated and (y_pred_proba is not None)
if plot_evaluation_charts:
    # Heatmap CM
    print(f"\n   - Vẽ và lưu heatmap ma trận nhầm lẫn: {output_cm_plot_path}")
    fig_cm = None
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 12})
        ax_cm.set_xlabel('Nhãn Dự đoán', fontsize=12)
        ax_cm.set_ylabel('Nhãn Thực', fontsize=12)
        ax_cm.set_title('Heatmap Ma trận Nhầm lẫn', fontsize=14)
        ax_cm.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig(output_cm_plot_path)
        print("     💾 Heatmap đã lưu.")
    except Exception as e:
        print(f"     ⚠️ Lỗi khi vẽ/lưu heatmap: {e}")
    finally:
        plt.close(fig_cm) if fig_cm else None
    # ROC Curve
    print(f"\n   - Vẽ và lưu đường cong ROC (One-vs-Rest): {output_roc_plot_path}")
    fig_roc = None
    try:
        y_test_binarized = label_binarize(y_true_labels, classes=range(num_classes))
        assert y_test_binarized.shape[1] >= 2, "Cần ít nhất 2 lớp để vẽ ROC."
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax_roc.plot(fpr[i], tpr[i], lw=2, label=f'ROC lớp {target_names[i]} (AUC = {roc_auc[i]:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Ngẫu nhiên')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('Tỷ lệ Dương tính Giả')
        ax_roc.set_ylabel('Tỷ lệ Dương tính Thực')
        ax_roc.set_title('Đường cong ROC - One-vs-Rest')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        plt.tight_layout()
        plt.savefig(output_roc_plot_path)
        print("     💾 Biểu đồ ROC đã lưu.")
    except AssertionError as ae:
        print(f"     ⚠️ Lỗi khi vẽ ROC: {ae}")
    except ValueError as ve:
        print(f"     ⚠️ Lỗi ValueError khi vẽ ROC: {ve}")
    except Exception as e:
        print(f"     ⚠️ Lỗi không xác định khi vẽ/lưu ROC: {e}")
    finally:
        plt.close(fig_roc) if fig_roc else None
    # PR Curve
    print(f"\n   - Vẽ và lưu đường cong Precision-Recall (One-vs-Rest): {output_pr_plot_path}")
    fig_pr = None
    try:
        assert y_test_binarized.shape[1] >= 2, "Cần ít nhất 2 lớp để vẽ PR."
        precision = dict()
        recall = dict()
        average_precision = dict()
        fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])
            ax_pr.plot(recall[i], precision[i], lw=2, label=f'PR lớp {target_names[i]} (AP = {average_precision[i]:.2f})')
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Độ Nhớ lại (Recall)')
        ax_pr.set_ylabel('Độ Chính xác (Precision)')
        ax_pr.set_title('Đường cong Precision-Recall - One-vs-Rest')
        ax_pr.legend(loc="best")
        ax_pr.grid(True)
        plt.tight_layout()
        plt.savefig(output_pr_plot_path)
        print("     💾 Biểu đồ PR đã lưu.")
    except AssertionError as ae:
        print(f"     ⚠️ Lỗi khi vẽ PR: {ae}")
    except ValueError as ve:
        print(f"     ⚠️ Lỗi ValueError khi vẽ PR: {ve}")
    except Exception as e:
        print(f"     ⚠️ Lỗi không xác định khi vẽ/lưu PR: {e}")
    finally:
        plt.close(fig_pr) if fig_pr else None
else:
    print("\n   ⚠️ Bỏ qua vẽ biểu đồ đánh giá do lỗi hoặc thiếu xác suất.")
end_step_time = time.time()
print(f"✅ Đánh giá chi tiết hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# =============================================================
# ===            9. GIẢI THÍCH LIME                        ===
# =============================================================
print("\n[Bước 9/10] Giải thích mô hình với LIME...")
start_step_time = time.time()
lime_plot_paths = []
try:
    # Hàm dự đoán cho LIME, tương thích với Booster
    def predict_fn(X):
        dmatrix = xgb.DMatrix(X)
        return bst.predict(dmatrix, iteration_range=(0, optimal_num_trees))
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality'],
        class_names=target_names,
        mode="classification"
    )
    target_num_samples = 10
    samples_per_label = target_num_samples // num_classes
    selected_indices = []
    selected_labels = []
    for label in range(num_classes):
        label_indices = np.where(y_test == label)[0]
        if len(label_indices) == 0:
            print(f"     ⚠️ Cảnh báo: Không có mẫu nào cho nhãn {target_names[label]}.")
            continue
        num_to_select = min(samples_per_label, len(label_indices))
        selected = np.random.choice(label_indices, size=num_to_select, replace=False)
        selected_indices.extend(selected)
        selected_labels.extend([label] * num_to_select)
    current_num_samples = len(selected_indices)
    if current_num_samples < target_num_samples:
        remaining_needed = target_num_samples - current_num_samples
        all_indices = np.arange(len(y_test))
        remaining_indices = np.setdiff1d(all_indices, selected_indices)
        if len(remaining_indices) < remaining_needed:
            print(f"     ⚠️ Cảnh báo: Chỉ có {len(remaining_indices)} mẫu còn lại, không đủ để chọn {remaining_needed} mẫu.")
            selected_indices.extend(remaining_indices)
            selected_labels.extend(y_test[remaining_indices].tolist())
        else:
            extra_selected = np.random.choice(remaining_indices, size=remaining_needed, replace=False)
            selected_indices.extend(extra_selected)
            selected_labels.extend(y_test[extra_selected].tolist())
    print(f"   - Số mẫu được chọn cho LIME: {len(selected_indices)} (mục tiêu: {target_num_samples})")
    print(f"   - Phân bố nhãn được chọn: {dict(pd.Series(selected_labels).value_counts())}")
    for i, sample_idx in enumerate(selected_indices[:target_num_samples]):
        sample = X_test[sample_idx]
        true_label = target_names[y_test[sample_idx]]
        explanation = explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn,
            num_features=10
        )
        print(f"   - Giải thích LIME cho mẫu {i} - Nhãn: {true_label}")
        fig = explanation.as_pyplot_figure()
        plt.title(f"Giải thích LIME - Mẫu {i} - Nhãn: {true_label}")
        plt.tight_layout()
        lime_plot_path = os.path.join(output_dir, f"lime_explanation_label{selected_labels[i]}_sample{i}.png")
        plt.savefig(lime_plot_path)
        plt.close(fig)
        lime_plot_paths.append(lime_plot_path)
        print(f"     💾 Lưu biểu đồ LIME: {lime_plot_path}")
except Exception as e:
    print(f"❌ Lỗi khi tạo giải thích LIME: {e}")
end_step_time = time.time()
print(f"✅ Giải thích LIME hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# =============================================================
# ===            10. GIẢI THÍCH SHAP                       ===
# =============================================================
print("\n[Bước 10/10] Giải thích mô hình với SHAP...")
plt.switch_backend('agg')
start_step_time = time.time()
shap_plot_path = ""
shap_waterfall_paths = []
try:
    print("   - Khởi tạo TreeExplainer với feature_perturbation='interventional'...")
    explainer = shap.TreeExplainer(
        bst,
        feature_perturbation='interventional',
        data=X_train[:100]  # Sử dụng tập nền nhỏ
    )
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    print(f"   - Kích thước X_test: {X_test.shape}")
    print(f"   - Kích thước y_test: {y_test.shape}")
    if len(X_test) != len(y_test):
        raise ValueError(f"Kích thước không khớp: X_test ({len(X_test)}) ≠ y_test ({len(y_test)})")
    # Gỡ lỗi: Kiểm tra đầu ra mô hình so với tổng giá trị SHAP
    sample_idx = 0
    dmatrix_sample = xgb.DMatrix(X_test[sample_idx:sample_idx+1])
    model_output = bst.predict(dmatrix_sample, iteration_range=(0, optimal_num_trees))[0]
    shap_sum = np.sum(shap_values, axis=1)[sample_idx]
    print(f"   - Gỡ lỗi (mẫu 0): Đầu ra mô hình = {model_output}, Tổng giá trị SHAP = {shap_sum}")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality'],
        plot_type="bar",
        class_names=target_names,
        color_bar=False,
        max_display=20,
        show=False
    )
    plt.title(f"Biểu đồ Tóm tắt SHAP cho Mô hình XGBoost ({len(X_test)} Mẫu Kiểm tra)", fontsize=14, pad=20)
    plt.xlabel("Trung bình(|Giá trị SHAP|) [tác động trung bình đến đầu ra mô hình]", fontsize=12)
    plt.ylabel("Đặc trưng", fontsize=12)
    plt.tight_layout()
    shap_plot_path = os.path.join(output_dir, "shap_summary_model.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     💾 Lưu biểu đồ SHAP tóm tắt toàn bộ mô hình: {shap_plot_path}")
    target_num_samples = 5
    valid_range = np.arange(min(2050, len(X_test)))
    selected_indices = np.random.choice(valid_range, size=target_num_samples, replace=False)
    selected_indices = [i for i in selected_indices if i < len(X_test)]
    if not selected_indices:
        raise ValueError("Không có chỉ số hợp lệ nào được chọn cho biểu đồ SHAP waterfall.")
    print(f"   - Số mẫu được chọn cho SHAP waterfall: {len(selected_indices)} (chỉ số: {selected_indices})")
    for i, sample_idx in enumerate(selected_indices):
        if sample_idx >= len(X_test):
            print(f"     ⚠️ Cảnh báo: Chỉ số {sample_idx} vượt quá kích thước X_test ({len(X_test)}). Bỏ qua mẫu này.")
            continue
        true_label = target_names[y_test[sample_idx]]
        predicted_label = target_names[y_pred[sample_idx]] if y_pred is not None else "N/A"
        pred_class = y_pred[sample_idx] if y_pred is not None else np.argmax(bst.predict(xgb.DMatrix(X_test[sample_idx:sample_idx+1]), iteration_range=(0, optimal_num_trees))[0])
        shap_explanation = shap.Explanation(
            values=shap_values[pred_class][sample_idx],
            base_values=explainer.expected_value[pred_class],
            data=X_test[sample_idx],
            feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality']
        )
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(
            shap_explanation,
            max_display=10,
            show=False
        )
        plt.title(f"Biểu đồ SHAP Waterfall - Mẫu {i} - Thực: {true_label}, Dự đoán: {predicted_label}", fontsize=12, pad=10)
        plt.xlabel("Giá trị SHAP", fontsize=10)
        plt.ylabel("Đặc trưng", fontsize=10)
        plt.tight_layout()
        waterfall_plot_path = os.path.join(output_dir, f"shap_waterfall_sample{i}.png")
        plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        shap_waterfall_paths.append(waterfall_plot_path)
        print(f"     💾 Lưu biểu đồ SHAP waterfall cho mẫu {i}: {waterfall_plot_path}")
except Exception as tree_e:
    print(f"❌ Lỗi TreeExplainer: {tree_e}")
    print("   - Chuyển sang dùng KernelExplainer làm phương án dự phòng...")
    try:
        background_size = min(100, X_train.shape[0])
        background_indices = np.random.choice(X_train.shape[0], size=background_size, replace=False)
        X_background = X_train[background_indices]
        def predict_fn_kernel(X):
            dmatrix = xgb.DMatrix(X)
            return bst.predict(dmatrix, iteration_range=(0, optimal_num_trees))
        explainer = shap.KernelExplainer(predict_fn_kernel, X_background)
        shap_values = explainer.shap_values(X_test, nsamples=100)
        print(f"   - Kích thước X_test: {X_test.shape}")
        print(f"   - Kích thước y_test: {y_test.shape}")
        if len(X_test) != len(y_test):
            raise ValueError(f"Kích thước không khớp: X_test ({len(X_test)}) ≠ y_test ({len(y_test)})")
        sample_idx = 0
        dmatrix_sample = xgb.DMatrix(X_test[sample_idx:sample_idx+1])
        model_output = bst.predict(dmatrix_sample, iteration_range=(0, optimal_num_trees))[0]
        shap_sum = np.sum(shap_values, axis=1)[sample_idx]
        print(f"   - Gỡ lỗi (mẫu 0): Đầu ra mô hình = {model_output}, Tổng giá trị SHAP = {shap_sum}")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality'],
            plot_type="bar",
            class_names=target_names,
            color_bar=False,
            max_display=20,
            show=False
        )
        plt.title(f"Biểu đồ Tóm tắt SHAP cho Mô hình XGBoost ({len(X_test)} Mẫu Kiểm tra)", fontsize=14, pad=20)
        plt.xlabel("Trung bình(|Giá trị SHAP|) [tác động trung bình đến đầu ra mô hình]", fontsize=12)
        plt.ylabel("Đặc trưng", fontsize=12)
        plt.tight_layout()
        shap_plot_path = os.path.join(output_dir, "shap_summary_model_kernel.png")
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     💾 Lưu biểu đồ SHAP tóm tắt toàn bộ mô hình (KernelExplainer): {shap_plot_path}")
        target_num_samples = 5
        valid_range = np.arange(min(2050, len(X_test)))
        selected_indices = np.random.choice(valid_range, size=target_num_samples, replace=False)
        selected_indices = [i for i in selected_indices if i < len(X_test)]
        if not selected_indices:
            raise ValueError("Không có chỉ số hợp lệ nào được chọn cho biểu đồ SHAP waterfall.")
        print(f"   - Số mẫu được chọn cho SHAP waterfall: {len(selected_indices)} (chỉ số: {selected_indices})")
        for i, sample_idx in enumerate(selected_indices):
            if sample_idx >= len(X_test):
                print(f"     ⚠️ Cảnh báo: Chỉ số {sample_idx} vượt quá kích thước X_test ({len(X_test)}). Bỏ qua mẫu này.")
                continue
            true_label = target_names[y_test[sample_idx]]
            predicted_label = target_names[y_pred[sample_idx]] if y_pred is not None else "N/A"
            pred_class = y_pred[sample_idx] if y_pred is not None else np.argmax(bst.predict(xgb.DMatrix(X_test[sample_idx:sample_idx+1]), iteration_range=(0, optimal_num_trees))[0])
            shap_explanation = shap.Explanation(
                values=shap_values[pred_class][sample_idx],
                base_values=explainer.expected_value[pred_class],
                data=X_test[sample_idx],
                feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality']
            )
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(
                shap_explanation,
                max_display=10,
                show=False
            )
            plt.title(f"Biểu đồ SHAP Waterfall - Mẫu {i} - Thực: {true_label}, Dự đoán: {predicted_label}", fontsize=12, pad=10)
            plt.xlabel("Giá trị SHAP", fontsize=10)
            plt.ylabel("Đặc trưng", fontsize=10)
            plt.tight_layout()
            waterfall_plot_path = os.path.join(output_dir, f"shap_waterfall_sample{i}_kernel.png")
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            shap_waterfall_paths.append(waterfall_plot_path)
            print(f"     💾 Lưu biểu đồ SHAP waterfall cho mẫu {i} (KernelExplainer): {waterfall_plot_path}")
    except Exception as kernel_e:
        print(f"❌ Lỗi KernelExplainer: {kernel_e}")
        print("   - Bỏ qua SHAP do lỗi liên tục. Vui lòng kiểm tra dữ liệu hoặc báo cáo vấn đề trên GitHub SHAP.")
end_step_time = time.time()
print(f"✅ Giải thích SHAP hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# =============================================================
# ===          LƯU KẾT QUẢ TEXT                           ===
# =============================================================
print("\n[Lưu Kết Quả] Lưu kết quả đánh giá vào file text...")
start_step_time = time.time()
try:
    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time_main
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("="*53 + "\n")
        f.write("===       KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI BI-RADS       ===\n")
        f.write("="*53 + "\n\n")
        f.write(f"Thời gian chạy tổng cộng script: {total_script_duration:.2f} giây ({total_script_duration/60:.2f} phút)\n\n")
        f.write(f"Cấu hình chính:\n")
        f.write(f"  - Model CNN: {MODEL_NAME}\n")
        f.write(f"  - Kích thước ảnh: {IMAGE_SIZE}\n")
        f.write(f"  - Tham số XGBoost Core API: {json.dumps(params, indent=4)}\n")
        f.write(f"  - Số cây tối đa (num_boost_round): {NUM_BOOST_ROUND}\n")
        f.write(f"  - Early Stopping Rounds: {EARLY_STOPPING_ROUNDS}\n")
        f.write(f"  - Kích thước tập kiểm tra: {TEST_SIZE:.1%}\n")
        f.write(f"  - Số lớp BI-RADS: {num_classes}\n")
        f.write(f"  - Số lượng cây tối ưu (best iteration + 1): {optimal_num_trees}\n")
        f.write("-"*53 + "\n\n")
        if metrics_calculated:
            f.write(f"Độ chính xác (Accuracy): {accuracy:.4f}\n\n")
            f.write("Báo cáo phân loại chi tiết (Classification Report):\n")
            f.write(report_str)
            f.write("\n\nMa trận nhầm lẫn (Confusion Matrix):\n")
            f.write(cm_df.to_string())
            f.write("\n\n")
            if plot_evaluation_charts:
                f.write("Các biểu đồ đánh giá:\n")
                f.write(f"  - Biểu đồ hiệu suất huấn luyện: {output_training_plot_path}\n")
                f.write(f"  - Ma trận nhầm lẫn: {output_cm_plot_path}\n")
                f.write(f"  - Đường cong ROC: {output_roc_plot_path}\n")
                f.write(f"  - Đường cong Precision-Recall: {output_pr_plot_path}\n")
            else:
                f.write("Các biểu đồ (Loss, CM) đã được lưu. ROC/PR không vẽ do lỗi hoặc thiếu xác suất.\n")
            f.write("\nGiải thích LIME ({len(lime_plot_paths)} biểu đồ):\n")
            if lime_plot_paths:
                for path in lime_plot_paths:
                    f.write(f"  - Biểu đồ LIME: {path}\n")
            else:
                f.write("  - Không có biểu đồ LIME do lỗi.\n")
            f.write("\nGiải thích SHAP:\n")
            if shap_plot_path:
                f.write(f"  - Biểu đồ SHAP tóm tắt toàn bộ mô hình: {shap_plot_path}\n")
            else:
                f.write("  - Không có biểu đồ SHAP tóm tắt do lỗi.\n")
            f.write(f"  - Biểu đồ SHAP waterfall ({len(shap_waterfall_paths)} biểu đồ):\n")
            if shap_waterfall_paths:
                for path in shap_waterfall_paths:
                    f.write(f"    - Biểu đồ SHAP waterfall: {path}\n")
            else:
                f.write("    - Không có biểu đồ SHAP waterfall do lỗi.\n")
        else:
            f.write("!!! LỖI TRONG QUÁ TRÌNH DỰ ĐOÁN HOẶC TÍNH TOÁN METRICS !!!\n")
            f.write("   Vui lòng kiểm tra lại log lỗi trên màn hình.\n")
        f.write("="*53 + "\n")
    print(f"     💾 Lưu báo cáo thành công vào: {output_report_path}")
except Exception as e:
    print(f"     ⚠️ Lỗi khi lưu file báo cáo: {e}")
end_step_time = time.time()
print(f"✅ Lưu kết quả hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
print("\n-------------------------------------------------------------")
print("--- Quy trình Kết hợp và Phân loại BI-RADS Hoàn tất ---")
print("-------------------------------------------------------------")