# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time
import warnings
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import psutil
import logging
from torch.utils.data import Dataset, DataLoader
import uuid

# Configure logging
logging.basicConfig(
    filename='classification_output/script.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Script started")

# Ghi lại thời gian bắt đầu
script_start_time_main = time.time()

# Bật debug CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.info("-------------------------------------------------------------")
logging.info("--- Script Kết hợp Annotations và Phân loại BI-RADS v5 (Multiple Algorithms) ---")
logging.info("-------------------------------------------------------------")

# Kiểm tra tài nguyên hệ thống
logging.info(f"Bộ nhớ RAM khả dụng: {psutil.virtual_memory().available / (1024**3):.2f} GB")
if torch.cuda.is_available():
    logging.info(f"VRAM khả dụng: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")

# =============================================================
# ===                1. THIẾT LẬP CẤU HÌNH                 ===
# =============================================================
breast_level_csv = 'them data/breast-level_annotations.csv'
finding_level_csv = 'them data/finding_annotations.csv'
image_base_dir = 'orr'
train_dir_name = "train"
test_dir_name = "test"
IMAGE_EXTENSION = ".png"
BASE_OUTPUT_DIR = "classification_output"
SAVE_AUGMENTED_IMAGES = False  # Toggle to save augmented images
ALGORITHMS = {
    'dt': {
        'name': 'DecisionTree',
        'model': DecisionTreeClassifier,
        'params': {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42}
    },
    'rf': {
        'name': 'RandomForest',
        'model': RandomForestClassifier,
        'params': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    },
    'xgb': {
        'name': 'XGBoost',
        'model': XGBClassifier,
        'params': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'eval_metric': 'mlogloss'}
    },
    'nb': {
        'name': 'NaiveBayes',
        'model': GaussianNB,
        'params': {'var_smoothing': 1e-9}
    },
    'svm': {
        'name': 'SVM',
        'model': SVC,
        'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42}
    },
    'cnn': {
        'name': 'CNN',
        'model': None,  # Will use SimpleCNN class
        'params': {'input_dim': None, 'num_classes': None}  # Set dynamically
    }
}
KEY_COLUMNS = ['image_id']
IMAGE_PATH_COL = 'image_path'
MODEL_NAME = 'resnet50'
FEATURE_VECTOR_SIZE = 2048
IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2
RANDOM_STATE = 42
METADATA_COLS_FOR_MODEL = ['view_position', 'breast_density', 'laterality']
TARGET_COL = 'breast_birads'

# Đặt seed để đảm bảo tính nhất quán
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Validate image directory
if not os.path.exists(image_base_dir):
    logging.error(f"Thư mục ảnh '{image_base_dir}' không tồn tại.")
    print(f"❌ Lỗi: Thư mục ảnh '{image_base_dir}' không tồn tại.")
    exit()

# Tạo thư mục đầu ra
for algo_key in ALGORITHMS.keys():
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"{algo_key}_lime_shap")
    os.makedirs(output_dir, exist_ok=True)
    ALGORITHMS[algo_key]['output_dir'] = output_dir
    ALGORITHMS[algo_key]['output_processed_csv_path'] = os.path.join(output_dir, f"final_processed_annotations_{algo_key}.csv")
    ALGORITHMS[algo_key]['output_model_path'] = os.path.join(output_dir, f"{algo_key}_birads_model.pkl")
    ALGORITHMS[algo_key]['output_report_path'] = os.path.join(output_dir, f"classification_results_{algo_key}.txt")
    ALGORITHMS[algo_key]['output_cm_plot_path'] = os.path.join(output_dir, f"confusion_matrix_heatmap_{algo_key}.png")
    ALGORITHMS[algo_key]['output_roc_plot_path'] = os.path.join(output_dir, f"roc_curves_ovr_{algo_key}.png")
    ALGORITHMS[algo_key]['output_pr_plot_path'] = os.path.join(output_dir, f"precision_recall_curves_ovr_{algo_key}.png")

# Lưu cấu hình
with open(os.path.join(BASE_OUTPUT_DIR, "config.json"), 'w') as f:
    json.dump({
        'ALGORITHMS': {k: v['params'] for k, v in ALGORITHMS.items()},
        'IMAGE_SIZE': IMAGE_SIZE,
        'TEST_SIZE': TEST_SIZE,
        'RANDOM_STATE': RANDOM_STATE
    }, f, indent=4)
logging.info(f"Đã lưu cấu hình vào {os.path.join(BASE_OUTPUT_DIR, 'config.json')}")

# =============================================================
# ===          2. ĐỌC, LỌC VÀ KẾT HỢP ANNOTATIONS          ===
# =============================================================
logging.info("[Bước 1/10] Đọc, lọc và kết hợp annotations...")
start_step_time = time.time()
if not os.path.exists(breast_level_csv):
    logging.error(f"Không tìm thấy file '{breast_level_csv}'")
    print(f"❌ Lỗi: Không tìm thấy file '{breast_level_csv}'")
    exit()
if not os.path.exists(finding_level_csv):
    logging.error(f"Không tìm thấy file '{finding_level_csv}'")
    print(f"❌ Lỗi: Không tìm thấy file '{finding_level_csv}'")
    exit()
try:
    df_breast = pd.read_csv(breast_level_csv)
    logging.info(f"Đọc {len(df_breast)} dòng từ '{os.path.basename(breast_level_csv)}'.")
except Exception as e:
    logging.error(f"Lỗi khi đọc file CSV '{breast_level_csv}': {e}")
    print(f"❌ Lỗi khi đọc file CSV '{breast_level_csv}': {e}")
    exit()
try:
    df_finding = pd.read_csv(finding_level_csv)
    logging.info(f"Đọc {len(df_finding)} dòng từ '{os.path.basename(finding_level_csv)}'.")
except Exception as e:
    logging.error(f"Lỗi khi đọc file CSV '{finding_level_csv}': {e}")
    print(f"❌ Lỗi khi đọc file CSV '{finding_level_csv}': {e}")
    exit()
missing_keys_breast = [col for col in KEY_COLUMNS if col not in df_breast.columns]
missing_keys_finding = [col for col in KEY_COLUMNS if col not in df_finding.columns]
if missing_keys_breast:
    logging.error(f"File '{os.path.basename(breast_level_csv)}' thiếu cột khóa: {missing_keys_breast}")
    print(f"❌ Lỗi: File '{os.path.basename(breast_level_csv)}' thiếu cột khóa: {missing_keys_breast}")
    exit()
if missing_keys_finding:
    logging.error(f"File '{os.path.basename(finding_level_csv)}' thiếu cột khóa: {missing_keys_finding}")
    print(f"❌ Lỗi: File '{os.path.basename(finding_level_csv)}' thiếu cột khóa: {missing_keys_finding}")
    exit()
common_columns = list(set(df_breast.columns) & set(df_finding.columns))
df_breast_common = df_breast[common_columns].copy()
df_finding_common = df_finding[common_columns].copy()
logging.info(f"Thực hiện 'inner' join trên {KEY_COLUMNS}...")
merged_df = pd.merge(df_breast_common, df_finding_common, on=KEY_COLUMNS, how='inner', suffixes=('_breast', '_finding'))
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
initial_merged_rows = len(merged_df)
merged_df.drop_duplicates(subset=KEY_COLUMNS, keep='first', inplace=True)
duplicates_dropped = initial_merged_rows - len(merged_df)
if duplicates_dropped > 0:
    logging.info(f"Đã loại bỏ {duplicates_dropped} bản ghi trùng lặp.")
merged_df = merged_df[final_cols]
initial_merged_rows = len(merged_df)
merged_df.drop_duplicates(subset=KEY_COLUMNS, keep='first', inplace=True)
duplicates_dropped = initial_merged_rows - len(merged_df)
if duplicates_dropped > 0:
    logging.info(f"Đã loại bỏ {duplicates_dropped} bản ghi trùng lặp.")
logging.info(f"Số dòng cuối cùng trong DataFrame kết hợp: {len(merged_df)}")
if merged_df.empty:
    logging.error("Không có dữ liệu chung. Kết thúc.")
    print("❌ Lỗi: Không có dữ liệu chung. Kết thúc.")
    exit()
end_step_time = time.time()
logging.info(f"Kết hợp annotations hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
del df_breast, df_finding, df_breast_common, df_finding_common
gc.collect()

# =============================================================
# ===     3. TẠO ĐƯỜNG DẪN ẢNH VÀ KIỂM TRA DỮ LIỆU        ===
# =============================================================
logging.info("[Bước 2/10] Tạo đường dẫn ảnh và kiểm tra dữ liệu kết hợp...")
start_step_time = time.time()
if 'split' not in merged_df.columns:
    logging.error("Thiếu cột 'split'.")
    print(f"❌ Lỗi: Thiếu cột 'split'.")
    exit()
if 'image_id' not in merged_df.columns:
    logging.error("Thiếu cột 'image_id'.")
    print(f"❌ Lỗi: Thiếu cột 'image_id'.")
    exit()
def create_image_path(row):
    split_folder = train_dir_name if row['split'] == 'training' else test_dir_name
    image_filename = f"{row['image_id']}{IMAGE_EXTENSION}"
    path = os.path.join(split_folder, image_filename)
    return path
logging.info(f"Tạo cột '{IMAGE_PATH_COL}' với đuôi file '{IMAGE_EXTENSION}'...")
merged_df[IMAGE_PATH_COL] = merged_df.apply(create_image_path, axis=1)
if merged_df[IMAGE_PATH_COL].isnull().any():
    logging.warning("Có lỗi khi tạo đường dẫn ảnh.")
    merged_df.dropna(subset=[IMAGE_PATH_COL], inplace=True)
logging.info(f"Ví dụ đường dẫn: {merged_df[IMAGE_PATH_COL].iloc[0]}" if not merged_df.empty else "(Không có dữ liệu)")
required_cols_model = [IMAGE_PATH_COL] + METADATA_COLS_FOR_MODEL + [TARGET_COL]
missing_cols_model = [col for col in required_cols_model if col not in merged_df.columns]
if missing_cols_model:
    logging.error(f"DataFrame thiếu cột cho mô hình: {missing_cols_model}")
    print(f"❌ Lỗi: DataFrame thiếu cột cho mô hình: {missing_cols_model}")
    exit()
initial_rows = len(merged_df)
merged_df.dropna(subset=['breast_birads', 'image_id'], inplace=True)
dropped_rows = initial_rows - len(merged_df)
if dropped_rows > 0:
    logging.info(f"Đã loại bỏ {dropped_rows} dòng chứa giá trị thiếu trong các cột ['breast_birads', 'image_id'].")
logging.info(f"Số dòng hợp lệ sau khi loại bỏ NaN: {len(merged_df)}")
if merged_df.empty:
    logging.error("Không còn dữ liệu hợp lệ.")
    print("❌ Lỗi: Không còn dữ liệu hợp lệ.")
    exit()
logging.info("Mã hóa metadata và nhãn...")
le_view = LabelEncoder()
le_laterality = LabelEncoder()
le_density = LabelEncoder()
le_birads = LabelEncoder()
try:
    merged_df['view_position_encoded'] = le_view.fit_transform(merged_df['view_position'])
    merged_df['laterality_encoded'] = le_laterality.fit_transform(merged_df['laterality'])
    if not pd.api.types.is_numeric_dtype(merged_df['breast_density']):
        merged_df['breast_density_encoded'] = le_density.fit_transform(merged_df['breast_density'].astype(str))
    else:
        merged_df['breast_density_encoded'] = merged_df['breast_density']
    merged_df['birads_encoded'] = le_birads.fit_transform(merged_df[TARGET_COL])
    num_classes = len(le_birads.classes_)
    target_names = [str(cls) for cls in le_birads.classes_]
    logging.info(f"Đã mã hóa các cột. Số lớp BI-RADS: {num_classes}")
except KeyError as ke:
    logging.error(f"KeyError khi mã hóa cột: {ke}.")
    print(f"❌ Lỗi KeyError khi mã hóa cột: {ke}.")
    exit()
except Exception as e:
    logging.error(f"Lỗi không xác định khi mã hóa: {e}")
    print(f"❌ Lỗi không xác định khi mã hóa: {e}")
    exit()
end_step_time = time.time()
logging.info(f"Tiền xử lý dữ liệu kết hợp hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# Kiểm tra tệp ảnh trước khi trích xuất
logging.info("Kiểm tra tính toàn vẹn của tệp ảnh...")
missing_files = []
corrupted_files = []
for img_path in merged_df[IMAGE_PATH_COL]:
    full_path = os.path.join(image_base_dir, img_path)
    if not os.path.exists(full_path):
        missing_files.append(full_path)
    try:
        Image.open(full_path)
    except:
        corrupted_files.append(full_path)
if missing_files:
    logging.warning(f"{len(missing_files)} tệp bị thiếu: {missing_files[:5]}...")
if corrupted_files:
    logging.warning(f"{len(corrupted_files)} tệp bị hỏng: {corrupted_files[:5]}...")
merged_df = merged_df[~merged_df[IMAGE_PATH_COL].isin([os.path.relpath(f, image_base_dir) for f in missing_files + corrupted_files])]
logging.info(f"Số dòng sau khi loại bỏ tệp bị thiếu/hỏng: {len(merged_df)}")

# =============================================================
# ===          4. THIẾT LẬP TRÍCH XUẤT ĐẶC TRƯNG           ===
# =============================================================
logging.info("[Bước 3/10] Thiết lập mô hình trích xuất đặc trưng ảnh...")
start_step_time = time.time()
image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Sử dụng thiết bị: {str(device).upper()}")
logging.info(f"Tải mô hình {MODEL_NAME} pretrained...")
try:
    cnn_model = None
    if MODEL_NAME == 'resnet50':
        cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        cnn_model.fc = nn.Identity()
    elif MODEL_NAME == 'resnet101':
        cnn_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        cnn_model.fc = nn.Identity()
    else:
        logging.error(f"Model '{MODEL_NAME}' không được hỗ trợ.")
        print(f"❌ Lỗi: Model '{MODEL_NAME}' không được hỗ trợ.")
        exit()
    if cnn_model:
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
    else:
        logging.error("Không thể khởi tạo cnn_model.")
        print("❌ Lỗi: Không thể khởi tạo cnn_model.")
        exit()
    end_step_time = time.time()
    logging.info(f"Tải và cấu hình {MODEL_NAME} thành công (Thời gian: {end_step_time - start_step_time:.2f}s).")
except Exception as e:
    logging.error(f"Lỗi khi tải/cấu hình CNN: {e}")
    print(f"❌ Lỗi khi tải/cấu hình CNN: {e}")
    exit()

# =============================================================
# ===             5. TRÍCH XUẤT ĐẶC TRƯNG                   ===
# =============================================================
augmented_images_dir = "augmented_images_full"
os.makedirs(augmented_images_dir, exist_ok=True)
LABEL_COL = 'birads_encoded'
logging.info("[Bước 4/10] Bắt đầu trích xuất đặc trưng ảnh (augment nhãn 1,2,3,4 để cân bằng với nhãn 1)...")
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

# Dataset for DataLoader
class ImageDataset(Dataset):
    def __init__(self, df, transform, base_dir):
        self.df = df
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.df.iloc[idx]['image_path'])
        label = self.df.iloc[idx][LABEL_COL]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, idx

features_list = []
valid_indices_processed = []
augmentation_flags = []
skipped_files_count = 0
augment_image_count = 0
label_counts = merged_df[LABEL_COL].value_counts()
logging.info(f"Số lượng ban đầu: {dict(label_counts)}")
augment_labels = [1, 2, 3, 4]
target_per_label = 9000
max_augment_per_image = 100
current_augmented_counts = {label: label_counts.get(label, 0) for label in augment_labels}
logging.info(f"Mục tiêu số lượng cho mỗi nhãn: {target_per_label}")
batch_size = 32
dataset = ImageDataset(merged_df, image_transforms, image_base_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
for batch_imgs, batch_labels, batch_indices in tqdm(dataloader, desc="Trích xuất đặc trưng"):
    batch_imgs = batch_imgs.to(device)
    try:
        with torch.no_grad():
            features = cnn_model(batch_imgs).cpu().numpy()
        for feat, idx, label in zip(features, batch_indices, batch_labels):
            if feat.shape == (FEATURE_VECTOR_SIZE,) and not np.isnan(feat).any():
                features_list.append(feat)
                valid_indices_processed.append(idx.item())
                augmentation_flags.append(False)
                if label.item() in augment_labels:
                    augment_count = 0
                    img_path = merged_df.iloc[idx.item()]['image_path']
                    img = Image.open(os.path.join(image_base_dir, img_path)).convert('RGB')
                    while current_augmented_counts[label.item()] < target_per_label and augment_count < max_augment_per_image:
                        aug = strong_augment if label.item() == 4 else light_augment
                        aug_img = aug(img)
                        aug_feat = extract_feature_from_pil(aug_img)
                        if aug_feat is not None and aug_feat.shape == (FEATURE_VECTOR_SIZE,):
                            features_list.append(aug_feat)
                            valid_indices_processed.append(idx.item())
                            augmentation_flags.append(True)
                            augment_image_count += 1
                            current_augmented_counts[label.item()] += 1
                            augment_count += 1
                            if SAVE_AUGMENTED_IMAGES:
                                orig_filename = os.path.splitext(os.path.basename(img_path))[0]
                                aug_filename = f"{orig_filename}_aug{augment_image_count}.jpg"
                                aug_save_path = os.path.join(augmented_images_dir, aug_filename)
                                aug_img.save(aug_save_path)
                        else:
                            logging.warning(f"Bỏ qua tăng cường cho {img_path}: Đặc trưng không hợp lệ")
                            break
            else:
                logging.warning(f"Bỏ qua đặc trưng tại chỉ số {idx.item()}: Không hợp lệ")
                skipped_files_count += 1
    except Exception as e:
        logging.error(f"Lỗi trong lô xử lý: {e}")
        skipped_files_count += len(batch_imgs)
    finally:
        torch.cuda.empty_cache() if device == 'cuda' else None
    if len(features_list) % 1000 == 0 and len(features_list) > 0:
        np.save(os.path.join(BASE_OUTPUT_DIR, f"features_checkpoint_{len(features_list)}.npy"), np.stack(features_list))
        merged_df.iloc[valid_indices_processed].to_csv(os.path.join(BASE_OUTPUT_DIR, f"df_checkpoint_{len(features_list)}.csv"), index=False)
end_step_time = time.time()
logging.info(f"Trích xuất hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
logging.info(f"Số lượng ảnh gốc thành công: {len(merged_df) - skipped_files_count}")
logging.info(f"Số lượng ảnh augment tạo thêm: {augment_image_count}")
logging.info(f"Tổng số feature vector (gốc + augment): {len(features_list)}")
logging.info(f"Số lượng mới sau augment: {current_augmented_counts}")
if skipped_files_count > 0:
    logging.info(f"Ảnh bị lỗi hoặc bỏ qua: {skipped_files_count}")
if not features_list:
    logging.error("Không trích xuất được bất kỳ đặc trưng nào!")
    print("❌ Lỗi: Không trích xuất được bất kỳ đặc trưng nào!")
    exit()
df_final = merged_df.iloc[valid_indices_processed].reset_index(drop=True)
image_features = np.stack(features_list)
logging.info(f"Kích thước mảng đặc trưng ảnh cuối cùng: {image_features.shape}")
gc.collect()
torch.cuda.empty_cache() if device == 'cuda' else None

# =============================================================
# ===       6. KẾT HỢP ĐẶC TRƯNG CUỐI & CHIA DỮ LIỆU       ===
# =============================================================
logging.info("[Bước 5/10] Kết hợp đặc trưng ảnh và metadata cuối cùng...")
start_step_time = time.time()
try:
    metadata_features_final = df_final[['view_position_encoded', 'breast_density_encoded', 'laterality_encoded']].values
    scaler = StandardScaler()
    metadata_scaled_final = scaler.fit_transform(metadata_features_final)
    logging.info("Đã áp dụng StandardScaler cho metadata.")
    X = np.hstack([image_features, metadata_scaled_final])
    y = df_final['birads_encoded'].values
    logging.info(f"Kích thước mảng đặc trưng kết hợp (X): {X.shape}")
    logging.info(f"Kích thước mảng nhãn (y): {y.shape}")
except KeyError as ke:
    logging.error(f"KeyError khi lấy cột metadata: {ke}.")
    print(f"❌ Lỗi KeyError khi lấy cột metadata: {ke}.")
    exit()
except Exception as e:
    logging.error(f"Lỗi khi kết hợp đặc trưng: {e}")
    print(f"❌ Lỗi khi kết hợp đặc trưng: {e}")
    exit()
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    logging.error("Tìm thấy NaN hoặc giá trị vô cực trong X!")
    print("Lỗi: Tìm thấy NaN hoặc giá trị vô cực trong X!")
    exit()
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    logging.error("Tìm thấy NaN hoặc giá trị vô cực trong y!")
    print("Lỗi: Tìm thấy NaN hoặc giá trị vô cực trong y!")
    exit()
np.save(os.path.join(BASE_OUTPUT_DIR, "X_final.npy"), X)
np.save(os.path.join(BASE_OUTPUT_DIR, "y_final.npy"), y)
df_final.to_csv(os.path.join(BASE_OUTPUT_DIR, "df_final.csv"), index=False)
logging.info("Đã lưu X_final.npy, y_final.npy và df_final.csv")
end_step_time = time.time()
logging.info(f"Kết hợp đặc trưng hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
logging.info("[Bước 6/10] Chia dữ liệu thành tập huấn luyện và kiểm tra...")
start_step_time = time.time()
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        logging.error(f"Kích thước không khớp - X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}, X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}")
        print(f"❌ Lỗi: Kích thước không khớp - X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}, X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}")
        exit()
    logging.info(f"Tập huấn luyện: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Tập kiểm tra: X={X_test.shape}, y={y_test.shape}")
    logging.info(f"Phân phối lớp trong y_train: {np.bincount(y_train)}")
    logging.info(f"Phân phối lớp trong y_test: {np.bincount(y_test)}")
except Exception as e:
    logging.error(f"Lỗi khi chia dữ liệu: {e}")
    print(f"❌ Lỗi khi chia dữ liệu: {e}")
    exit()
end_step_time = time.time()
logging.info(f"Chia dữ liệu hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# Định nghĩa mô hình CNN
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate_model(algo_key, X_train, X_test, y_train, y_test, num_classes, target_names, device):
    algo = ALGORITHMS[algo_key]
    output_dir = algo['output_dir']
    output_model_path = algo['output_model_path']
    output_report_path = algo['output_report_path']
    output_cm_plot_path = algo['output_cm_plot_path']
    output_roc_plot_path = algo['output_roc_plot_path']
    output_pr_plot_path = algo['output_pr_plot_path']
    
    logging.info(f"[Bước 7/10] Huấn luyện mô hình {algo['name']} và Lưu mô hình...")
    start_step_time = time.time()
    
    model = None
    if algo_key == 'cnn':
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
            logging.error("Dữ liệu chứa NaN trước khi huấn luyện CNN!")
            print("❌ Lỗi: Dữ liệu chứa NaN trước khi huấn luyện CNN!")
            return None, None, None, False, -1.0, "Lỗi dự đoán/lấy nhãn", pd.DataFrame()
        input_dim = X_train.shape[1]
        model = SimpleCNN(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        batch_size = 8
        epochs = 20
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        model.train()
        for epoch in range(epochs):
            model.zero_grad()
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
    else:
        model = algo['model'](**algo['params'])
        try:
            model.fit(X_train, y_train)
            logging.info(f"Huấn luyện {algo['name']} thành công.")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        except Exception as e:
            logging.error(f"Lỗi khi huấn luyện {algo['name']}: {e}")
            print(f"❌ Lỗi khi huấn luyện {algo['name']}: {e}")
            return None, None, None, False, -1.0, "Lỗi dự đoán/lấy nhãn", pd.DataFrame()
    
    logging.info(f"Số mẫu dự đoán: {len(y_pred) if y_pred is not None else 'None'}")
    logging.info(f"Số mẫu kỳ vọng: {len(y_test)}")
    
    end_step_time = time.time()
    logging.info(f"Huấn luyện {algo['name']} hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
    
    logging.info(f"Lưu mô hình {algo['name']} vào: {output_model_path}")
    try:
        if algo_key == 'cnn':
            torch.save(model.state_dict(), output_model_path.replace('.pkl', '.pth'))
            logging.info("Lưu mô hình CNN thành công (file .pth).")
        else:
            joblib.dump(model, output_model_path)
            logging.info("Lưu mô hình thành công.")
    except Exception as e:
        logging.error(f"Lỗi khi lưu mô hình: {e}")
        print(f"⚠️ Lỗi khi lưu mô hình: {e}")
    
    try:
        joblib.dump(le_view, os.path.join(output_dir, "le_view.pkl"))
        joblib.dump(le_laterality, os.path.join(output_dir, "le_laterality.pkl"))
        joblib.dump(le_density, os.path.join(output_dir, "le_density.pkl"))
        joblib.dump(le_birads, os.path.join(output_dir, "le_birads.pkl"))
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        logging.info("Lưu các encoder và scaler thành công.")
        if os.path.exists(os.path.join(output_dir, "scaler.pkl")):
            logging.info(f"File scaler.pkl đã được lưu thành công tại: {os.path.join(output_dir, 'scaler.pkl')}")
        else:
            logging.warning("File scaler.pkl không được tạo.")
    except Exception as e:
        logging.error(f"Lỗi khi lưu encoder/scaler: {e}")
        print(f"⚠️ Lỗi khi lưu encoder/scaler: {e}")
    
    logging.info("Lưu tập đặc trưng huấn luyện cho LIME...")
    try:
        max_samples = 5000
        if X_train.shape[0] > max_samples:
            indices = np.random.choice(X_train.shape[0], max_samples, replace=False)
            X_train_subset = X_train[indices]
        else:
            X_train_subset = X_train
        np.save(os.path.join(output_dir, "X_train_features.npy"), X_train_subset)
        logging.info(f"Đã lưu X_train_features.npy với {X_train_subset.shape[0]} mẫu")
    except Exception as e:
        logging.error(f"Lỗi khi lưu X_train_features.npy: {e}")
        print(f"⚠️ Lỗi khi lưu X_train_features.npy: {e}")
    
    logging.info(f"[Bước 8/10] Đánh giá chi tiết mô hình {algo['name']}...")
    start_step_time = time.time()
    metrics_calculated = False
    if y_pred is not None and (y_pred_proba is not None or algo_key == 'cnn'):
        if y_pred_proba is not None and y_pred_proba.shape[0] == len(X_test) and y_pred_proba.shape[1] == num_classes:
            logging.info("Dự đoán thành công.")
            metrics_calculated = True
        else:
            logging.error(f"Output dự đoán xác suất có shape không mong muốn: {y_pred_proba.shape if y_pred_proba is not None else 'None'}")
            print(f"❌ Lỗi: Output dự đoán xác suất có shape không mong muốn: {y_pred_proba.shape if y_pred_proba is not None else 'None'}")
    accuracy = -1.0
    report_str = "Lỗi dự đoán/lấy nhãn"
    cm_df = pd.DataFrame()
    if metrics_calculated:
        try:
            accuracy = accuracy_score(y_test, y_pred)
            report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
            cm_df = pd.DataFrame(cm, index=le_birads.classes_, columns=le_birads.classes_)
            logging.info(f"Tổng số mẫu trong ma trận nhầm lẫn: {cm.sum()}, Kỳ vọng: {len(y_test)}")
            if cm.sum() != len(y_test):
                logging.warning("Số lượng mẫu trong ma trận nhầm lẫn không khớp với y_test!")
        except Exception as e:
            logging.error(f"Lỗi khi tính toán metrics: {e}")
            print(f"❌ Lỗi khi tính toán metrics: {e}")
            metrics_calculated = False
    
    logging.info(f"--- Kết quả đánh giá trên tập kiểm tra ({algo['name']}) ---")
    logging.info(f"Độ chính xác (Accuracy): {accuracy:.4f}" if metrics_calculated else "Độ chính xác (Accuracy): Lỗi")
    if metrics_calculated:
        logging.info("Báo cáo phân loại chi tiết (Classification Report):")
        logging.info(report_str)
        logging.info("Ma trận nhầm lẫn (Confusion Matrix):")
        logging.info(cm_df.to_string())
    else:
        logging.info("Không có báo cáo và ma trận nhầm lẫn do lỗi.")
    
    plot_evaluation_charts = metrics_calculated and (y_pred_proba is not None)
    if plot_evaluation_charts:
        logging.info(f"Vẽ và lưu heatmap ma trận nhầm lẫn: {output_cm_plot_path}")
        fig_cm = None
        try:
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 12})
            ax_cm.set_xlabel('Predicted Labels', fontsize=12)
            ax_cm.set_ylabel('True Labels', fontsize=12)
            ax_cm.set_title('Confusion Matrix Heatmap', fontsize=14)
            ax_cm.tick_params(axis='both', which='major', labelsize=10)
            plt.tight_layout()
            plt.savefig(output_cm_plot_path)
            plt.close(fig_cm)
            logging.info("Heatmap đã lưu.")
        except Exception as e:
            logging.error(f"Lỗi khi vẽ/lưu heatmap: {e}")
            print(f"⚠️ Lỗi khi vẽ/lưu heatmap: {e}")
        
        logging.info(f"Vẽ và lưu đường cong ROC (One-vs-Rest): {output_roc_plot_path}")
        fig_roc = None
        try:
            y_test_binarized = label_binarize(y_test, classes=range(num_classes))
            if y_test_binarized.shape[1] < 2:
                raise ValueError("Cần ít nhất 2 lớp để vẽ ROC.")
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax_roc.plot(fpr[i], tpr[i], lw=2, label=f'ROC class {target_names[i]} (AUC = {roc_auc[i]:.2f})')
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve - One-vs-Rest')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True)
            plt.tight_layout()
            plt.savefig(output_roc_plot_path)
            plt.close(fig_roc)
            logging.info("Biểu đồ ROC đã lưu.")
        except ValueError as ve:
            logging.error(f"ValueError khi vẽ ROC: {ve}")
            print(f"⚠️ Lỗi ValueError khi vẽ ROC: {ve}")
        except Exception as e:
            logging.error(f"Lỗi không xác định khi vẽ/lưu ROC: {e}")
            print(f"⚠️ Lỗi không xác định khi vẽ/lưu ROC: {e}")
        
        logging.info(f"Vẽ và lưu đường cong Precision-Recall (One-vs-Rest): {output_pr_plot_path}")
        fig_pr = None
        try:
            y_test_binarized = label_binarize(y_test, classes=range(num_classes))
            if y_test_binarized.shape[1] < 2:
                raise ValueError("Cần ít nhất 2 lớp để vẽ PR.")
            precision = dict()
            recall = dict()
            average_precision = dict()
            fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
            for i in range(num_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])
                ax_pr.plot(recall[i], precision[i], lw=2, label=f'PR class {target_names[i]} (AP = {average_precision[i]:.2f})')
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title('Precision-Recall Curve - One-vs-Rest')
            ax_pr.legend(loc="best")
            ax_pr.grid(True)
            plt.tight_layout()
            plt.savefig(output_pr_plot_path)
            plt.close(fig_pr)
            logging.info("Biểu đồ PR đã lưu.")
        except ValueError as ve:
            logging.error(f"ValueError khi vẽ PR: {ve}")
            print(f"⚠️ Lỗi ValueError khi vẽ PR: {ve}")
        except Exception as e:
            logging.error(f"Lỗi không xác định khi vẽ/lưu PR: {e}")
            print(f"⚠️ Lỗi không xác định khi vẽ/lưu PR: {e}")
    else:
        logging.info("Bỏ qua vẽ biểu đồ đánh giá do lỗi hoặc thiếu xác suất.")
    
    end_step_time = time.time()
    logging.info(f"Đánh giá chi tiết hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
    
    return model, y_pred, y_pred_proba, metrics_calculated, accuracy, report_str, cm_df

# Hàm giải thích LIME
def explain_with_lime(model, algo_key, X_train, X_test, y_test, num_classes, target_names):
    algo = ALGORITHMS[algo_key]
    output_dir = algo['output_dir']
    logging.info(f"[Bước 9/10] Giải thích mô hình {algo['name']} với LIME...")
    start_step_time = time.time()
    lime_plot_paths = []
    if len(X_test) < 1:
        logging.error("Tập kiểm tra rỗng, bỏ qua LIME.")
        print("❌ Lỗi: Tập kiểm tra rỗng, bỏ qua LIME.")
        return []
    try:
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality'],
            class_names=target_names,
            mode="classification"
        )
        target_num_samples = 5
        samples_per_label = target_num_samples // num_classes
        selected_indices = []
        selected_labels = []
        for label in range(num_classes):
            label_indices = np.where(y_test == label)[0]
            if len(label_indices) == 0:
                logging.warning(f"Không có mẫu nào cho nhãn {target_names[label]}.")
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
                logging.warning(f"Chỉ có {len(remaining_indices)} mẫu còn lại, không đủ để chọn {remaining_needed} mẫu.")
                selected_indices.extend(remaining_indices)
                selected_labels.extend(y_test[remaining_indices].tolist())
            else:
                extra_selected = np.random.choice(remaining_indices, size=remaining_needed, replace=False)
                selected_indices.extend(extra_selected)
                selected_labels.extend(y_test[extra_selected].tolist())
        logging.info(f"Số mẫu được chọn cho LIME: {len(selected_indices)} (mục tiêu: {target_num_samples})")
        logging.info(f"Phân bố nhãn được chọn: {dict(pd.Series(selected_labels).value_counts())}")
        for i, sample_idx in enumerate(selected_indices[:target_num_samples]):
            sample = X_test[sample_idx]
            true_label = target_names[y_test[sample_idx]]
            predict_fn = model.predict_proba if algo_key != 'cnn' else lambda x: torch.softmax(model(torch.FloatTensor(x).to(device)), dim=1).detach().cpu().numpy()
            explanation = explainer.explain_instance(
                data_row=sample,
                predict_fn=predict_fn,
                num_features=10
            )
            logging.info(f"Giải thích LIME cho mẫu {i} - Nhãn: {true_label}")
            fig = explanation.as_pyplot_figure()
            plt.title(f"Giải thích LIME - Mẫu {i} - Nhãn: {true_label}")
            plt.tight_layout()
            lime_plot_path = os.path.join(output_dir, f"lime_explanation_label{selected_labels[i]}_sample{i}.png")
            plt.savefig(lime_plot_path)
            plt.close(fig)
            lime_plot_paths.append(lime_plot_path)
            logging.info(f"Lưu biểu đồ LIME: {lime_plot_path}")
    except Exception as e:
        logging.error(f"Lỗi khi tạo giải thích LIME: {e}")
        print(f"❌ Lỗi khi tạo giải thích LIME: {e}")
    end_step_time = time.time()
    logging.info(f"Giải thích LIME hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
    return lime_plot_paths

# Hàm unforexplain_with_shap
def explain_with_shap(model, algo_key, X_train, X_test, y_test, y_pred, num_classes, target_names):
    algo = ALGORITHMS[algo_key]
    output_dir = algo['output_dir']
    logging.info(f"[Bước 10/10] Giải thích mô hình {algo['name']} với SHAP...")
    plt.switch_backend('agg')
    start_step_time = time.time()
    shap_plot_path = ""
    shap_waterfall_paths = []
    if len(X_test) < 1:
        logging.error("Tập kiểm tra rỗng!")
        print("❌ Lỗi: Tập kiểm tra rỗng!")
        return "", []
    try:
        logging.info("Khởi tạo KernelExplainer...")
        background_size = 50
        background_indices = np.random.choice(X_train.shape[0], size=background_size, replace=False)
        X_background = X_train[background_indices]
        predict_fn = model.predict_proba if algo_key != 'cnn' else lambda x: torch.softmax(model(torch.FloatTensor(x).to(device)), dim=1).detach().cpu().numpy()
        explainer = shap.KernelExplainer(predict_fn, X_background)
        max_test_samples = min(100, len(X_test))
        X_test_subset = X_test[:max_test_samples]
        y_test_subset = y_test[:max_test_samples]
        y_pred_subset = y_pred[:max_test_samples] if y_pred is not None else None
        try:
            shap_values = explainer.shap_values(X_test_subset, nsamples=50)
        except MemoryError:
            logging.error("Bộ nhớ không đủ để tính SHAP values.")
            print("❌ Lỗi: Bộ nhớ không đủ để tính SHAP values.")
            return "", []
        except Exception as e:
            logging.error(f"Lỗi khi tính SHAP values: {e}")
            print(f"❌ Lỗi khi tính SHAP values: {e}")
            return "", []
        if len(shap_values) != num_classes:
            logging.error(f"Số giá trị SHAP ({len(shap_values)}) không khớp với số lớp ({num_classes})")
            print(f"❌ Lỗi: Số giá trị SHAP ({len(shap_values)}) không khớp với số lớp ({num_classes})")
            return "", []
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test_subset,
            feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality'],
            plot_type="bar",
            class_names=target_names,
            color_bar=False,
            max_display=20,
            show=False
        )
        plt.title(f"SHAP Summary Plot for {algo['name']} Model ({len(X_test_subset)} Test Samples)", fontsize=14, pad=20)
        plt.xlabel("mean(|SHAP value|) [average impact on model output magnitude]", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        shap_plot_path = os.path.join(output_dir, f"shap_summary_model_{algo_key}.png")
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Lưu biểu đồ SHAP tóm tắt toàn bộ mô hình: {shap_plot_path}")
        target_num_samples = 5
        valid_range = np.arange(min(2050, len(X_test_subset)))
        selected_indices = np.random.choice(valid_range, size=min(target_num_samples, len(valid_range)), replace=False)
        selected_indices = [i for i in selected_indices if i < len(X_test_subset)]
        if not selected_indices:
            raise ValueError("Không có chỉ số hợp lệ nào được chọn cho SHAP waterfall plots.")
        logging.info(f"Số mẫu được chọn cho SHAP waterfall: {len(selected_indices)} (chỉ số: {selected_indices})")
        for i, sample_idx in enumerate(selected_indices):
            true_label = target_names[y_test_subset[sample_idx]]
            predicted_label = target_names[y_pred_subset[sample_idx]] if y_pred_subset is not None else "N/A"
            pred_class = y_pred_subset[sample_idx] if y_pred_subset is not None else np.argmax(predict_fn(X_test_subset[sample_idx:sample_idx+1])[0])
            shap_explanation = shap.Explanation(
                values=shap_values[pred_class][sample_idx],
                base_values=explainer.expected_value[pred_class],
                data=X_test_subset[sample_idx],
                feature_names=[f"img_f{i}" for i in range(FEATURE_VECTOR_SIZE)] + ['view_position', 'breast_density', 'laterality']
            )
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(
                shap_explanation,
                max_display=10,
                show=False
            )
            plt.title(f"SHAP Waterfall Plot - Sample {i} - True: {true_label}, Predicted: {predicted_label}", fontsize=12, pad=10)
            plt.xlabel("SHAP Value", fontsize=10)
            plt.ylabel("Features", fontsize=10)
            plt.tight_layout()
            waterfall_plot_path = os.path.join(output_dir, f"shap_waterfall_sample{i}_{algo_key}.png")
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            shap_waterfall_paths.append(waterfall_plot_path)
            logging.info(f"Lưu biểu đồ SHAP waterfall cho mẫu {i}: {waterfall_plot_path}")
    except Exception as e:
        logging.error(f"Lỗi khi tạo giải thích SHAP: {e}")
        print(f"❌ Lỗi khi tạo giải thích SHAP: {e}")
    end_step_time = time.time()
    logging.info(f"Giải thích SHAP hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")
    return shap_plot_path, shap_waterfall_paths

# Hàm lưu kết quả
def save_results(algo_key, metrics_calculated, accuracy, report_str, cm_df, lime_plot_paths, shap_plot_path, shap_waterfall_paths):
    algo = ALGORITHMS[algo_key]
    output_report_path = algo['output_report_path']
    output_cm_plot_path = algo['output_cm_plot_path']
    output_roc_plot_path = algo['output_roc_plot_path']
    output_pr_plot_path = algo['output_pr_plot_path']
    logging.info(f"[Lưu Kết Quả] Lưu kết quả đánh giá {algo['name']} vào file text...")
    start_step_time = time.time()
    try:
        script_end_time = time.time()
        total_script_duration = script_end_time - script_start_time_main
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write("="*53 + "\n")
            f.write(f"===       KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI BI-RADS ({algo['name']})       ===\n")
            f.write("="*53 + "\n\n")
            f.write(f"Thời gian chạy tổng cộng script: {total_script_duration:.2f} giây ({total_script_duration/60:.2f} phút)\n\n")
            f.write(f"Cấu hình chính:\n")
            f.write(f"  - Model CNN: {MODEL_NAME}\n")
            f.write(f"  - Kích thước ảnh: {IMAGE_SIZE}\n")
            f.write(f"  - Tham số {algo['name']}: {json.dumps(algo['params'], indent=4)}\n")
            f.write(f"  - Kích thước tập kiểm tra: {TEST_SIZE:.1%}\n")
            f.write(f"  - Số lớp BI-RADS: {num_classes}\n")
            f.write("-"*53 + "\n\n")
            if metrics_calculated:
                f.write(f"Độ chính xác (Accuracy): {accuracy:.4f}\n\n")
                f.write("Báo cáo phân loại chi tiết (Classification Report):\n")
                f.write(report_str)
                f.write("\n\nMa trận nhầm lẫn (Confusion Matrix):\n")
                f.write(cm_df.to_string())
                f.write("\n\n")
                f.write("Các biểu đồ đánh giá:\n")
                f.write(f"  - Ma trận nhầm lẫn: {output_cm_plot_path}\n")
                f.write(f"  - Đường cong ROC: {output_roc_plot_path}\n")
                f.write(f"  - Đường cong Precision-Recall: {output_pr_plot_path}\n")
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
                f.write("  Vui lòng kiểm tra lại log lỗi trên màn hình.\n")
            f.write("="*53 + "\n")
        logging.info(f"Lưu báo cáo thành công vào: {output_report_path}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file báo cáo: {e}")
        print(f"⚠️ Lỗi khi lưu file báo cáo: {e}")
    end_step_time = time.time()
    logging.info(f"Lưu kết quả hoàn tất (Thời gian: {end_step_time - start_step_time:.2f}s).")

# Chạy lần lượt từng thuật toán
for algo_key in ALGORITHMS.keys():
    logging.info(f"=== BẮT ĐẦU XỬ LÝ THUẬT TOÁN: {ALGORITHMS[algo_key]['name']} ===")
    try:
        df_final.to_csv(ALGORITHMS[algo_key]['output_processed_csv_path'], index=False)
        logging.info(f"Đã lưu DataFrame đã xử lý vào: {ALGORITHMS[algo_key]['output_processed_csv_path']}")
        result = train_and_evaluate_model(algo_key, X_train, X_test, y_train, y_test, num_classes, target_names, device)
        if result:
            model, y_pred, y_pred_proba, metrics_calculated, accuracy, report_str, cm_df = result
            lime_plot_paths = explain_with_lime(model, algo_key, X_train, X_test, y_test, num_classes, target_names)
            shap_plot_path, shap_waterfall_paths = explain_with_shap(model, algo_key, X_train, X_test, y_test, y_pred, num_classes, target_names)
            save_results(algo_key, metrics_calculated, accuracy, report_str, cm_df, lime_plot_paths, shap_plot_path, shap_waterfall_paths)
            np.save(os.path.join(ALGORITHMS[algo_key]['output_dir'], f"y_pred_{algo_key}.npy"), y_pred)
            np.save(os.path.join(ALGORITHMS[algo_key]['output_dir'], f"y_pred_proba_{algo_key}.npy"), y_pred_proba)
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {ALGORITHMS[algo_key]['name']}: {e}")
        print(f"❌ Lỗi khi xử lý {ALGORITHMS[algo_key]['name']}: {e}")
    finally:
        del model, y_pred, y_pred_proba
        gc.collect()
        torch.cuda.empty_cache() if device == 'cuda' else None
        logging.info(f"=== KẾT THÚC XỬ LÝ THUẬT TOÁN: {ALGORITHMS[algo_key]['name']} ===")

logging.info("-------------------------------------------------------------")
logging.info(f"Quy trình Kết hợp và Phân loại BI-RADS Hoàn tất ({time.time() - script_start_time_main:.2f}s)")
logging.info("-------------------------------------------------------------")
print(f"\n-------------------------------------------------------------")
print(f"--- Quy trình Kết hợp và Phân loại BI-RADS Hoàn tất ({time.time() - script_start_time_main:.2f}s) ---")
print("-------------------------------------------------------------")