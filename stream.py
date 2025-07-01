import os
import numpy as np
import joblib
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap
import gc
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa hằng số
FEATURE_VECTOR_SIZE = 2048
METADATA_COLS = ["view_position", "breast_density", "laterality"]
EXPECTED_FEATURE_SIZE = FEATURE_VECTOR_SIZE + len(METADATA_COLS)  # 2051
MAX_SAMPLES = 100
MAX_IMAGES = 5  # Giới hạn tối đa 5 ảnh

# Định nghĩa đường dẫn tệp mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_files = {
    "Decision Tree": {
        "model": os.path.join(BASE_DIR, "kết quả/dt_lime_shap/dt_birads_model.pkl"),
        "dir": os.path.join(BASE_DIR, "kết quả/dt_lime_shap")
    },
    "Random Forest": {
        "model": os.path.join(BASE_DIR, "kết quả/rf_lime_shap/rf_birads_model.pkl"),
        "dir": os.path.join(BASE_DIR, "kết quả/rf_lime_shap")
    },
    "XGBoost": {
        "model": os.path.join(BASE_DIR, "kết quả/xgb_lime_shap/xgb_birads_model.pkl"),
        "dir": os.path.join(BASE_DIR, "kết quả/xgb_lime_shap")
    },
    "CNN": {
        "model": os.path.join(BASE_DIR, "kết quả/cnn_lime_shap/cnn_birads_model.pth"),  # Đường dẫn cho mô hình CNN đã huấn luyện
        "dir": os.path.join(BASE_DIR, "kết quả/cnn_lime_shap")
    },
    "SVM": {
        "model": os.path.join(BASE_DIR, "kết quả/svm_lime_shap/svm_birads_model.pkl"),
        "dir": os.path.join(BASE_DIR, "kết quả/svm_lime_shap")
    },
    "Naive Bayes": {
        "model": os.path.join(BASE_DIR, "kết quả/nb_lime_shap/nb_birads_model.pkl"),
        "dir": os.path.join(BASE_DIR, "kết quả/nb_lime_shap")
    }
}

# Định nghĩa mô hình CNN (nếu chưa có mô hình đã huấn luyện)
class FeatureCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeatureCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def load_cnn_model(model_path):
    model = FeatureCNN(input_dim=EXPECTED_FEATURE_SIZE, num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Hàm trích xuất đặc trưng từ ảnh
def extract_features(image):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image).cpu().numpy().flatten().astype(np.float32)
    if features.shape[0] != FEATURE_VECTOR_SIZE:
        raise ValueError(f"Kích thước đặc trưng ảnh ({features.shape[0]}) không khớp với mong đợi ({FEATURE_VECTOR_SIZE})")
    return features

# Hàm dự đoán mức độ BI-RADS
def predict(model, X, le_birads, is_cnn=False):
    if is_cnn:
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X_tensor), dim=1).cpu().numpy()[0]
    else:
        probs = model.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred_label = le_birads.inverse_transform([pred_idx])[0]
    return pred_label, probs

# Hàm giải thích LIME
def explain_lime(model, X, X_train_features, feature_names, class_names, is_cnn=False):
    try:
        explainer = LimeTabularExplainer(
            X_train_features,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification"
        )
        predict_fn = model.predict_proba if not is_cnn else (lambda x: torch.softmax(model(torch.FloatTensor(x).to(device)), dim=1).cpu().numpy())
        exp = explainer.explain_instance(X[0], predict_fn, num_features=6)
        fig = exp.as_pyplot_figure()
        return fig
    except Exception as e:
        st.error(f"Lỗi khi tạo LIME: {e}")
        return None

# Hàm giải thích SHAP
def explain_shap(model, algo_key, X, X_train_features, feature_names, is_cnn=False):
    try:
        if algo_key in ["Decision Tree", "Random Forest", "XGBoost", "SVM", "Naive Bayes"]:
            explainer = shap.TreeExplainer(model) if algo_key in ["Decision Tree", "Random Forest", "XGBoost"] else shap.KernelExplainer(model.predict_proba, shap.sample(X_train_features, 50))
        else:  # CNN
            explainer = shap.KernelExplainer(lambda x: torch.softmax(model(torch.FloatTensor(x).to(device)), dim=1).cpu().numpy(), shap.sample(X_train_features, 50))
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        return fig
    except Exception as e:
        st.error(f"Lỗi khi tạo SHAP: {e}")
        return None

# Streamlit giao diện
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>DỰ ĐOÁN MỨC ĐỘ BI-RADS (1-5) TỪ ẢNH SIÊU ÂM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Tải lên tối đa 5 ảnh siêu âm và nhập thông tin cho từng ảnh.</p>", unsafe_allow_html=True)

# Tải bộ mã hóa, scaler và X_train_features
output_dir = os.path.join(BASE_DIR, "kết quả/dt_lime_shap")
try:
    le_view = joblib.load(os.path.join(output_dir, "le_view.pkl"))
    le_density = joblib.load(os.path.join(output_dir, "le_density.pkl"))
    le_laterality = joblib.load(os.path.join(output_dir, "le_laterality.pkl"))
    le_birads = joblib.load(os.path.join(output_dir, "le_birads.pkl"))
    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    X_train_features = np.load(os.path.join(output_dir, "X_train_features.npy"), mmap_mode='r').astype(np.float32)
    view_options = list(le_view.classes_)
    density_options = list(le_density.classes_)
    laterality_options = list(le_laterality.classes_)
except FileNotFoundError:
    st.error("Không tìm thấy tệp mã hóa, scaler hoặc X_train_features.")
    view_options = ["CC", "MLO"]
    density_options = ["A", "B", "C", "D"]
    laterality_options = ["L", "R"]
    le_birads = None
    scaler = None
    X_train_features = None

# Tải lên nhiều ảnh
uploaded_files = st.file_uploader("Tải lên ảnh siêu âm", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.error(f"Vui lòng chỉ tải lên tối đa {MAX_IMAGES} ảnh.")
        uploaded_files = uploaded_files[:MAX_IMAGES]

    # Hiển thị ảnh và nhập metadata cho từng ảnh
    images_display = []
    metadata_inputs = {}
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Ảnh {i+1}")
        image = Image.open(uploaded_file).convert("RGB")
        display_size = (300, 300)
        image_display = image.copy()
        image_display.thumbnail(display_size, Image.LANCZOS)
        images_display.append(image_display)
        st.image(image_display, caption=f"Ảnh {i+1}", use_container_width=False)

        with st.expander(f"Nhập thông tin cho Ảnh {i+1}"):
            view_pos = st.selectbox(f"Vị Trí Chụp (View Position) - Ảnh {i+1}", view_options, key=f"view_{i}")
            density = st.selectbox(f"Mật Độ Vú (Breast Density) - Ảnh {i+1}", density_options, key=f"density_{i}")
            laterality = st.selectbox(f"Bên Vú (Laterality) - Ảnh {i+1}", laterality_options, key=f"laterality_{i}")
            metadata_inputs[i] = {"view_position": view_pos, "breast_density": density, "laterality": laterality}
            explain = st.checkbox("Hiển thị giải thích LIME và SHAP", key=f"explain_{i}")

    st.markdown("---")

    # Xử lý dự đoán
    with st.spinner("Đang xử lý ảnh và dự đoán..."):
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            features = extract_features(image)

            if scaler is None or le_birads is None or X_train_features is None:
                st.error(f"Không thể xử lý ảnh {i+1} do thiếu le_birads.pkl, scaler.pkl hoặc X_train_features.npy")
            else:
                metadata = np.array([
                    le_view.transform([metadata_inputs[i]["view_position"]])[0],
                    le_density.transform([metadata_inputs[i]["breast_density"]])[0],
                    le_laterality.transform([metadata_inputs[i]["laterality"]])[0]
                ]).reshape(1, -1)
                metadata_scaled = scaler.transform(metadata).astype(np.float32)

                X = np.hstack([features.reshape(1, -1), metadata_scaled])

                if X_train_features.shape[1] != EXPECTED_FEATURE_SIZE:
                    st.error(f"Kích thước đặc trưng X_train_features ({X_train_features.shape[1]}) không khớp với mong đợi ({EXPECTED_FEATURE_SIZE}) cho ảnh {i+1}")
                else:
                    if X_train_features.shape[0] > MAX_SAMPLES:
                        indices = np.random.choice(X_train_features.shape[0], MAX_SAMPLES, replace=False)
                        X_train_features_subset = X_train_features[indices]
                    else:
                        X_train_features_subset = X_train_features

                    st.markdown(f"<h2 style='color: #2E86C1;'>Kết Quả Dự Đoán - Ảnh {i+1}</h2>", unsafe_allow_html=True)
                    feature_names = [f"feature_{i}" for i in range(FEATURE_VECTOR_SIZE)] + METADATA_COLS
                    
                    for algo_key, files in model_files.items():
                        st.markdown(f"#### {algo_key}")
                        try:
                            if algo_key == "CNN":
                                model = load_cnn_model(files["model"])
                                pred_label, probs = predict(model, X, le_birads, is_cnn=True)
                            else:
                                model = joblib.load(files["model"])
                                pred_label, probs = predict(model, X, le_birads)
                            st.markdown(f"**Mức Độ BI-RADS**: <span style='color: #E74C3C;'>{pred_label}</span>", unsafe_allow_html=True)

                            if metadata_inputs[i].get("explain", False):
                                st.markdown("##### Giải Thích LIME")
                                lime_fig = explain_lime(model, X, X_train_features_subset, feature_names, le_birads.classes_, is_cnn=(algo_key == "CNN"))
                                if lime_fig:
                                    st.pyplot(lime_fig)
                                    plt.close(lime_fig)

                                st.markdown("##### Giải Thích SHAP")
                                shap_fig = explain_shap(model, algo_key, X, X_train_features_subset, feature_names, is_cnn=(algo_key == "CNN"))
                                if shap_fig:
                                    st.pyplot(shap_fig)
                                    plt.close(shap_fig)

                            gc.collect()
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý {algo_key} cho ảnh {i+1}: {e}")

else:
    st.info("Vui lòng tải lên ít nhất một ảnh để dự đoán.")

# Thêm footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Ứng dụng được phát triển để hỗ trợ dự đoán BI-RADS từ ảnh siêu âm. Kết quả chỉ mang tính tham khảo.</p>", unsafe_allow_html=True)