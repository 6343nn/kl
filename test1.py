import os
import time
import shutil # Để di chuyển file
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from skimage.metrics import structural_similarity as ssim
from PIL import UnidentifiedImageError

#import brisque # Bỏ comment nếu đã cài đặt và muốn sử dụng. Cần cài: pip install pybrisque

print("Đã import các thư viện cần thiết.")

# =============================================================
# ===                THIẾT LẬP CẤU HÌNH                    ===
# =============================================================

# --- Đường dẫn ---
input_dir = "ảnh_đã_tách_3"  # <<<--- !!! CẬP NHẬT ĐƯỜNG DẪN NÀY !!!
output_dir_base = "ket_qua_mammogram_final3"        # Thư mục gốc cho output
output_accepted_dir = os.path.join(output_dir_base, "accepted") # Thư mục chứa ảnh chấp nhận
output_rejected_dir = os.path.join(output_dir_base, "rejected") # Thư mục chứa ảnh bị loại

# --- Prompt và Negative Prompt ---
prompt = (
    "high-resolution mammogram, MLO view (mediolateral oblique), detailed anatomically accurate breast tissue structure, "
    "realistic grayscale contrast, high dynamic range, sharp focus, minimal artifacts, clean background, "
    "consistent BIRADS features, medically plausible appearance, photorealistic"
)
negative_prompt = (
    "low quality, worst quality, bad quality, blurry, blurred, noise, noisy, grain, grainy, text, signature, watermark, label, drawing, sketch, cartoon, illustration, painting, art, multiple views, multiple breasts, "
    "distorted anatomy, deformed, disfigured, unrealistic, unnatural, artifact, jpeg artifacts, compression artifacts, overexposed, underexposed, low contrast, bad contrast, "
    "implants, surgical clips, foreign object, metal, grid lines, markers, annotations, overlays, borders, frame, "
    "duplicate structures, fused anatomy, incorrect view (e.g., CC view instead of MLO), motion blur, patient movement, low resolution, pixelated"
)

# --- Tham số Sinh Ảnh ---
strength_values = [0.15, 0.25, 0.35] # Mức độ thay đổi so với ảnh gốc/canny (thấp -> giữ cấu trúc)
guidance_values = [3.0, 4.5, 6.0]    # Mức độ bám sát prompt (thấp-trung bình -> tự nhiên hơn)
num_inference_steps = 30             # Số bước khử nhiễu (tăng -> chi tiết hơn, chậm hơn)
# controlnet_scales = [0.9, 1.0, 1.1]  # Tùy chọn: Trọng số ảnh hưởng của ControlNet

# --- Tham số khác ---
pixel_to_cm = 0.05                   # Tỷ lệ chuyển đổi pixel sang cm (cho đánh giá contour)
pipeline_size = (512, 512)           # Kích thước ảnh input cho pipeline (thường là 512 hoặc 768)
use_adaptive_canny = True            # True: Dùng ngưỡng canny tự động; False: Dùng ngưỡng cố định bên dưới
canny_lower_threshold = 100          # Ngưỡng dưới cố định (chỉ dùng nếu use_adaptive_canny=False)
canny_upper_threshold = 200          # Ngưỡng trên cố định (chỉ dùng nếu use_adaptive_canny=False)
reproducible_seed = False            # True: Dùng seed cố định (tái lập); False: Dùng seed ngẫu nhiên (đa dạng)
base_seed = 42                       # Seed gốc (chỉ dùng nếu reproducible_seed=True)

# --- Ngưỡng Lọc Tự động ---
# <<<--- !!! QUAN TRỌNG: CẦN ĐIỀU CHỈNH CÁC NGƯỠNG NÀY SAU KHI THỬ NGHIỆM !!! ---<<<
ENABLE_CONTOUR_FILTER = True         # True để bật lọc theo contour
MAX_ACCEPTABLE_AVG_DIFF_CM = 1.0     # Ngưỡng max cho độ lệch contour TRUNG BÌNH (cm)
MAX_ACCEPTABLE_MAX_DIFF_CM = 2.5     # Ngưỡng max cho độ lệch contour LỚN NHẤT (cm)

ENABLE_SSIM_FILTER = True            # True để bật lọc theo SSIM
MIN_ACCEPTABLE_SSIM = 0.65           # Ngưỡng SSIM TỐI THIỂU so với ảnh gốc (0-1, càng cao càng giống)

ENABLE_BRISQUE_FILTER = False        # True để bật lọc theo BRISQUE (cần cài pybrisque)
# MAX_ACCEPTABLE_BRISQUE = 40        # Ngưỡng BRISQUE TỐI ĐA (điểm càng thấp, chất lượng càng tốt)
# <<<--------------------------------------------------------------------------->>>

# --- Thiết lập Device và Độ chính xác ---
device = "cuda" if torch.cuda.is_available() else "cpu"
use_float16 = device == "cuda" # Sử dụng float16 trên GPU cho hiệu năng tốt hơn

print(f"\n🚀 Sử dụng thiết bị: {device.upper()}")
if use_float16: print("⚡ Sử dụng độ chính xác Float16 trên GPU.")
else: print("🧊 Sử dụng độ chính xác Float32 trên CPU (sẽ chậm hơn).")

# Tạo các thư mục output nếu chưa có
os.makedirs(output_accepted_dir, exist_ok=True)
os.makedirs(output_rejected_dir, exist_ok=True)
print(f"📂 Thư mục ảnh chấp nhận: {os.path.abspath(output_accepted_dir)}")
print(f"🗑️ Thư mục ảnh bị loại: {os.path.abspath(output_rejected_dir)}")

# =============================================================
# ===                   HÀM PHỤ TRỢ                       ===
# =============================================================

def get_lower_contour(img_array_gray, crop_ratio=0.3):
    """Tìm contour dưới cùng của ảnh từ mảng NumPy thang độ xám."""
    if img_array_gray is None or img_array_gray.ndim != 2:
        print("    ⚠️ [Contour] Đầu vào không hợp lệ.")
        return None, (0, 0)
    h, w = img_array_gray.shape
    if h == 0 or w == 0: return None, (0, 0)
    start_row = int(h * (1 - crop_ratio))
    lower_crop = img_array_gray[start_row:, ]
    if lower_crop.size == 0: return None, (h, w) # Crop không thành công

    ksize = (5, 5)
    if lower_crop.shape[0] < ksize[0] or lower_crop.shape[1] < ksize[1]:
        ksize = (min(3, max(1, lower_crop.shape[0] // 2 * 2 + 1)), min(3, max(1, lower_crop.shape[1] // 2 * 2 + 1)))

    try:
        # Áp dụng Gaussian Blur và Threshold
        blurred = cv2.GaussianBlur(lower_crop, ksize, 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        # Tìm contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Tìm contour lớn nhất theo diện tích
            max_contour = max(contours, key=cv2.contourArea)
            # Điều chỉnh tọa độ y của contour về hệ tọa độ ảnh gốc
            max_contour[:, :, 1] += start_row
            return max_contour, img_array_gray.shape
        else:
            # print("    ⚠️ [Contour] Không tìm thấy contour nào.")
            return None, img_array_gray.shape
    except cv2.error as e:
         print(f"    ⚠️ [Contour] Lỗi OpenCV: {e}")
         return None, img_array_gray.shape

def compare_contours(contour1, contour2, shape):
    """So sánh hai contours dựa trên sự khác biệt trung bình và lớn nhất theo trục y."""
    h, w = shape
    if contour1 is None or contour2 is None or not contour1.size or not contour2.size: return None, None
    diffs = []
    y_coords1 = {}
    y_coords2 = {}
    # Tạo cấu trúc dữ liệu để truy cập nhanh các điểm y theo x
    for pt in contour1: x, y = pt[0]; y_coords1.setdefault(x, []).append(y)
    for pt in contour2: x, y = pt[0]; y_coords2.setdefault(x, []).append(y)
    # Duyệt qua các cột x có điểm ở cả hai contour
    common_x = set(y_coords1.keys()) & set(y_coords2.keys())
    if not common_x:
        # print("    ⚠️ [Contour Compare] Không có điểm x chung.")
        return None, None
    for x in common_x:
        mean_y1 = np.mean(y_coords1[x])
        mean_y2 = np.mean(y_coords2[x])
        diffs.append(abs(mean_y1 - mean_y2))
    if diffs: return np.mean(diffs), np.max(diffs)
    else: return None, None # Không nên xảy ra nếu common_x không rỗng

def get_adaptive_canny(image_np_gray, sigma=0.33):
    """Tạo ảnh Canny với ngưỡng tự động dựa trên median."""
    # Làm mờ nhẹ để giảm nhiễu trước khi tính median
    blurred = cv2.GaussianBlur(image_np_gray, (3, 3), 0)
    v = np.median(blurred)
    # Áp dụng công thức Canny tự động
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # print(f"    - Ngưỡng Canny tự động: ({lower}, {upper})") # Bỏ comment nếu muốn xem ngưỡng
    edged = cv2.Canny(image_np_gray, lower, upper)
    return edged

# =============================================================
# ===                    TẢI MODEL AI                      ===
# =============================================================
controlnet = None
pipe = None
model_load_successful = False
try:
    start_load_time = time.time()
    print("\n⏳ Đang tải ControlNet model (lllyasviel/sd-controlnet-canny)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if use_float16 else torch.float32
    )
    print("✅ ControlNet model đã tải xong.")

    print("⏳ Đang tải Stable Diffusion pipeline (runwayml/stable-diffusion-v1-5)...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if use_float16 else torch.float32,
        safety_checker=None # Tắt safety checker (phổ biến khi dùng ControlNet tùy chỉnh)
    ).to(device)
    print(f"✅ Stable Diffusion pipeline đã tải xong và chuyển sang device '{device}'.")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print("✅ Đã cấu hình Scheduler (UniPCMultistepScheduler).")

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✅ Đã bật Xformers memory efficient attention (tăng tốc/giảm bộ nhớ).")
        except Exception as e:
            print(f"⚠️ Không thể bật Xformers: {e}. Có thể ảnh hưởng hiệu năng/bộ nhớ.")
        # Cân nhắc bật nếu VRAM cực kỳ hạn chế (< 6GB), nhưng sẽ chậm hơn đáng kể:
        # try:
        #      pipe.enable_model_cpu_offload()
        #      print("✅ Đã bật CPU offloading (cho VRAM thấp).")
        # except Exception as e:
        #      print(f"⚠️ Không thể bật CPU offloading: {e}.")

    end_load_time = time.time()
    print(f"⏱️ Tổng thời gian tải model: {end_load_time - start_load_time:.2f} giây")
    model_load_successful = True

except Exception as e:
    print(f"\n❌ Lỗi nghiêm trọng khi tải hoặc cấu hình model AI: {e}")
    print("   Vui lòng kiểm tra kết nối mạng, dung lượng đĩa và cài đặt thư viện (đặc biệt là PyTorch với CUDA nếu dùng GPU).")
    print("   Thoát chương trình.")
    exit()

if not model_load_successful:
     print("\n❌ Không thể tải model AI. Thoát chương trình.")
     exit()

# =============================================================
# ===                 CHỌN ẢNH ĐẦU VÀO                    ===
# =============================================================
if not os.path.isdir(input_dir):
    print(f"❌ Thư mục đầu vào '{input_dir}' không tồn tại hoặc không phải là thư mục.")
    exit()

try:
    all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))])
    num_available_images = len(all_images)
except Exception as e:
    print(f"❌ Lỗi khi đọc thư mục đầu vào '{input_dir}': {e}")
    exit()


if not all_images:
    print(f"❌ Không tìm thấy file ảnh nào được hỗ trợ trong thư mục '{input_dir}'.")
    exit()

print(f"\n🖼️ Tìm thấy tổng cộng {num_available_images} ảnh trong thư mục đầu vào.")
# Chỉ hiển thị một phần danh sách nếu quá dài
max_display = 30
if num_available_images > 0:
    print("   Danh sách (một phần):")
    for idx, img_name in enumerate(all_images):
        if idx < max_display:
            print(f"     {idx+1}. {img_name}")
        elif idx == max_display:
            print(f"     ... và {num_available_images - max_display} ảnh khác.")
            break
else:
     print("   Thư mục không chứa ảnh nào.")
     exit()


selected_images_paths = []
while not selected_images_paths:
    print("\n" + "-"*30)
    print("Lựa chọn ảnh để xử lý:")
    print("  - Nhập một số N: Chọn các ảnh từ 1 đến N.")
    print("  - Nhập danh sách số cách nhau bởi dấu phẩy (VD: 1,5,10): Chọn các ảnh cụ thể.")
    selected_input = input(f"👉 Nhập lựa chọn của bạn: ")
    selected_input = selected_input.strip()

    valid_indices = [] # Lưu index 0-based
    invalid_inputs = []

    if not selected_input:
        print("   ❌ Vui lòng nhập lựa chọn.")
        continue

    if ',' in selected_input:
        # --- Xử lý danh sách số ---
        parts = selected_input.split(',')
        potential_indices = set()
        for part in parts:
            part = part.strip()
            if not part: continue
            try:
                index_num = int(part) # Số người dùng nhập (1-based)
                if 1 <= index_num <= num_available_images:
                    potential_indices.add(index_num - 1) # Chuyển sang 0-based
                else:
                    invalid_inputs.append(part)
            except ValueError:
                invalid_inputs.append(part)
        valid_indices = sorted(list(potential_indices))
    else:
        # --- Xử lý một số N ---
        try:
            num_to_select = int(selected_input)
            if 1 <= num_to_select <= num_available_images:
                valid_indices = list(range(num_to_select)) # Index từ 0 đến N-1
            elif num_to_select <= 0:
                 print("   ❌ Số lượng ảnh chọn phải là số dương.")
                 invalid_inputs.append(selected_input)
            else: # num_to_select > num_available_images
                print(f"   ❌ Chỉ có {num_available_images} ảnh, không thể chọn từ 1 đến {num_to_select}.")
                invalid_inputs.append(selected_input)
        except ValueError:
            print("   ❌ Nhập liệu không hợp lệ. Chỉ nhập số hoặc danh sách số.")
            invalid_inputs.append(selected_input)

    if invalid_inputs:
        print(f"   ⚠️ Các mục nhập sau không hợp lệ hoặc ngoài phạm vi [1-{num_available_images}] đã bị bỏ qua: {invalid_inputs}")

    if not valid_indices:
        print("   ❌ Không có ảnh hợp lệ nào được chọn. Vui lòng thử lại.")
    else:
        selected_images_paths = [os.path.join(input_dir, all_images[i]) for i in valid_indices]
        print(f"✅ Đã chọn {len(selected_images_paths)} ảnh để xử lý.")
        if len(valid_indices) > 1:
             print(f"   (Số thứ tự từ {valid_indices[0]+1} đến {valid_indices[-1]+1} nếu chọn theo khoảng, hoặc các số cụ thể)")
        elif len(valid_indices) == 1:
             print(f"   (Số thứ tự: {valid_indices[0]+1})")

# =============================================================
# ===               XỬ LÝ CHÍNH TỪNG ẢNH                  ===
# =============================================================
total_start_time = time.time()
total_generated_count = 0
total_accepted_count = 0
total_rejected_count = 0
total_error_count = 0
total_contour_fail_count = 0

print(f"\n{'='*20} BẮT ĐẦU XỬ LÝ {len(selected_images_paths)} ẢNH {'='*20}")

# --- Tính toán tổng số lần tạo ảnh dự kiến ---
num_combinations = len(strength_values) * len(guidance_values)
# if controlnet_scales: num_combinations *= len(controlnet_scales) # Bỏ comment nếu dùng controlnet_scales
estimated_total_generations = len(selected_images_paths) * num_combinations
print(f"⚙️ Cấu hình: {num_combinations} phiên bản cho mỗi ảnh gốc.")
print(f"⏳ Ước tính tổng số lần tạo ảnh: {estimated_total_generations}")
print("-" * 60)

# --- Bắt đầu vòng lặp xử lý ---
for img_idx, img_path in enumerate(selected_images_paths):
    img_process_start_time = time.time()
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Tạo thư mục con cho từng ảnh gốc trong accepted và rejected
    sub_output_accepted_dir = os.path.join(output_accepted_dir, base_name)
    sub_output_rejected_dir = os.path.join(output_rejected_dir, base_name)
    os.makedirs(sub_output_accepted_dir, exist_ok=True)
    os.makedirs(sub_output_rejected_dir, exist_ok=True)

    print(f"\n--- [{img_idx+1}/{len(selected_images_paths)}] 🔍 Bắt đầu xử lý ảnh: {base_name} ---")
    print(f"    Đường dẫn gốc: {img_path}")

    accepted_count_per_image = 0
    generation_count_per_image = 0
    error_count_per_image = 0
    contour_orig = None
    original_image_pil = None
    original_image_np_gray = None

    try:
        # --- 1. Đọc và Chuẩn bị Ảnh Gốc ---
        print("    1. Đọc và chuẩn bị ảnh gốc...")
        original_image_pil = Image.open(img_path).convert("RGB")
        original_size_actual = original_image_pil.size # Lấy kích thước thực tế (width, height)

        # Lưu ảnh gốc vào thư mục accepted để tham chiếu
        original_save_path = os.path.join(sub_output_accepted_dir, f"{base_name}_original.png")
        original_image_pil.save(original_save_path)
        print(f"       💾 Đã lưu ảnh gốc vào: {os.path.relpath(original_save_path)}")

        # Chuyển sang NumPy (thang độ xám)
        original_image_np_gray = np.array(original_image_pil.convert("L"))

        # --- 2. Lấy Contour Gốc ---
        print("    2. Tìm contour ảnh gốc...")
        contour_orig, shape_orig = get_lower_contour(original_image_np_gray)
        if contour_orig is None:
            print("       ⚠️ Không phát hiện được contour gốc hợp lệ. Bỏ qua ảnh này.")
            total_contour_fail_count += 1
            # Di chuyển ảnh gốc vào thư mục rejected để dễ kiểm tra
            try:
                 shutil.move(original_save_path, os.path.join(sub_output_rejected_dir, os.path.basename(original_save_path)))
            except Exception as move_err:
                 print(f"       Lỗi khi di chuyển file gốc bị lỗi contour: {move_err}")
            continue # Chuyển sang ảnh tiếp theo trong vòng lặp for img_path

        print("       ✅ Tìm thấy contour gốc.")

        # --- 3. Chuẩn bị Ảnh cho Pipeline AI ---
        print("    3. Chuẩn bị ảnh cho pipeline...")
        pipeline_input_image_pil = original_image_pil.resize(pipeline_size, Image.LANCZOS)
        pipeline_input_image_np = np.array(pipeline_input_image_pil) # Ảnh RGB cho Canny
        pipeline_input_gray_np = cv2.cvtColor(pipeline_input_image_np, cv2.COLOR_RGB2GRAY) # Ảnh Gray cho Canny

        # Tạo ảnh Canny (Thích ứng hoặc Cố định)
        if use_adaptive_canny:
            canny_np = get_adaptive_canny(pipeline_input_gray_np)
            # print("       - Đã tạo Canny map thích ứng.")
        else:
            canny_np = cv2.Canny(pipeline_input_gray_np, canny_lower_threshold, canny_upper_threshold)
            # print(f"       - Đã tạo Canny map cố định ({canny_lower_threshold}, {canny_upper_threshold}).")

        canny_image_pil = Image.fromarray(cv2.cvtColor(canny_np, cv2.COLOR_GRAY2RGB))

        # Lưu ảnh Canny vào thư mục accepted để tham chiếu
        canny_save_path = os.path.join(sub_output_accepted_dir, f"{base_name}_canny_control.png")
        canny_image_pil.save(canny_save_path)
        # print(f"       💾 Đã lưu ảnh Canny control vào: {os.path.relpath(canny_save_path)}")
        print("       ✅ Ảnh đầu vào và Canny map đã sẵn sàng.")


        # --- 4. Vòng lặp Tạo Ảnh và Lọc ---
        print(f"    4. Bắt đầu tạo {num_combinations} phiên bản ảnh...")

        for strength in strength_values:
            for guidance in guidance_values:
                # for control_scale in controlnet_scales: # Bỏ comment nếu dùng controlnet_scale
                    gen_start_time = time.time()
                    generation_count_per_image += 1
                    total_generated_count += 1

                    # --- Tạo tên file output ---
                    # Cập nhật tên file nếu dùng controlnet_scale:
                    # out_name = f"{base_name}_s{strength}_g{guidance}_cs{control_scale}.png"
                    out_name = f"{base_name}_s{strength}_g{guidance}.png"
                    progress_percent = (total_generated_count / estimated_total_generations) * 100 if estimated_total_generations > 0 else 0

                    print(f"       ⏳ [{generation_count_per_image}/{num_combinations} | Tổng: {total_generated_count}/{estimated_total_generations} ({progress_percent:.1f}%)] Tạo: {out_name} ...")

                    # --- Thực hiện Inference ---
                    try:
                        # Tạo seed (Tái lập hoặc Ngẫu nhiên)
                        if reproducible_seed:
                             current_seed = base_seed + img_idx * num_combinations + generation_count_per_image
                        else:
                             current_seed = int(time.time() * 1000) + generation_count_per_image # Seed ngẫu nhiên thay đổi
                        generator = torch.Generator(device=device).manual_seed(current_seed)

                        # Gọi pipeline AI
                        result_pil = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=pipeline_input_image_pil,      # Ảnh màu đã resize
                            control_image=canny_image_pil,   # Ảnh Canny 3 kênh
                            strength=strength,
                            guidance_scale=guidance,
                            num_inference_steps=num_inference_steps,
                            # controlnet_conditioning_scale=control_scale, # Bỏ comment nếu dùng
                            generator=generator
                        ).images[0]

                        # --- Hậu xử lý kết quả ---
                        result_resized_pil = result_pil.resize(original_size_actual, Image.LANCZOS)
                        result_resized_np_gray = np.array(result_resized_pil.convert("L")) # Chuyển sang Gray để đánh giá

                        # --- Đánh giá và Lọc ---
                        contour_gen, _ = get_lower_contour(result_resized_np_gray)

                        contour_ok = not ENABLE_CONTOUR_FILTER # Mặc định OK nếu không bật filter
                        ssim_ok = not ENABLE_SSIM_FILTER       # Mặc định OK nếu không bật filter
                        brisque_ok = not ENABLE_BRISQUE_FILTER # Mặc định OK nếu không bật filter

                        log_details = [] # Thu thập thông tin chi tiết để in
                        avg_diff_cm, max_diff_cm, similarity_index, brisque_score = None, None, None, None

                        # 1. Đánh giá Contour (nếu bật)
                        if ENABLE_CONTOUR_FILTER:
                            if contour_gen is None:
                                log_details.append("ContourGen:⛔")
                                contour_ok = False
                            else:
                                avg_diff, max_diff = compare_contours(contour_orig, contour_gen, shape_orig)
                                if avg_diff is not None:
                                    avg_diff_cm = avg_diff * pixel_to_cm
                                    max_diff_cm = max_diff * pixel_to_cm
                                    log_details.append(f"ΔTB:{avg_diff_cm:.2f}cm")
                                    log_details.append(f"ΔMAX:{max_diff_cm:.2f}cm")
                                    if avg_diff_cm <= MAX_ACCEPTABLE_AVG_DIFF_CM and max_diff_cm <= MAX_ACCEPTABLE_MAX_DIFF_CM:
                                        contour_ok = True
                                        log_details[-1] += "✅" # Thêm tick vào log cuối
                                        log_details[-2] += "✅"
                                    else:
                                        contour_ok = False
                                        log_details[-1] += "⛔" # Thêm X vào log cuối
                                        log_details[-2] += "⛔"
                                else:
                                    log_details.append("ContourCmp:⚠️") # Không so sánh được
                                    contour_ok = False
                        else:
                            log_details.append("ContourFilt:OFF")

                        # 2. Đánh giá SSIM (nếu bật)
                        if ENABLE_SSIM_FILTER:
                            try:
                                if original_image_np_gray.shape == result_resized_np_gray.shape:
                                    similarity_index = ssim(original_image_np_gray, result_resized_np_gray, data_range=255)
                                    log_details.append(f"SSIM:{similarity_index:.3f}")
                                    if similarity_index >= MIN_ACCEPTABLE_SSIM:
                                        ssim_ok = True
                                        log_details[-1] += "✅"
                                    else:
                                        ssim_ok = False
                                        log_details[-1] += "⛔"
                                else:
                                    log_details.append("SSIM:⚠️(Size)")
                                    ssim_ok = False
                            except Exception as ssim_e:
                                log_details.append(f"SSIM:⚠️(Err)")
                                print(f"        Lỗi SSIM: {ssim_e}")
                                ssim_ok = False
                        else:
                             log_details.append("SSIMFilt:OFF")

                        # 3. Đánh giá BRISQUE (nếu bật và đã cài đặt)
                        if ENABLE_BRISQUE_FILTER:
                            try:
                                import brisque # Import ở đây để tránh lỗi nếu chưa cài
                                brisque_score = brisque.score(result_resized_np_gray)
                                log_details.append(f"BRISQUE:{brisque_score:.2f}")
                                if brisque_score <= MAX_ACCEPTABLE_BRISQUE:
                                    brisque_ok = True
                                    log_details[-1] += "✅"
                                else:
                                    brisque_ok = False
                                    log_details[-1] += "⛔"
                            except ImportError:
                                log_details.append("BRISQUE:⚠️(NotInstalled)")
                                brisque_ok = True # Coi như OK nếu chưa cài
                            except Exception as brisque_e:
                                log_details.append(f"BRISQUE:⚠️(Err)")
                                print(f"        Lỗi BRISQUE: {brisque_e}")
                                brisque_ok = False # Coi là lỗi nếu có lỗi tính toán
                        # else: # Không cần thêm log nếu filter OFF


                        # --- Quyết định cuối cùng ---
                        is_accepted = contour_ok and ssim_ok and brisque_ok

                        gen_end_time = time.time()
                        log_status = "✅ ACCEPTED" if is_accepted else "⛔ REJECTED"
                        log_string = f"       -> {log_status} | {' | '.join(log_details)} | (Time: {gen_end_time - gen_start_time:.2f}s)"
                        print(log_string)

                        # --- Lưu vào thư mục tương ứng ---
                        if is_accepted:
                            total_accepted_count += 1
                            accepted_count_per_image += 1
                            out_path = os.path.join(sub_output_accepted_dir, out_name)
                            result_resized_pil.save(out_path)
                        else:
                            total_rejected_count += 1
                            out_path = os.path.join(sub_output_rejected_dir, out_name)
                            result_resized_pil.save(out_path)

                    # --- Xử lý lỗi trong quá trình Inference ---
                    except torch.cuda.OutOfMemoryError:
                         print(f"       ❌ Lỗi OutOfMemoryError khi tạo {out_name}! Có thể cần giảm pipeline_size hoặc dùng CPU offload.")
                         total_error_count += 1
                         error_count_per_image += 1
                         if device == 'cuda': torch.cuda.empty_cache() # Cố gắng giải phóng bộ nhớ
                         time.sleep(1)
                         continue # Bỏ qua lần lặp này
                    except Exception as infer_e:
                        gen_end_time = time.time()
                        print(f"       ❌ Lỗi không xác định khi tạo {out_name}: {infer_e} (Time: {gen_end_time - gen_start_time:.2f}s)")
                        total_error_count += 1
                        error_count_per_image += 1
                        continue # Bỏ qua lần lặp này

        print(f"    ✅ Hoàn thành tạo các phiên bản cho {base_name}.")

    # --- Xử lý lỗi khi đọc/chuẩn bị ảnh gốc ---
    except FileNotFoundError:
        print(f"❌ Lỗi FileNotFoundError: Không tìm thấy file ảnh: {img_path}")
        total_error_count += 1
    except UnidentifiedImageError:
         print(f"❌ Lỗi UnidentifiedImageError: Không thể đọc hoặc file bị hỏng: {img_path}")
         total_error_count += 1
    except Exception as prep_e:
        print(f"❌ Lỗi không mong muốn khi xử lý file {base_name}: {prep_e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi để debug
        total_error_count += 1

    # --- Tổng kết cho ảnh hiện tại ---
    img_process_end_time = time.time()
    print(f"    ⏱️ Hoàn thành xử lý {base_name} sau {img_process_end_time - img_process_start_time:.2f} giây.")
    print(f"       Kết quả: {accepted_count_per_image} chấp nhận, {generation_count_per_image - accepted_count_per_image - error_count_per_image} bị loại, {error_count_per_image} lỗi.")
    print("-" * 60)

    # --- Giải phóng bộ nhớ GPU ---
    # Xóa các biến không cần thiết nữa
    del original_image_pil, original_image_np_gray, contour_orig
    del pipeline_input_image_pil, pipeline_input_image_np, pipeline_input_gray_np, canny_image_pil
    # Gọi garbage collector và làm trống cache CUDA
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

# =============================================================
# ===                   TỔNG KẾT CUỐI CÙNG                 ===
# =============================================================
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"\n{'='*25} HOÀN THÀNH TẤT CẢ {'='*25}")
print(f"Tổng thời gian thực thi: {total_duration:.2f} giây ({total_duration/60:.2f} phút)")
print("-" * 65)
print("Thống kê:")
print(f"  - Tổng số ảnh gốc đã xử lý: {len(selected_images_paths)}")
print(f"  - Tổng số ảnh đã tạo:      {total_generated_count}")
print(f"  - Số ảnh được chấp nhận:   {total_accepted_count}")
print(f"  - Số ảnh bị loại bỏ:      {total_rejected_count}")
print(f"  - Số lần tạo ảnh gặp lỗi: {total_error_count}")
print(f"  - Số ảnh gốc bị lỗi contour: {total_contour_fail_count}")
print("-" * 65)
print(f"Kết quả đã được lưu vào thư mục gốc: {os.path.abspath(output_dir_base)}")
print(f"   - Ảnh chấp nhận: {os.path.abspath(output_accepted_dir)}")
print(f"   - Ảnh bị loại bỏ: {os.path.abspath(output_rejected_dir)}")
print("===================================================================")