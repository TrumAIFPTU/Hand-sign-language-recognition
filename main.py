import os 
import cv2
import csv
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Giả sử các module này bạn đã viết chuẩn
from src.data import download_datasets
from src.features import extract_landmarks
from src.model.train import train_model
import random # Thêm thư viện random cho Augmentation
from joblib import Parallel, delayed # Thêm xử lý đa luồng

def get_column_names():
    cols = ['Label']
    
    expert_cols = [
        'thumb_depth', 'dist_thumb_mid', 'thumb_offset_x', 'thumb_ratio',
        'cross_direction_x', 'ext_ring_finger', 'dist_thumb_mid_pip', 'curl_idx',
        'mn_diff', 'finger_orientation', 'pinky_curl',
        'd_tip', 'gap_y'  
    ]
    cols.extend(expert_cols)
    
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    for i in range(324):
        cols.append(f'hog_{i}')
        
    return cols
def augment_image(image):
    """
    Thực hiện Data Augmentation (Tăng cường dữ liệu) đơn giản bằng OpenCV.
    Trả về danh sách các ảnh đã được biến đổi (Bao gồm ảnh gốc).
    """
    aug_images = [image]
    
    # 1. Lật ảnh ngang (Flip Horizontal) - Biến tay phải thành tay trái
    # LƯU Ý: Rất cẩn thận với tập Sign Language! Một số chữ lật ngang sẽ mất ý nghĩa.
    # Trong ASL, đa số chữ cái dùng 1 tay thì lật ngang vẫn xài được (như J thì lật sẽ thành ngược).
    # Ta chỉ thêm độ sáng, độ tương phản để giữ cấu trúc chữ nguyên vẹn nhất!
    
    # 2. Thay đổi độ sáng ngẫu nhiên (Brightness)
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img_brightness = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    aug_images.append(img_brightness)
    
    # 3. Phóng to nhẹ (Zoom In - Scale) để mô phỏng đưa tay lại gần
    height, width = image.shape[:2]
    scale = random.uniform(1.05, 1.2)
    center_x, center_y = width / 2, height / 2
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
    img_zoomed = cv2.warpAffine(image, M, (width, height))
    aug_images.append(img_zoomed)
    
    return aug_images

def process_single_image(image_path, label):
    """
    Hàm xử lý cho 1 ảnh đơn lẻ để chạy đa luồng.
    Đọc ảnh, Augment, và Trích xuất Đặc trưng.
    Trả về list các rows (dòng data) hợp lệ.
    """
    rows = []
    img = cv2.imread(image_path)
    if img is None: return rows
    
    augmented_imgs = augment_image(img)
    for aug_img in augmented_imgs:
        features = extract_landmarks(aug_img)
        if features is not None:
            rows.append([label] + features)
    return rows

def extract_training_features():
    data_dir = 'Datasets/raw/asl_alphabet_train'
    
    # KAGGE FIX: Kaggle giải nén thư mục bị lồng vào trong (Ví dụ: asl_alphabet_train/asl_alphabet_train/A)
    # Ta phải chui vào thêm 1 lớp nếu thư mục đó xuất hiện
    if os.path.exists(os.path.join(data_dir, 'asl_alphabet_train')):
        data_dir = os.path.join(data_dir, 'asl_alphabet_train')
        
    output_dir = 'Datasets/preprocessing/train_features.csv'
    
    # CASH MEMORY: Khỏi chạy lại nửa tiếng nếu đã có sẵn
    if os.path.exists(output_dir):
        print(f"👉 [BỎ QUA] Đã tìm thấy tệp {output_dir}. Nhảy qua bước Trích xuất Features Train!")
        return
        
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    cols = get_column_names()

    # Thu thập toàn bộ đường dẫn ảnh và nhãn
    image_tasks = []
    if os.path.exists(data_dir):
        labels = os.listdir(data_dir)
        for label in labels:
            label_path = os.path.join(data_dir, label)
            if not os.path.isdir(label_path): continue
            
            for image_name in os.listdir(label_path):
                image_tasks.append((os.path.join(label_path, image_name), label))
                
    if not image_tasks:
        print("[THÔNG BÁO] Không tìm thấy dữ liệu Train. Bỏ qua bước Extract.")
        return

    print(f"Đang chuẩn bị trích xuất {len(image_tasks)} file ảnh gốc (Sẽ x3 nhờ Augmentation)...")
    
    # CẤU HÌNH TỐI ƯU CHO Intel i5-14600KF (20 threads)
    # Dùng 12 luồng để cân bằng tốc độ siêu nhanh và tính ổn định của Windows OS
    results = Parallel(n_jobs=12, batch_size=10)(
        delayed(process_single_image)(img_path, lbl) 
        for img_path, lbl in tqdm(image_tasks, desc="Extracting (Multi-core)")
    )
    
    # Gộp kết quả và Ghi ra file CSV
    total_images = 0
    with open(output_dir, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        
        for row_list in results:
            for row in row_list:
                writer.writerow(row)
                total_images += 1

    print(f"COMPLETED! EXTRACT TRAINING FEATURES {total_images} rows")
    print(f"Data saved: {output_dir}")
    
def extract_testing_features():
    test_dir = 'Datasets/raw/test_datasets/new_test'
    test_dir2 = 'Datasets/raw/asl_alphabet_test'
    
    # Tương tự như tập Train, nếu Kaggle giải nén bị lồng 2 thư mục
    if os.path.exists(os.path.join(test_dir2, 'asl_alphabet_test')):
        test_dir2 = os.path.join(test_dir2, 'asl_alphabet_test')
        
    output_dir = 'Datasets/preprocessing/test_features.csv'
    
    # CASH MEMORY 
    if os.path.exists(output_dir):
        print(f"👉 [BỎ QUA] Đã tìm thấy tệp {output_dir}. Nhảy qua bước Trích xuất Features Test!")
        return

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    cols = get_column_names()
    
    with open(output_dir, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        total_images = 0

        # --- Xử lý test_dir ---
        if os.path.exists(test_dir):
            labels = os.listdir(test_dir)
            for label in labels:
                label_path = os.path.join(test_dir, label)
                if not os.path.isdir(label_path): 
                    continue

                for image_name in tqdm(os.listdir(label_path), desc=f'Extracting {label} (test_dir)'):
                    image_path = os.path.join(label_path, image_name)
                    features = extract_landmarks(image_path)

                    if features is not None:
                        row = [label] + features
                        writer.writerow(row)
                        total_images += 1

        # --- Xử lý test_dir2 ---
        if os.path.exists(test_dir2):
            for image_name in tqdm(os.listdir(test_dir2), desc='Extracting test data (test_dir2)'):
                image_path = os.path.join(test_dir2, image_name)
                features = extract_landmarks(image_path)

                if features is not None:
                    real_label = image_name.split('_')[0] 
                    row = [real_label] + features
                    writer.writerow(row)
                    total_images += 1

        print("------------------------------------------------")

    print(f"COMPLETED! EXTRACT TESTING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")
    


def implement_model():
    model_dir = 'model_saved/moe_hybrid_clf.pkl'
    test_csv = 'Datasets/preprocessing/test_features.csv'
    
    if not os.path.exists(model_dir):
        print(f"LỖI: Không tìm thấy mô hình tại {model_dir}")
        return
    
    print(f"--- Đang tải mô hình từ {model_dir} ---")
    artifacts = joblib.load(model_dir)
    
    model_general = artifacts['model_general']
    experts = artifacts['experts']
    expert_configs = artifacts.get('expert_configs') 
    le = artifacts['label_encoder']
    
    if expert_configs is None:
        print("LỖI: File model thiếu key 'expert_configs'. Hãy kiểm tra lại file train.py!")
        return

    df_test = pd.read_csv(test_csv).dropna(subset=['Label'])
    
    # GUARD: Nếu test CSV rỗng thì bỏ qua
    if len(df_test) == 0:
        print("[CẢNH BÁO] Test CSV rỗng. Bỏ qua implement_model.")
        return
    
    X_test = df_test.drop('Label', axis=1)
    # BẮT BUỘC: Ép kiểu toàn bộ X về float32 (tránh lỗi 'object' dtype)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    y_test_raw = df_test['Label'].astype(str)
    y_test_encoded = le.transform(y_test_raw)
    
    print("--- Tầng 1: XGBoost đang xử lý tổng quát ---")
    y_pred_general = model_general.predict(X_test)
    y_pred_final = y_pred_general.copy()
    
    print("--- Tầng 2: SVM đang kiểm tra chéo các cụm nhạy cảm ---")
    
    for i in range(len(y_pred_general)):
        xgb_idx = y_pred_general[i]
        xgb_label = le.inverse_transform([xgb_idx])[0]
        
        target_expert_name = None
        for exp_name, config in expert_configs.items():
            if xgb_label in config['classes']:
                target_expert_name = exp_name
                break
        
        if target_expert_name and target_expert_name in experts:
            expert_model = experts[target_expert_name]
            weapons = expert_configs[target_expert_name]['weapons']
            
            # Chỉ trích xuất đúng các cột "vũ khí" mà chuyên gia này cần
            row_features = X_test.iloc[[i]][weapons] 
            
            # SVM ra quyết định cuối cùng
            svm_decision = expert_model.predict(row_features)[0]
            # Cập nhật lại kết quả vào mảng dự đoán cuối cùng
            y_pred_final[i] = le.transform([svm_decision])[0]

    # 6. Đánh giá và báo cáo
    acc = accuracy_score(y_test_encoded, y_pred_final)
    print(f"\n[KẾT QUẢ] Accuracy MoE Hybrid: {acc:.4f}")
    

    # 7. Vẽ Confusion Matrix (Chuẩn 29 lớp)
    cm = confusion_matrix(y_test_encoded, y_pred_final, labels=range(len(le.classes_)))
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (True)')
    plt.title('Confusion Matrix - Mixture of Experts (MoE) Final System')
    plt.show()
def main():
    # ============================================================
    # CÀI ĐẶT NHANH: Đặt True/False để bật/tắt từng bước
    # ============================================================
    SKIP_DOWNLOAD = False   # True = Bỏ qua bước tải Data từ Kaggle
    SKIP_EXTRACT  = False   # True = Bỏ qua bước trích xuất features (dùng lại CSV cũ)
    SKIP_TRAIN    = False   # True = Bỏ qua bước Train (dùng lại Model cũ)
    # ============================================================

    if not SKIP_DOWNLOAD:
        download_datasets()
    else:
        print("[BỎ QUA] Bước tải Data (SKIP_DOWNLOAD = True)")
    
    if not SKIP_EXTRACT:
        extract_training_features()
        extract_testing_features()
    else:
        print("[BỎ QUA] Bước trích xuất Features (SKIP_EXTRACT = True)")
    
    if not SKIP_TRAIN:
        train_model()
        implement_model()
    else:
        print("[BỎ QUA] Bước Train Model (SKIP_TRAIN = True)")

if __name__ == "__main__":
    main()