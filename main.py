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

def get_column_names():
    cols = ['Label']
    
    # 11 cột Expert
    expert_cols = [
        'thumb_depth', 'dist_thumb_mid', 'thumb_offset_x', 'thumb_ratio',
        'cross_direction_x', 'ext_ring_finger', 'dist_thumb_mid_pip', 'curl_idx',
        'mn_diff', 'finger_orientation', 'pinky_curl'
    ]
    cols.extend(expert_cols)
    
    # 63 cột Landmarks
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    # 324 cột HOG
    for i in range(324):
        cols.append(f'hog_{i}')
        
    return cols
def extract_training_features():
    data_dir = 'Datasets/raw/asl_alphabet_train'
    output_dir = 'Datasets/preprocessing/train_features.csv'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    cols = get_column_names()

    with open(output_dir, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        
        labels = os.listdir(data_dir)
        total_images = 0

        for label in labels:
            label_path = os.path.join(data_dir, label)
            if not os.path.isdir(label_path): 
                continue
            
            cnt = 0
            for image_name in tqdm(os.listdir(label_path), desc=f'Extracting {label}'):
                image_path = os.path.join(label_path, image_name)
                features = extract_landmarks(image_path)

                if features is not None:
                    row = [label] + features
                    writer.writerow(row)
                    total_images += 1
                    cnt += 1
                    
            print(f"EXTRACT: {cnt} {label} images")
            print("------------------------------------------------")

    print(f"COMPLETED! EXTRACT TRAINING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")
    
def extract_testing_features():
    test_dir = 'Datasets/raw/test_datasets/new_test'
    test_dir2 = 'Datasets/raw/asl_alphabet_test'
    output_dir = 'Datasets/preprocessing/test_features.csv'

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
    X_test = df_test.drop('Label', axis=1)
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
    # download_datasets()
    extract_training_features()
    extract_testing_features()
    train_model()
    implement_model()

if __name__ == "__main__":
    main()