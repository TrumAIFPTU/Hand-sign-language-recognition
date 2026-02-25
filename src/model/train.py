import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
from src.utils.paths import PREPROCESS_DIR
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import xgboost as xgb

# Callback tqdm cho XGBoost
class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="XGBoost (Trees)", unit="tree", ncols=80)
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False
    def after_training(self, model):
        self.pbar.close()
        return model

MODEL_DIR = 'model_saved/moe_hybrid_clf.pkl'

def read_datasets():
    train_path = os.path.join(PREPROCESS_DIR, 'train_features.csv')
    test_path = os.path.join(PREPROCESS_DIR, 'test_features.csv')
    df_train = pd.read_csv(train_path).dropna(subset=['Label'])
    df_test = pd.read_csv(test_path).dropna(subset=['Label'])
    return df_train, df_test

def train_model():
    df_train, df_test = read_datasets()

    # Tách X và y
    X_train = df_train.drop('Label', axis=1)
    y_train_raw = df_train['Label'].astype(str)
    X_test = df_test.drop('Label', axis=1)
    y_test_raw = df_test['Label'].astype(str)
    
    # BẮT BUỘC: Ép kiểu toàn bộ X về dạng Float/Numeric vì XGBoost không nhận 'object' (string)
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_raw)
    y_test_encoded = le.transform(y_test_raw)

    print("--- Đang huấn luyện Tầng 1: XGBoost ---")
    n_est = 700
    gpu_used = False
    
    # Thử 1: API mới (xgboost >= 2.0): device='cuda'
    if not gpu_used:
        try:
            model_general = xgb.XGBClassifier(
                max_depth=8, learning_rate=0.03, n_estimators=n_est,
                tree_method='hist', device='cuda', random_state=42
            )
            model_general.fit(X_train, y_train_encoded)
            gpu_used = True
            print("[THÀNH CÔNG] XGBoost GPU (API mới: device='cuda')!")
        except:
            pass
    
    # Thử 2: API cũ (xgboost < 2.0): tree_method='gpu_hist'
    if not gpu_used:
        try:
            model_general = xgb.XGBClassifier(
                max_depth=8, learning_rate=0.03, n_estimators=n_est,
                tree_method='gpu_hist', random_state=42
            )
            model_general.fit(X_train, y_train_encoded)
            gpu_used = True
            print("[THÀNH CÔNG] XGBoost GPU (API cũ: gpu_hist)!")
        except:
            pass
    
    # Thử 3: CPU fallback
    if not gpu_used:
        print("[CẢNH BÁO] Không tìm thấy GPU. Chạy bằng CPU...")
        model_general = xgb.XGBClassifier(
            max_depth=8, learning_rate=0.03, n_estimators=n_est,
            tree_method='hist', n_jobs=-1, random_state=42
        )
        model_general.fit(X_train, y_train_encoded)


    expert_configs = {
        'mn_expert': {
            'classes': ['M', 'N'],
            'weapons': ['mn_diff', 'pinky_curl', 'y16', 'y12','curl_idx', 'ext_ring_finger']
        },
        'hp_expert': {
            'classes': ['H', 'P'],
            'weapons': ['finger_orientation','y8', 'y5','x8', 'x5', 'y8', 'y5', 'z8', 'z5']
        },
        'vkw_expert': {
            'classes': ['V', 'K', 'W'],
            'weapons': ['thumb_depth', 'ext_ring_finger', 'dist_thumb_mid_pip']
        },
        'ur_expert': { 
            'classes': ['U', 'R'],
            'weapons': ['cross_direction_x', 'curl_idx', 'y8', 'y12']
        },
        'oc_expert': {
            'classes': ['O', 'C'],
            'weapons': ['d_tip', 'gap_y', 'thumb_depth', 'z8', 'z4'] # Bổ sung z để SVM thấy độ sâu
        }
    }
    
    trained_experts = {}
    for name, config in tqdm(expert_configs.items(), desc="SVM Experts", unit="cluster", ncols=80):
        print(f"\n  => Huấn luyện SVM cụm: {config['classes']}")
        df_sub = df_train[df_train['Label'].isin(config['classes'])]
        
        # BẢO VỆ CHỐNG TRÀN ZERO-LENGHT ARRAY KHI THIẾU DATA
        if len(df_sub) < 5:
            print(f"[CẢNH BÁO] Cụm {config['classes']} KHÔNG ĐỦ DỮ LIỆU ({len(df_sub)} ảnh). Đã bỏ qua.")
            continue
            
        kernel_type = 'rbf' 
        
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel_type, C=10, class_weight='balanced', random_state=42, probability=True)
        )
        clf.fit(df_sub[config['weapons']], df_sub['Label'])
        trained_experts[name] = clf

    print("--- Đang đánh giá hệ thống Hybrid ---")
    
    # GUARD: Bỏ qua Evaluation nếu Test Set rỗng (VD: chạy trên VM không có ảnh Test)
    if len(X_test) == 0 or len(y_test_encoded) == 0:
        print("[CẢNH BÁO] Không tìm thấy dữ liệu Test. Bỏ qua Evaluation, vẫn lưu Model.")
    else:
        y_pred_xgb = model_general.predict(X_test)
        y_pred_final = y_pred_xgb.copy()
        
        class_to_expert = {}
        for name, config in expert_configs.items():
            for cls in config['classes']:
                cls_idx = le.transform([cls])[0]
                class_to_expert[cls_idx] = name

        for i in range(len(y_pred_xgb)):
            xgb_res = y_pred_xgb[i]
            if xgb_res in class_to_expert:
                expert_name = class_to_expert[xgb_res]
                weapons = expert_configs[expert_name]['weapons']
                row_input = X_test.iloc[[i]][weapons]
                
                final_label = trained_experts[expert_name].predict(row_input)[0]
                y_pred_final[i] = le.transform([final_label])[0]

        acc = accuracy_score(y_test_encoded, y_pred_final)
        print(f"\n[FINAL HYBRID ACCURACY]: {acc:.4f}")
        
        print(classification_report(
        y_test_encoded, y_pred_final, 
        labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0
    ))

    # XOÁ CALLBACK TQDM TRƯỚC KHI LƯU (Tránh lỗi pickle TextIOWrapper)
    model_general.set_params(callbacks=None)
    
    artifacts = {
        'model_general': model_general,
        'experts': trained_experts,
        'expert_configs': expert_configs, # Lưu đúng tên key để implement_model không bị KeyError
        'label_encoder': le,
        'feature_names': X_train.columns.tolist()
    }
    
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    joblib.dump(artifacts, MODEL_DIR)
    print(f"Hệ thống đã lưu thành công tại: {MODEL_DIR}")

