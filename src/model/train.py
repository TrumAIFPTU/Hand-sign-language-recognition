import pandas as pd
import numpy as np
import os
import joblib
from src.utils.paths import PREPROCESS_DIR
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import xgboost as xgb

MODEL_DIR = 'model_saved/moe_hybrid_clf.pkl'

def read_datasets():
    train_path = os.path.join(PREPROCESS_DIR, 'train_features.csv')
    test_path = os.path.join(PREPROCESS_DIR, 'test_features.csv')
    df_train = pd.read_csv(train_path).dropna(subset=['Label'])
    df_test = pd.read_csv(test_path).dropna(subset=['Label'])
    return df_train, df_test

def train_model():
    df_train, df_test = read_datasets()

    X_train = df_train.drop('Label', axis=1)
    y_train_raw = df_train['Label'].astype(str)
    X_test = df_test.drop('Label', axis=1)
    y_test_raw = df_test['Label'].astype(str)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_raw)
    y_test_encoded = le.transform(y_test_raw)

    print("--- Đang huấn luyện Tầng 1: XGBoost ---")
    model_general = xgb.XGBClassifier(
        max_depth=8,              
        learning_rate=0.03,       
        n_estimators=700,         
        tree_method='hist',
        device='cuda', 
        random_state=42
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
    for name, config in expert_configs.items():
        print(f"--- Đang huấn luyện SVM cho cụm: {config['classes']} ---")
        df_sub = df_train[df_train['Label'].isin(config['classes'])]
        
        kernel_type = 'rbf' 
        
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel_type, C=10, class_weight='balanced', random_state=42, probability=True)
        )
        clf.fit(df_sub[config['weapons']], df_sub['Label'])
        trained_experts[name] = clf

    print("--- Đang đánh giá hệ thống Hybrid ---")
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

