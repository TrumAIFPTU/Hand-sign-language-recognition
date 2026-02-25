import cv2
import numpy as np
import mediapipe as mp
import os
import joblib
import warnings
import pyttsx3
import threading

warnings.filterwarnings("ignore")
from src.features.features_extract import extract_landmarks

# --------------------------
# Cấu Hình TTS (Đọc văn bản)
# --------------------------
# Khởi tạo engine TTS trên luồng riêng để không đứng hình camera
engine = pyttsx3.init()
def speak_text(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

# --------------------------
# Cấu Hình MediaPipe (Chế độ Ảnh Tĩnh)
# --------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# static_image_mode=True: Đoán chính xác 1 frame hình tĩnh độc lập
hands_detector = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --------------------------
# Khởi tạo Mô Hình
# --------------------------
MODEL_DIR = 'model_saved/moe_hybrid_clf.pkl'
if not os.path.exists(MODEL_DIR):
    print(f"[LỖI] Không tìm thấy file mô hình tại {MODEL_DIR}")
    print("Vui lòng chạy 'python main.py' để huấn luyện mô hình trước!")
    exit(1)

artifacts = joblib.load(MODEL_DIR)
model_general = artifacts['model_general']
experts = artifacts['experts']
expert_configs = artifacts['expert_configs']
le = artifacts['label_encoder']

# Features order từ lúc training
train_feature_names = artifacts.get('feature_names', None)

# --------------------------
# Vòng Lặp Chính
# --------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    last_prediction = ""
    sentence = ""
    is_processing = False
    
    print("[THÔNG BÁO] App Point & Shoot đã mở.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Lật gương cho tự nhiên
        display_frame = frame.copy()
        h, w, c = frame.shape
        
        # --- PHASE: VẼ UI VÀ HƯỚNG DẪN BÊN TRÁI MÀN HÌNH ---
        # Panel mờ bên trái
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (380, 250), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Text Hướng dẫn
        instructions = [
            "Press 's' to Scan/Capture",
            "Press 'Enter' to Speak text",
            "Press 'Backspace' to Delete",
            "Press 'Space' to Add Space",
            "Press 'c' to Clear All",
            "Press 'q' to Quit"
        ]
        y_offset = 40
        for text in instructions:
            if "'s'" in text:
                color = (0, 255, 0) # Xanh lá
            elif "'q'" in text:
                color = (0, 0, 255) # Đỏ
            else:
                color = (255, 255, 255) # Trắng
            cv2.putText(display_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35

        # Khung Camera cho người dùng căn tay
        box_w, box_h = 350, 400
        x_center, y_center = w // 2 + 100, h // 2
        top_left = (x_center - box_w//2, y_center - box_h//2)
        bottom_right = (x_center + box_w//2, y_center + box_h//2)
        
        cv2.rectangle(display_frame, top_left, bottom_right, (100, 255, 100), 2)
        cv2.putText(display_frame, "Place hand here & Press 's' to Scan", 
                    (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # --- PHASE: HIỂN THỊ CÂU (SENTENCE) VÀ KẾT QUẢ ĐANG TÍNH ---
        cv2.rectangle(display_frame, (0, h-120), (w, h), (30, 30, 30), cv2.FILLED)
        
        if is_processing:
            cv2.putText(display_frame, "DANG QUET...", (w//2, h//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
                        
        # Hiển thị Sentence
        cv2.putText(display_frame, "Sentence: ", (20, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        cv2.putText(display_frame, sentence, (320, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        
        cv2.imshow("Hand Sign AI - Point & Shoot", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # --- BỘ ĐIỀU KHIỂN BÀN PHÍM ---
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
        elif key == 32: # Phím Space -> Thêm dấu cách
            sentence += " "
        elif key == 8: # Phím Backspace -> Xóa 1 ký tự cuối
            sentence = sentence[:-1]
        elif key == 13: # Phím Enter -> Đọc câu
            if sentence.strip():
                speak_text(sentence)
        elif key == ord('s'): # Phím 's': Quét 1 frame tĩnh
            is_processing = True
            
            # Ép vẽ UI "ĐANG QUÉT" trước khi CPU bị treo vì tính HOG
            cv2.putText(display_frame, "DANG QUET...", (w//2, h//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
            cv2.imshow("Hand Sign AI - Point & Shoot", display_frame)
            cv2.waitKey(1)
            
            # 1. Trộn MediaPipe và HOG
            features = extract_landmarks(frame)
            
            if features is not None:
                # 2. Xử lý qua numpy siêu tốc
                X_array = np.array(features, dtype=np.float32).reshape(1, -1)
                
                # 3. XGBoost Tier 1
                y_pred_xgb = model_general.predict(X_array)[0]
                pred_label_str = le.inverse_transform([y_pred_xgb])[0]
                
                # 4. SVM Tier 2
                final_label_str = pred_label_str
                for exp_name, config in expert_configs.items():
                    if pred_label_str in config['classes']:
                        expert_model = experts.get(exp_name)
                        if expert_model:
                            weapons = config['weapons']
                            if train_feature_names:
                                weapon_indices = [train_feature_names.index(w) for w in weapons if w in train_feature_names]
                                row_weapons = X_array[:, weapon_indices]
                            else:
                                # Fallback nếu model cũ không có feature_names
                                import pandas as pd
                                df_input = pd.DataFrame([features])
                                if getattr(model_general, 'feature_names_in_', None) is not None:
                                    df_input.columns = model_general.feature_names_in_
                                row_weapons = df_input[weapons]
                                
                            final_label_str = expert_model.predict(row_weapons)[0]
                        break
                
                last_prediction = final_label_str
                # Tự động gắn kết quả tĩnh vào câu
                if final_label_str != "del" and final_label_str != "space" and final_label_str != "nothing":
                    sentence += final_label_str
                elif final_label_str == "space":
                    sentence += " "
                elif final_label_str == "del":
                    sentence = sentence[:-1]
            else:
                last_prediction = "Khong tim thay tay"
                
            is_processing = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
