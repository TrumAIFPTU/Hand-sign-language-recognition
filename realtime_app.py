import cv2
import numpy as np
import mediapipe as mp
import time
import os
import joblib
import pandas as pd
from collections import deque
import threading
import pyttsx3
import warnings
warnings.filterwarnings("ignore")

from src.features.features_extract import extract_landmarks

# --------------------------
# Trình Đọc TTS (Text to Speech)
# --------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_text(text):
    if text.strip() == "": return
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# --------------------------
# Cấu Hình MediaPipe
# --------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------
# Khởi tạo Mô Hình
# --------------------------
MODEL_DIR = 'model_saved/moe_hybrid_clf.pkl'
if not os.path.exists(MODEL_DIR):
    print(f"[LỖI] Không tìm thấy mô hình tại {MODEL_DIR}")
    exit(1)

print("[THÔNG BÁO] Đang tải mô hình...")
artifacts = joblib.load(MODEL_DIR)
model_general = artifacts['model_general']
experts = artifacts['experts']
expert_configs = artifacts['expert_configs']
le = artifacts['label_encoder']

# Lấy danh sách feature chuẩn mà XGBoost được huấn luyện
train_feature_names = getattr(model_general, 'feature_names_in_', None)

# --------------------------
# Hàm Xử Lý Ảnh Môi Trường
# --------------------------
def auto_brightness_contrast(img):
    """ Tự động cân bằng sáng nếu môi trường quá tối hoặc sáng """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val < 50:  # Quá tối
        alpha, beta = 1.5, 30
    elif mean_val > 200: # Quá sáng
        alpha, beta = 0.8, -30
    else:
        return img
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def blur_background(img):
    """ Làm mờ nền để tập trung vào tay và người """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = segmentation.process(img_rgb)
    
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = cv2.GaussianBlur(img, (55, 55), 0)
    # Lấy background hơi tối một chút để tay nổi bật
    bg_image = cv2.convertScaleAbs(bg_image, alpha=0.7, beta=0) 
    
    output_image = np.where(condition, img, bg_image)
    return output_image

# --------------------------
# Vòng Lặp Chính
# --------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # -- Biến Trạng Thái Từ Vựng & Câu --
    prediction_buffer = deque(maxlen=15) # Bộ đệm lưu 15 frame liên tiếp để tránh nháy
    current_sentence = ""
    last_word = ""
    
    # -- Biến Hệ Thống & Sleep Mode --
    last_hand_time = time.time()
    sleep_mode = False
    
    print("[THÔNG BÁO] Hệ thống đã sẵn sàng!")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # Lật gương
        current_time = time.time()
        
        # --- PHASE: KIỂM TRA SLEEP MODE ---
        if current_time - last_hand_time > 30:
            sleep_mode = True
        else:
            sleep_mode = False
            
        if sleep_mode:
            # Chạy nhàn rỗi (Low FPS)
            cv2.putText(frame, "SLEEP MODE", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.putText(frame, "Raise your hand to wake up", (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Hand Sign AI AI", frame)
            
            # Thỉnh thoảng mới dùng tay thăm dò để đỡ tốn CPU
            results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                last_hand_time = current_time 
                
            time.sleep(0.5) 
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # --- PHASE: TIỀN XỬ LÝ & NHẬN DIỆN ---
        frame = auto_brightness_contrast(frame)
        display_frame = blur_background(frame)
        
        h, w, c = frame.shape
        
        # Vẽ Ghost Frame (Khung người dùng định hướng tay)
        cv2.rectangle(display_frame, (w//2 - 200, h//2 - 250), (w//2 + 200, h//2 + 150), (100, 255, 100), 2)
        cv2.putText(display_frame, "Place hand here", (w//2 - 90, h//2 - 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

        # Xử lý MediaPipe lấy landmark
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        
        predicted_char = ""
        
        if results.multi_hand_landmarks:
            last_hand_time = current_time # Đánh thức / Giữ thức
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks mờ mờ cho UI sinh động (Real-time tracking loop)
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # Sử dụng hàm cũ trong pipeline của bạn để lấy data
            features = extract_landmarks(frame)
            
            if features is not None:
                # Trích xuất đúng tên cột nếu có
                if train_feature_names is not None:
                    df_input = pd.DataFrame([features], columns=train_feature_names)
                else:
                    df_input = pd.DataFrame([features])
                
                # Inference Tier 1: XGBoost (Tổng quát)
                y_pred_xgb = model_general.predict(df_input)[0]
                pred_label_str = le.inverse_transform([y_pred_xgb])[0]
                
                # Inference Tier 2: SVM (Chuyên gia nếu gặp khó khăn)
                final_label_str = pred_label_str
                
                for exp_name, config in expert_configs.items():
                    if pred_label_str in config['classes']:
                        expert_model = experts.get(exp_name)
                        if expert_model:
                            weapons = config['weapons']
                            row_weapons = df_input[weapons]
                            final_label_str = expert_model.predict(row_weapons)[0]
                        break
                
                predicted_char = final_label_str
                prediction_buffer.append(predicted_char)
                
                # Kiểm tra Word Buffer: Chỉ nhận từ nếu 10/15 frame liên tiếp đều dự đoán chung 1 chữ
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    count = prediction_buffer.count(most_common)
                    
                    if count >= 10 and most_common != last_word:
                        if most_common == "space" or most_common == "nothing":
                            current_sentence += " "
                            last_word = "space"
                        elif most_common == "del":
                            current_sentence = current_sentence[:-1]
                            last_word = "del"
                        else:
                            current_sentence += most_common
                            last_word = most_common
                        
                        # Xóa buffer để nhận diện từ tiếp theo tránh bị kẹp phím
                        prediction_buffer.clear()
                        
                        # Nếu câu quá dài, tự xén
                        if len(current_sentence) > 40:
                            current_sentence = current_sentence[-40:]
                            
        else:
            # Ko thấy tay -> clear buffer để thả lỏng
            prediction_buffer.clear()
            last_word = ""

        # --- PHASE: HIỂN THỊ KẾT QUẢ & UI ---
        # Panel dưới cùng hiển thị câu
        cv2.rectangle(display_frame, (0, h-100), (w, h), (0, 0, 0), cv2.FILLED)
        cv2.putText(display_frame, f"Sentence: {current_sentence}", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Chữ đang nhận diện thời gian thực thả trôi lơ lửng gần Ghost Frame
        if predicted_char:
            cv2.putText(display_frame, f"[{predicted_char}]", (w//2 + 220, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
            
        # Nút ảo Hướng dẫn
        cv2.putText(display_frame, "Press 'Enter' to Speak text", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'Backspace' to Delete", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display_frame, "Press 'Space' to Add Space", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display_frame, "Press 'q' to Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 200), 2)

        cv2.imshow("Hand Sign AI", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13: # Enter: Phát tiếng
            speak_text(current_sentence)
        elif key == 8: # Backspace: Xóa kí tự
            current_sentence = current_sentence[:-1]
        elif key == 32: # Phím Space: Dấu cách
            current_sentence += " "
        elif key == ord('c'): # Clear: Xóa toàn bộ
            current_sentence = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
