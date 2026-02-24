import cv2
import numpy as np
from skimage.feature import hog 

def extract_hu_moments(contour):
    """Trích xuất 7 giá trị Hu Moments từ đường viền và biến đổi Log để dễ scale"""
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    # Biến đổi logarit để các giá trị không bị quá nhỏ
    hu_moments_log = np.zeros(7)
    for i in range(0, 7):
        if hu_moments[i] != 0:
            hu_moments_log[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
    return hu_moments_log

def main():
    cap = cv2.VideoCapture(0)
    
    # Định nghĩa dải màu da trong không gian HSV (có thể cần tinh chỉnh tùy ánh sáng phòng bạn)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    print("Nhấn 'q' để thoát chương trình.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Lật ảnh như gương để dễ thao tác
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400] # Tạo một vùng khung tĩnh (Region of Interest) để đặt tay vào
        
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, "Dat tay vao khung xanh", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 1. Tiền xử lý: Chuyển sang HSV và lọc màu da
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Làm mịn mask để khử nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 2. Tìm đường viền (Contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Lấy contour lớn nhất (giả sử đó là bàn tay)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Chỉ xử lý nếu diện tích đủ lớn (tránh nhiễu nhỏ)
            if cv2.contourArea(max_contour) > 3000:
                # Vẽ contour lên ROI để trực quan
                cv2.drawContours(roi, [max_contour], -1, (0, 0, 255), 2)
                
                # Cắt phần bounding box ôm sát bàn tay
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                hand_roi_mask = mask[y:y+h, x:x+w]
                
                # Resize về một kích thước cố định để tính HOG (ví dụ 64x64)
                hand_roi_resized = cv2.resize(hand_roi_mask, (64, 64))

                # --- 3. TRÍCH XUẤT ĐẶC TRƯNG ---
                
                # 3.1. Hu Moments
                hu_features = extract_hu_moments(max_contour)
                
                # 3.2. HOG (Histogram of Oriented Gradients)
                hog_features, hog_image = hog(hand_roi_resized, orientations=9, pixels_per_cell=(8, 8),
                                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                
                # Nối tất cả thành 1 Vector đặc trưng duy nhất
                final_feature_vector = np.concatenate((hu_features, hog_features))
                
                # Hiển thị thông số lên màn hình
                cv2.putText(frame, f"Hu Moments: 7 dims", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"HOG Features: {len(hog_features)} dims", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Total Vector: {len(final_feature_vector)} dims", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Trực quan hóa HOG (tùy chọn để xem máy tính nhìn thấy gì)
                hog_image_rescaled = (hog_image * 255).astype(np.uint8)
                cv2.imshow('HOG Vision', hog_image_rescaled)

        # Hiển thị kết quả
        cv2.imshow('Sign Language Feature Extraction', frame)
        cv2.imshow('Binary Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()