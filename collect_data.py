import cv2
import os
import time

DATA_DIR = 'Datasets/raw/test_datasets/new_test'
os.makedirs(DATA_DIR,exist_ok=True)


cap = cv2.VideoCapture(0) # Mở webcam

print("Nhấn A-Z để lưu ảnh cho nhãn đó")
print("Hãy mở Capslock")
print("Nhấn 'q' để thoát")

counter = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) # Lật ngược gương cho dễ nhìn

    x1, y1, x2, y2 = 100, 10, 300, 200
    
    # Vẽ ô vuông lên màn hình để biết chỗ đặt tay
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Cắt vùng ảnh trong ô vuông ra để xử lý
    roi = frame[y1:y2, x1:x2]
    
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi) # Hiển thị riêng vùng tay

    key = cv2.waitKey(1) & 0xFF
    
    if key != 255:print(key)
    if 65 <= key <= 90: 
        label = chr(key)

        label_dir = os.path.join(DATA_DIR,label)
        os.makedirs(label_dir,exist_ok=True)

        counter[label] = counter.get(label,0)
        file_name = f"{label}{counter[label]}.jpg"
        file_path = os.path.join(label_dir,file_name)

        cv2.imwrite(file_path,roi)
        counter[label] += 1

        print(f"Saved: {file_path}")
        time.sleep(0.2)

    if key == 49:
        label = 'del'

        label_dir = os.path.join(DATA_DIR,label)
        os.makedirs(label_dir,exist_ok=True)

        counter[label] = counter.get(label,0)
        file_name = f"{label}{counter[label]}.jpg"
        file_path = os.path.join(label_dir,file_name)

        cv2.imwrite(file_path,roi)
        counter[label] += 1

        print(f"Saved: {file_path}")
        time.sleep(0.2)
    
    if key == 50: 
        label = 'nothing'

        label_dir = os.path.join(DATA_DIR,label)
        os.makedirs(label_dir,exist_ok=True)

        counter[label] = counter.get(label,0)
        file_name = f"{label}{counter[label]}.jpg"
        file_path = os.path.join(label_dir,file_name)

        cv2.imwrite(file_path,roi)
        counter[label] += 1

        print(f"Saved: {file_path}")
        time.sleep(0.2)

    if key == 51:
        label = 'space'

        label_dir = os.path.join(DATA_DIR,label)
        os.makedirs(label_dir,exist_ok=True)

        counter[label] = counter.get(label,0)
        file_name = f"{label}{counter[label]}.jpg"
        file_path = os.path.join(label_dir,file_name)

        cv2.imwrite(file_path,roi)
        counter[label] += 1

        print(f"Saved: {file_path}")
        time.sleep(0.2)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()