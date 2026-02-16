import cv2
import mediapipe as mp
import os 
import pandas as pd
import numpy as np
import math

DATA_DIR = './Datasets/raw/asl_alphabet_train'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, max_num_hands=1, min_detection_confidence=0.5)

def calculate_distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def calculate_angle(p1,p2):
    return math.atan2(p2.y-p1.y,p2.x-p1.x)

def extract_landmarks(image_path):

    img = cv2.imread(image_path)

    if img is None: 
        print(f"Error in reading: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks is not None:
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks_list = []

        # landmarks được tính tọa độ tự cổ tay bắt đầu từ điểm 0

        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        for lm in hand_landmarks.landmark:

            landmarks_list.append(lm.x - wrist_x) # vị trí 
            landmarks_list.append(lm.x - wrist_x) # vị trí
            landmarks_list.append(lm.z) # độ sâu
        
        row_data = landmarks_list

        thumb_tip = hand_landmarks.landmark[4]
        index_mcp = hand_landmarks.landmark[5]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        middle_mcp = hand_landmarks.landmark[9]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Thêm khoảng cách giữa các ngón

        row_data.append(calculate_distance(thumb_tip, index_tip))
        row_data.append(calculate_distance(thumb_tip, middle_tip))
        row_data.append(calculate_distance(thumb_tip, ring_tip))
        row_data.append(calculate_distance(thumb_tip, pinky_tip))
        row_data.append(calculate_distance(index_tip, middle_tip))

        # Giải quyết vấn đề của nhận diện chữ U và R
        # Ở đây do là 2 ngón trỏ và giữa của U và R chỉ khác mỗi vị trí nên khoảng cách của nó hoàn toàn là bằng nhau
        # -> Ta sử dụng một biến để kiểm tra xem 2 vị trí của 2 ngón này có đổi cho nhau không
        diff_index_middle_x = index_tip.x - middle_tip.x
        row_data.append(diff_index_middle_x)

        # Giải quyết vấn đề của 3 chữ cái G,P,Q 
        # Ở đây ta có thể thấy 3 chữ cái này đều có phần ngón trỏ được chỉ ra khác mỗi góc độ và các vị trí ngón khác
        # Nhận thấy P và Q  Ngón giữa và cái của 2 chữ cái này một gần và một xa 
        dist_thumb_middle_mcp = calculate_distance(thumb_tip,middle_mcp)
        row_data.append(dist_thumb_middle_mcp)
        # G và 2 chữ còn lại khác nhau ở hướng ngón trỏ
        index_angle = calculate_angle(index_mcp,index_tip)
        row_data.append(index_angle)
        # Giải quyết vấn đề của D và P và Q vì khác hướng ngón trỏ
        diff_index_wrist_y = index_tip.y - wrist_y
        row_data.append(diff_index_wrist_y)
        return row_data
    
    else: return None


    
        
