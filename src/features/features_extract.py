import cv2
import mediapipe as mp
import math
import numpy as np
import os
import warnings
from skimage.feature import hog

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def calculate_3d_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def extract_landmarks(image_path_or_frame):
    """Trích xuất 400 đặc trưng từ ảnh hoặc frame camera (Không chứa Label)"""
    if isinstance(image_path_or_frame, str):
        img = cv2.imread(image_path_or_frame)
    else:
        img = image_path_or_frame
        
    if img is None: 
        return None
    
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        raw_pts = []
        x_pixel_coords, y_pixel_coords = [], []
        
        for lm in hand_landmarks.landmark:
            raw_pts.append((lm.x, lm.y, lm.z))
            x_pixel_coords.append(int(lm.x * w))
            y_pixel_coords.append(int(lm.y * h))

        base_x, base_y, base_z = raw_pts[0]
        norm_pts = [(pt[0] - base_x, pt[1] - base_y, pt[2] - base_z) for pt in raw_pts]
        
        flat_coords = [c for pt in norm_pts for c in pt]
        max_val = max(map(abs, flat_coords)) if max(map(abs, flat_coords)) != 0 else 1e-6
        norm_pts = [(pt[0]/max_val, pt[1]/max_val, pt[2]/max_val) for pt in norm_pts]

        t_depth = calculate_3d_distance(norm_pts[4], norm_pts[17])
        d_thumb_mid = calculate_3d_distance(norm_pts[4], norm_pts[13])
        dist_8_12 = calculate_3d_distance(norm_pts[8], norm_pts[12])
        
        mn_diff = norm_pts[16][1] - norm_pts[12][1]
        finger_orientation = norm_pts[8][1] - norm_pts[5][1]
        pinky_curl = calculate_3d_distance(norm_pts[20], norm_pts[0])

        # --- VỊ TRÍ 2: Tính toán 2 đặc trưng cho O/C ---
        d_tip = calculate_3d_distance(norm_pts[8], norm_pts[4])
        gap_y = norm_pts[4][1] - norm_pts[8][1]

        # --- VỊ TRÍ 3: Thêm vào danh sách expert_features ---
        expert_features = [
            t_depth,                                            # thumb_depth
            d_thumb_mid,                                        # dist_thumb_mid
            norm_pts[4][0] - norm_pts[13][0],                   # thumb_offset_x
            t_depth / (dist_8_12 + 1e-6),                       # thumb_ratio
            norm_pts[8][0] - norm_pts[12][0],                   # cross_direction_x
            calculate_3d_distance(norm_pts[16], norm_pts[17]),  # ext_ring_finger
            calculate_3d_distance(norm_pts[4], norm_pts[10]),   # dist_thumb_mid_pip
            calculate_3d_distance(norm_pts[5], norm_pts[8]),    # curl_idx
            mn_diff,                                            # mn_diff
            finger_orientation,                                 # finger_orientation
            pinky_curl,                                         # pinky_curl
            d_tip,                                              # d_tip (Mới)
            gap_y                                               # gap_y (Mới)
        ]

        raw_landmarks_flat = [coord for pt in norm_pts for coord in pt]

        margin = 20
        x_min, x_max = max(0, min(x_pixel_coords) - margin), min(w, max(x_pixel_coords) + margin)
        y_min, y_max = max(0, min(y_pixel_coords) - margin), min(h, max(y_pixel_coords) + margin)
        
        hand_crop = img[y_min:y_max, x_min:x_max]
        hog_features = [0.0] * 324 
        
        if hand_crop.size > 0:
            try:
                hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                hand_resized = cv2.resize(hand_gray, (64, 64))
                hog_features = list(hog(hand_resized, orientations=9, pixels_per_cell=(16, 16),
                                        cells_per_block=(2, 2), block_norm='L2-Hys'))
            except:
                pass

        # Tổng cộng: 13 + 63 + 324 = 400 features (Không tính Label)
        final_row = expert_features + raw_landmarks_flat + hog_features
        return final_row
        
    return None