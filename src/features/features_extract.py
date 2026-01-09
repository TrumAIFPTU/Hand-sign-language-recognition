import os 
import cv2
import glob
from tqdm import tqdm
import csv
import numpy as np



def image_preprocessing(img):
    
    gray_scales = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_scales,(5,5),2)
    thresh = cv2.adaptiveThreshold(gaussian_blur,
                                maxValue=255, # giá trị pixel sau threshold  
                                adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # cách tính ngưỡng cục bộ
                                thresholdType= cv2.THRESH_BINARY_INV, # Quy tắc nhị phân hóa pixel > 0 -> đen và < threshold -> Trắng
                                blockSize = 11,
                                C = 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

def extract_contour(binary_img):
    contours, _ = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0: return None

    c = max(contours,key=cv2.contourArea)

    # getting area and perimeter of Hand
    area = cv2.contourArea(c) 
    perimeter = cv2.arcLength(c,True)

    if area < 1000: return None

    # getting SHAPE descriptions

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area > 0 else 0

    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h

    # getting HU MOMENTS

    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = hu_moments.flatten()

    hu_features = []
    for i in range(0,7):
        hu_features.append(-1 * np.copysign(1.0,hu_moments[i]) * np.log10(abs(hu_moments[i] + 1e-10)))
    
    features = [area,perimeter,solidity,aspect_ratio] + hu_features
    return features

