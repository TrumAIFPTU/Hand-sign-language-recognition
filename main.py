import os 
import cv2
import glob
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from src.data import download_datasets
from src.features import extract_landmarks
from src.model.train import train_model

def extract_training_features():
    data_dir = 'Datasets/raw/asl_alphabet_train'
    output_dir = 'Datasets/preprocessing/train_features.csv'

    cols = []
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    cols = ['Label'] + cols
    cols.extend([
    'dist_thumb_index', 'dist_thumb_middle', 'dist_thumb_ring', 'dist_thumb_pinky', 'dist_index_middle', # 5 cũ
    'diff_index_middle_x',  # Fix U/R
    'dist_thumb_middle_mcp', # Fix P/Q
    'index_angle', # Fix G/P/Q
    'diff_index_wrist_y'    # Fix Orientation
])

    with open(output_dir,mode='w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        
        labels = os.listdir(data_dir)
        total_images = 0

        for label in labels:
            label_path = os.path.join(data_dir,label)

            if not os.path.isdir(label_path): continue
            cnt = 0
            for image_name in tqdm(os.listdir(label_path),desc = f'extracting {label}'):
                image_path = os.path.join(label_path,image_name)
                img = cv2.imread(image_path)

                if img is None: 
                    print(f"Can't read {image_path}")
                    continue
                    
                features = extract_landmarks(image_path)

                if features is not None:
                    
                    row = [label] + features
                    writer.writerow(row)
                    total_images += 1
                    cnt+=1
            print(f"EXTRACT: {cnt} {label} images")
            print("------------------------------------------------")

    print(f"COMPLETED! EXTRACT TRAINING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")
    
def extract_testing_features():
    test_dir = 'Datasets/raw/test_datasets/new_test'
    output_dir = 'Datasets/preprocessing/test_features.csv'
    test_dir2 = 'Datasets/raw/asl_alphabet_test'

    cols = []
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    cols = ['Label'] + cols
    cols.extend([
    'dist_thumb_index', 'dist_thumb_middle', 'dist_thumb_ring', 'dist_thumb_pinky', 'dist_index_middle', # 5 cũ
    'diff_index_middle_x',  # Fix U/R
    'dist_thumb_middle_mcp', # Fix P/Q
    'index_angle', # Fix G/P/Q
    'diff_index_wrist_y'    # Fix Orientation
])
    
    with open(output_dir,mode='w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        labels = os.listdir(test_dir)
        total_images = 0



        for label in labels:
            label_path = os.path.join(test_dir,label)

            if not os.path.isdir(label_path): continue

            for image_name in tqdm(os.listdir(label_path),desc = f'extracting {label}'):
                image_path = os.path.join(label_path,image_name)
                img = cv2.imread(image_path)

                if img is None: 
                    print(f"Can't read {image_path}")
                    break
                    

                features = extract_landmarks(image_path)

                if features is not None:
                    row = [label] + features
                    writer.writerow(row)
                    total_images += 1

        for image_name in tqdm(os.listdir(test_dir2),desc = f'extracting test data'):
            image_path = os.path.join(test_dir2,image_name)
            img = cv2.imread(image_path)

            if img is None: 
                print(f"Can't read {image_path}")
                continue
                
            features = extract_landmarks(image_path)

            if features is not None:
                row = [label] + features
                writer.writerow(row)
                total_images += 1

        print("------------------------------------------------")


    print(f"COMPLETED! EXTRACT TESTING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")

def implement_model():
    model_dir = 'model_saved/random_forest_clf.pkl'
    test_dir = 'Datasets/preprocessing/test_features.csv'
    if not os.path.exists(model_dir):
        print("DO NOT EXIST MODEL")
    
    model = joblib.load(model_dir)
    df_test = pd.read_csv('Datasets/preprocessing/test_features.csv')
    X_test = df_test.drop('Label',axis=1)
    y_test = df_test['Label']
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_true=y_test,y_pred=y_pred)

    print(f"Accuracy: {acc:.2f}")


def main():

    #download_datasets
    extract_training_features()
    extract_testing_features()
    #train_model()
    #implement_model()

if __name__ == "__main__":
    main()