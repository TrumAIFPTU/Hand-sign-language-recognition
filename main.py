import os 
import cv2
import glob
from tqdm import tqdm
import csv
import numpy as np
from src.data import download_datasets
from src.features import image_preprocessing,extract_contour

def extract_training_features():
    data_dir = 'Datasets/raw/asl_alphabet_train'
    output_dir = 'Datasets/preprocessing/data_features.csv'

    print(f"SCANING_PATH {data_dir}")
    headers = ['Label','Area','Perimeter','Solidity','AspectRatio',
            'Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7']


    with open(output_dir,mode='w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        labels = os.listdir(data_dir)
        total_images = 0

        for label in labels:
            label_path = os.path.join(data_dir,label)

            if not os.path.isdir(label_path): continue

            for image_name in tqdm(os.listdir(label_path),desc = f'extracting {label}'):
                image_path = os.path.join(label_path,image_name)
                img = cv2.imread(image_path)

                if img is None: 
                    print(f"Can't read {image_path}")
                    continue
                    
                processed_img = image_preprocessing(img)
                features = extract_contour(processed_img)

                if features:
                    row = [label] + features
                    writer.writerow(row)
                    total_images += 1
            print("------------------------------------------------")

    print(f"COMPLETED! EXTRACT TRAINING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")
    
def extract_testing_features():
    data_dir = 'Datasets/raw/asl_alphabet_test'
    output_dir = 'Datasets/preprocessing/test_features.csv'

    print(f"SCANING_PATH {data_dir}")
    headers = ['Label','Area','Perimeter','Solidity','AspectRatio',
            'Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7']


    with open(output_dir,mode='w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        total_images = 0

        for image_name in tqdm(os.listdir(data_dir),desc = f'extracting test data'):
            image_path = os.path.join(data_dir,image_name)
            img = cv2.imread(image_path)

            if img is None: 
                print(f"Can't read {image_path}")
                continue
                
            processed_img = image_preprocessing(img)
            features = extract_contour(processed_img)
            label = image_name.replace('_test','')
            if features:
                row = [label] + features
                writer.writerow(row)
                total_images += 1

    print(f"COMPLETED! EXTRACT TRAINING FEATURES {total_images} images")
    print(f"Data saved: {output_dir}")

def main():

    download_datasets()
    extract_training_features()
    extract_testing_features()

if __name__ == "__main__":
    main()