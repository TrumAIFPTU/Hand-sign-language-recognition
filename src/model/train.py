import pandas as pd
import numpy as np
import os
import joblib
from src.utils.paths import PREPROCESS_DIR
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


MODEL_DIR = 'model_saved/random_forest_clf.pkl'

def read_datasets():
    df_train = pd.read_csv(PREPROCESS_DIR/'train_features.csv')
    df_test = pd.read_csv(PREPROCESS_DIR/'test_features.csv')

    return df_train,df_test

def train_model():
    df_train,df_test = read_datasets()

    X_train,y_train = df_train.drop('Label',axis=1), df_train['Label']
    X_test,y_test = df_test.drop('Label',axis=1), df_test['Label']

    model = RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred=y_pred,y_true=y_test)
    
    print(f"ACCURACY_SCORE: {accuracy:.2f}")

    if accuracy > 0.7: 
        joblib.dump(model,MODEL_DIR)
        print(f'Requirements accepted => MODEL SAVED: {MODEL_DIR}')
    else:
        print("N.G.U")


