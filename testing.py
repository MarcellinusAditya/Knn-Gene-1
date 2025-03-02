import skripsi as KnnGenetika
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

def import_data(url):
    
    df = pd.read_csv(url)
    X = df.iloc[:, :-1].values  # 8 fitur
    y = df.iloc[:, -1].values   # Label
    return X,y

def data(url):
    X,y=import_data(url)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    X_test = np.clip(X_test, 0, 1)
    return X_train, X_test, y_train, y_test

def detection(k,features,x_train,y_train, x_test, y_test):
    # selected_input= input[:, features]
    x_train_selected = x_train[:, features]
    x_test_selected= x_test[:, features]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_selected,y_train)
    predict=model.predict(x_test_selected)
 
    print(recall_score(predict, y_test))
    return predict

def main():
    # sample_input= [9, 145, 88, 34, 165, 30.3, 0.711, 53]
    # input=np.array([sample_input])
    # input=np.clip(input, 0, 1)
    k= 8
    features=[1, 2, 3, 5, 6, 7]
    url = "dataset/dataset_diabetes_1.csv"
    X_train, X_test, y_train, y_test=data(url)
    scaler=joblib.load("scaler.pkl")
    # sample_input_scaled=scaler.transform(sample_input)
    # sample_input_scaled=np.clip(sample_input_scaled, 0, 1)

    return detection(k,features,X_train,y_train,X_test,y_test)

if __name__ == "__main__":
    main()