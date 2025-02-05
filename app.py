import skripsi as KnnGenetika
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

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
    joblib.dump(scaler,"scaler.pkl")
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    X_test = np.clip(X_test, 0, 1)
    return X_train, X_test, y_train, y_test

def detection(k,features,input,x_train,y_train):
    selected_input= input[:, features]
    x_train_selected = x_train[:, features]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_selected,y_train)
    predict=model.predict(selected_input)
    return predict

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input=np.array([data["features"]])
    input=np.clip(input, 0, 1)
    result=main(input)
    result="Diabetes" if result[0] == 1 else "Sehat"
    return jsonify({"prediksi": result})

def main(sample_input):
    url = "dataset/dataset_diabetes_1_test.csv"
    X_train, X_test, y_train, y_test=data(url)
    # k,features=KnnGenetika.genetic_algorithm(X_train, X_test, y_train, y_test, pop_size=90, generations=10, k_max=31, mutation_rate=0.4)
    k=joblib.load("k.pkl")
    features=joblib.load("selected_features.pkl")
    # sample_input = np.array([[2, 110, 74, 29, 125, 32.4, 0.698, 27]])
    scaler=joblib.load("scaler.pkl")
    sample_input_scaled=scaler.transform(sample_input)
    sample_input_scaled=np.clip(sample_input_scaled, 0, 1)

    return detection(k,features,sample_input_scaled,X_train,y_train)

if __name__ == "__main__":
    app.run(debug=True)

