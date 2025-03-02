import skripsi as KnnGenetika
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

def detection(k,features,input,x_train,y_train):
    selected_input= input[:, features]
    x_train_selected = x_train[:, features]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_selected,y_train)
    predict=model.predict(selected_input)
    return predict

@app.route("/get-data", methods=["GET"])
def get_data():
    k=joblib.load("k.pkl")
    features=joblib.load("selected_features.pkl")
    accuracy=joblib.load("best_fitness.pkl")
    return jsonify(
        {
            "k": k,
            "features": features,
            "accuracy": accuracy
        }
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input=np.array([data["input"]])
    k=data['k']
    features=data['features']
    result=main(input, k, features)
    result="Diabetes" if result[0] == 1 else "Sehat"
    return jsonify({"prediksi": result})

@app.route("/genetika", methods=["GET"])
def genetika():
    url = "dataset/dataset_diabetes_1.csv"
    X_train, X_test, y_train, y_test=data(url)
    k,features,accuracy=KnnGenetika.genetic_algorithm(X_train, X_test, y_train, y_test, pop_size=120, generations=100, k_max=31, mutation_rate=0.4)
    return jsonify({
            "k": k,
            "features": features,
            "accuracy": accuracy
        })

def main(sample_input, k, features):
    url = "dataset/dataset_diabetes_1.csv"
    X_train, X_test, y_train, y_test=data(url)
    # k=joblib.load("k.pkl")
    # features=joblib.load("selected_features.pkl")
    scaler=joblib.load("scaler.pkl")
    sample_input_scaled=scaler.transform(sample_input)
    sample_input_scaled=np.clip(sample_input_scaled, 0, 1)
    
    return detection(k,features,sample_input_scaled,X_train,y_train)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

