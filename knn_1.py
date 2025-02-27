import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load dataset
data = pd.read_csv('dataset/dataset_diabetes_1_test.csv')

# Prepare data
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = MinMaxScaler()

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test = np.clip(x_test, 0, 1)

# List to store values of k and their corresponding accuracy
k_values = []
accuracies = []

k = 1
while k < 30:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("k:", k, " Accuracy:", acc, "%")
    
    # Append k and accuracy to lists
    k_values.append(k)
    accuracies.append(acc)
    
    k += 2


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies)
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.grid(True)
plt.show()
