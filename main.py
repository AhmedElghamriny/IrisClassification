import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.switch_backend('TkAgg')

# Load dataset
df = pd.read_csv('Iris.csv')

# Split dataset
x_train = df.iloc[:, 1:5].to_numpy()  # Get relevant x features
y_train = df.iloc[:, -1].to_numpy()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

m = len(x_train)
n = 4


def featureScale(trainingExamples):
    largestNum = np.max(trainingExamples)
    return trainingExamples / largestNum


x_train_scaled = np.array([featureScale(x_train[:, i]) for i in range(n)]).T

print(len(x_train_scaled))
print(len(y_train))


def plotDataPre():
    fig1, ax1 = plt.subplots(2, 2)

    ax1[0, 0].set_title('SepalLengthCm')
    ax1[0, 0].scatter(x_train_scaled[:, 0], y_train)
    ax1[0, 0].set_yticks(np.arange(0, 3, step=1))

    ax1[0, 1].set_title('SepalWidthCm')
    ax1[0, 1].scatter(x_train_scaled[:, 1], y_train)
    ax1[0, 1].set_yticks(np.arange(0, 3, step=1))

    ax1[1, 0].set_title('PetalLengthCm')
    ax1[1, 0].scatter(x_train_scaled[:, 2], y_train)
    ax1[1, 0].set_yticks(np.arange(0, 3, step=1))

    ax1[1, 1].set_title('PetalWidthCm')
    ax1[1, 1].scatter(x_train_scaled[:, 3], y_train)
    ax1[1, 1].set_yticks(np.arange(0, 3, step=1))

    plt.show()


# Initializing Params w & b
weights_x = [0, 0, 0, 0]  # Starting weights
bias = 0  # Starting bias


# Function to apply logistic regression
def sigmoidFunction(z):
    return 1 / (1 + np.exp(-z))


def computePrediction(w, b):
    output = []
    for i in range(m):
        output.append(sigmoidFunction(np.dot(w, x_train_scaled[i]) + b))
    return output


def gradientDescentLogisticRegression(w, b, iterations, learning_rate):
    m = len(x_train_scaled)
    cost_history = []

    for iteration in range(iterations):
        z = np.dot(x_train_scaled, w) + b
        predictions = sigmoidFunction(z)

        # Compute gradients
        dw = (1 / m) * np.dot(x_train_scaled.T, (predictions - y_train))
        db = (1 / m) * np.sum(predictions - y_train)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Compute cost
        cost = (-1 / m) * np.sum(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
        cost_history.append(cost)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost}")

    return w, b, cost_history


final_w, final_b, cost_history = gradientDescentLogisticRegression(weights_x, bias, 1000, 0.01)

print(f'W: {final_w}\nB: {final_b}\nCost:\n{cost_history[995: 999]}')

plotDataPre()
