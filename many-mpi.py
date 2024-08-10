import numpy as np
from sklearn.model_selection import train_test_split
import socket
import pickle
import time
import cProfile
import random
from mpi4py import MPI
import threading
import pandas as pd
from sklearn.preprocessing import StandardScaler

state = {
    "total_time": 0,
    "weights": np.array([0.0]),
    "bias": np.array([0.0]),
    "total_clients": 0,

    "total_weights": np.array([0.0]),
    "total_bias": np.array([0.0])

}
lock = threading.Lock()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class LinearRegressionSGD:
    def __init__(self, learning_rate=0.0001, n_iterations=100, weight_diff_threshold=0.001):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.weight_diff_threshold = weight_diff_threshold

    def fit(self, X, y):
        start = time.time()
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Stochastic Gradient Descent
        for it in range(self.n_iterations):
            # print("Iteration: ", it)
            prev_weights = self.weights.copy()  # Copy previous weights

            for i in range(n_samples):
                # Predictions for individual sample
                y_predicted = np.dot(X[i], self.weights) + self.bias

                # Compute gradients for individual sample
                dw = 2 * X[i] * (y_predicted - y[i])
                db = 2 * (y_predicted - y[i])

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Check if the difference in weights is less than threshold
            # weight_diff = np.linalg.norm(prev_weights - self.weights)
            # if weight_diff < self.weight_diff_threshold:
            #     print(f"Training stopped as weight difference is below threshold at iteration {it}.")
            #     break

        end = time.time()
        total_time = end - start

        return self.weights, self.bias, total_time

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def divide_array_into_chunks(array, n):
    """
    Divide a NumPy array into n approximately equal chunks.
    
    Parameters:
        array (numpy.ndarray): The input array to be divided.
        n (int): The number of chunks to divide the array into.
        
    Returns:
        list of numpy.ndarray: A list containing the divided chunks of the array.
    """
    chunk_size = len(array) // n
    remainder = len(array) % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(array[start:end])
        start = end
    return chunks


def load_data_from_csv(filename):
    """
    Load data from a CSV file, normalize the features (X), and increase precision to float64.

    Parameters:
        filename (str): The name of the CSV file.

    Returns:
        numpy.ndarray: The normalized feature matrix X with float64 precision.
        numpy.ndarray: The target vector y with float64 precision.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Selecting only desired columns for X and y
    selected_columns = ['GRE Score', 'TOEFL Score', 'University Rating',
                        'SOP', 'LOR', 'CGPA', 'Research']
    X = df[selected_columns[:-1]].values.astype(np.float64)  # Features as numpy array
    y = df[selected_columns[-1]].values.astype(np.float64)   # Target as numpy array

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y

# filename = sys.argv[1]
X, y = load_data_from_csv("Admission.csv")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



"""Divided the train batch into n datasets to train separately."""
X_train_batches = divide_array_into_chunks(X_train, size - 1)
y_train_batches = divide_array_into_chunks(y_train, size - 1)

# print(X_train_batches[0])
# input()
# print("MPI Process starting with total ranks = ", size)
# print("Current Is: ", rank)


if rank == 0:
    # print("MPI Process starting with total ranks = ", size)
    # print(rank, "There are ", len(X_train_batches), " Batches")
    for i in range(len(X_train_batches)):
        # print("Batch Size", len(X_train_batches[i]))
        comm.send((X_train_batches[i], y_train_batches[i]), dest=i + 1)

    received_weights = []
    received_biases = []
    received_times = []

    for i in range(1, size):
        weights, biases, time = comm.recv(source=i)
        received_weights.append(weights)
        received_biases.append(biases)
        received_times.append(time)

    # Average the weights
    averaged_weights = np.mean(received_weights, axis=0)
    # Average the biases
    averaged_biases = np.mean(received_biases, axis=0)
    # Calculate total time
    avg_time = np.mean(received_times)

    print("Averaged Weights:", averaged_weights)
    print("Averaged Biases:", averaged_biases)
    print("Average Time (ms) :", avg_time * 1000)

    y_pred = np.dot(X_test, averaged_weights) + averaged_biases
    mse = np.mean((y_test - y_pred) ** 2)
    print("Mean Squared Error (MSE):", mse)

    model = LinearRegressionSGD()
    model.weights = averaged_weights
    model.bias = averaged_biases


    # while(True):
    #     number = float(input("The Outcome: "))
    #     if(number == -1):
    #         break
    #     else:
    #         predicted = model.predict([number])
    #         print(predicted)
    # print()

else:
    X_train_b, y_train_b = comm.recv(source=0)
    # print(f"Rank {rank} received chunk: {X_train_b}, {y_train_b}")
    model = LinearRegressionSGD()
    weights, biases, time = model.fit(X_train_b, y_train_b)
    print(rank, weights, biases, time)
    comm.send((weights, biases, time), dest = 0)