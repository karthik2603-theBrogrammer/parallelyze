import numpy as np
from sklearn.model_selection import train_test_split
import socket
import pickle
import time
import cProfile
import random
from mpi4py import MPI
import threading

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
    def __init__(self, learning_rate=0.001, n_iterations=100, weight_diff_threshold=0.001):
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
            #     print(f"Training stopped as weight difference is below threshold.", {it})
            #     break

            # print(self.weights, self.bias)
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


X = np.array([[9.65], [8.87], [8.], [8.67], [8.21], [9.34], [8.2], [7.9], [8.], [8.6], [8.4],
                [9.], [9.1], [8.], [8.2], [8.3], [8.7], [
                    8.], [8.8], [8.5], [7.9], [8.4],
                [9.5], [9.7], [9.8], [9.6], [8.8], [7.5], [
                    7.2], [7.3], [8.1], [8.3], [9.4],
                [9.6], [9.8], [9.2], [8.4], [7.8], [
    7.5], [7.7], [8.], [8.2], [8.5], [9.1],
    [9.4], [9.1], [9.3], [9.7], [8.85], [
        8.4], [8.3], [7.9], [8.], [8.1], [8.],
    [7.7], [7.4], [7.6], [6.8], [8.3], [8.1], [
        8.2], [8.2], [8.5], [8.7], [8.92],
    [9.02], [8.64], [9.22], [9.16], [9.64], [
        9.76], [9.45], [9.04], [8.9], [8.56], [8.72],
    [8.22], [7.54], [7.36], [8.02], [9.5], [9.22], [
        9.36], [9.45], [8.66], [8.42], [8.28],
    [8.14], [8.76], [7.92], [7.66], [8.03], [
        7.88], [7.66], [7.84], [8.], [8.96], [9.24],
    [8.88], [8.46], [8.12], [8.25], [8.47], [9.05], [
        8.78], [9.18], [9.46], [9.38], [8.64],
    [8.48], [8.68], [8.34], [8.56], [8.45], [9.04], [
        8.62], [7.46], [7.28], [8.84], [9.56],
    [9.48], [8.36], [8.22], [8.47], [8.66], [
        9.32], [8.71], [9.1], [9.35], [9.76], [8.65],
    [8.56], [8.78], [9.28], [8.77], [8.45], [8.16], [
        9.08], [9.12], [9.15], [9.36], [9.44],
    [9.92], [8.96], [8.64], [8.48], [9.11], [9.8], [
        8.26], [9.43], [9.28], [9.06], [8.75],
    [8.89], [8.69], [8.34], [8.26], [8.14], [
        7.9], [7.86], [7.46], [8.5], [8.56], [9.01],
    [8.97], [8.33], [8.27], [7.8], [7.98], [8.04], [
        9.07], [9.13], [9.23], [8.97], [8.87],
    [9.16], [9.04], [8.12], [8.27], [8.16], [
        8.42], [7.88], [8.8], [8.32], [9.11], [8.68],
    [9.44], [9.36], [9.08], [9.16], [8.98], [8.94], [
        9.53], [8.76], [8.52], [8.26], [8.33],
    [8.43], [8.69], [8.54], [8.46], [9.91], [9.87], [8.54], [
        7.65], [7.89], [8.02], [8.16], [8.12], [9.06], [9.14],
    [9.66], [9.78], [9.42], [9.36], [9.26], [9.13], [
        8.97], [8.42], [8.75], [8.56], [8.79],
    [8.45], [8.23], [8.03], [8.45], [8.53], [8.67], [
        9.01], [8.65], [8.33], [8.27], [8.07],
    [9.31], [9.23], [9.17], [9.19], [8.37], [7.89], [
        7.68], [8.15], [8.76], [9.04], [8.56],
    [9.02], [8.73], [8.48], [8.87], [8.83], [
        8.57], [9.], [8.54], [9.68], [9.12], [8.37],
    [8.56], [8.64], [8.76], [9.34], [9.13], [8.09], [
        8.36], [8.79], [8.76], [8.68], [8.45],
    [8.17], [9.14], [8.34], [8.22], [7.86], [7.64], [
        8.01], [7.95], [8.96], [9.45], [8.62],
    [8.49], [8.73], [8.64], [9.11], [8.79], [8.9], [
        9.66], [9.26], [9.19], [9.08], [9.02],
    [9.], [7.65], [7.87], [7.97], [8.18], [8.32], [
        8.57], [8.67], [9.11], [9.24], [8.65],
    [8.], [8.76], [8.45], [8.55], [8.43], [
        8.8], [9.1], [9.], [8.53], [8.6], [8.74],
    [9.18], [9.], [8.04], [8.13], [8.07], [7.86], [
        8.01], [8.8], [8.69], [8.5], [8.44],
    [8.27], [8.18], [8.33], [9.14], [8.02], [7.86], [
        8.77], [7.89], [8.66], [8.12], [8.21],
    [8.54], [8.65], [9.11], [8.79], [9.47], [8.74], [
        8.66], [8.46], [8.76], [8.24], [8.13],
    [7.34], [7.43], [7.64], [7.34], [7.25], [8.04], [
        8.27], [8.67], [8.06], [8.17], [7.67],
    [8.12], [8.77], [7.89], [7.64], [8.44], [
        8.64], [9.54], [9.23], [8.36], [8.9], [9.17],
    [8.34], [7.46], [7.88], [8.03], [8.24], [9.22], [
        9.62], [8.54], [7.65], [7.66], [7.43],
    [7.56], [7.65], [8.43], [8.84], [8.67], [
        9.15], [8.26], [9.74], [9.82], [7.96], [8.1],
    [7.8], [8.44], [8.24], [8.65], [9.12], [8.76], [
        9.23], [9.04], [9.11], [9.45], [8.78],
    [9.66]])

y = np.array([0.92, 0.76, 0.72, 0.8, 0.65, 0.9, 0.75, 0.68, 0.5, 0.45, 0.52,
                0.84, 0.78, 0.62, 0.61, 0.54, 0.66, 0.65, 0.63, 0.62, 0.64, 0.7,
                0.94, 0.95, 0.97, 0.94, 0.76, 0.44, 0.46, 0.54, 0.65, 0.74, 0.91,
                0.9, 0.94, 0.88, 0.64, 0.58, 0.52, 0.48, 0.46, 0.49, 0.53, 0.87,
                0.91, 0.88, 0.86, 0.89, 0.82, 0.78, 0.76, 0.56, 0.78, 0.72, 0.7,
                0.64, 0.64, 0.46, 0.36, 0.42, 0.48, 0.47, 0.54, 0.56, 0.52, 0.55,
                0.61, 0.57, 0.68, 0.78, 0.94, 0.96, 0.93, 0.84, 0.74, 0.72, 0.74,
                0.64, 0.44, 0.46, 0.5, 0.96, 0.92, 0.92, 0.94, 0.76, 0.72, 0.66,
                0.64, 0.74, 0.64, 0.38, 0.34, 0.44, 0.36, 0.42, 0.48, 0.86, 0.9,
                0.79, 0.71, 0.64, 0.62, 0.57, 0.74, 0.69, 0.87, 0.91, 0.93, 0.68,
                0.61, 0.69, 0.62, 0.72, 0.59, 0.66, 0.56, 0.45, 0.47, 0.71, 0.94,
                0.94, 0.57, 0.61, 0.57, 0.64, 0.85, 0.78, 0.84, 0.92, 0.96, 0.77,
                0.71, 0.79, 0.89, 0.82, 0.76, 0.71, 0.8, 0.78, 0.84, 0.9, 0.92,
                0.97, 0.8, 0.81, 0.75, 0.83, 0.96, 0.79, 0.93, 0.94, 0.86, 0.79,
                0.8, 0.77, 0.7, 0.65, 0.61, 0.52, 0.57, 0.53, 0.67, 0.68, 0.81,
                0.78, 0.65, 0.64, 0.64, 0.65, 0.68, 0.89, 0.86, 0.89, 0.87, 0.85,
                0.9, 0.82, 0.72, 0.73, 0.71, 0.71, 0.68, 0.75, 0.72, 0.89, 0.84,
                0.93, 0.93, 0.88, 0.9, 0.87, 0.86, 0.94, 0.77, 0.78, 0.73, 0.73,
                0.7, 0.72, 0.73, 0.72, 0.97, 0.97, 0.69, 0.57, 0.63, 0.66, 0.64,
                0.68, 0.79, 0.82, 0.95, 0.96, 0.94, 0.93, 0.91, 0.85, 0.84, 0.74,
                0.76, 0.75, 0.76, 0.71, 0.67, 0.61, 0.63, 0.64, 0.71, 0.82, 0.73,
                0.74, 0.69, 0.64, 0.91, 0.88, 0.85, 0.86, 0.7, 0.59, 0.6, 0.65,
                0.7, 0.76, 0.63, 0.81, 0.72, 0.71, 0.8, 0.77, 0.74, 0.7, 0.71,
                0.93, 0.85, 0.79, 0.76, 0.78, 0.77, 0.9, 0.87, 0.71, 0.7, 0.7,
                0.75, 0.71, 0.72, 0.73, 0.83, 0.77, 0.72, 0.54, 0.49, 0.52, 0.58,
                0.78, 0.89, 0.7, 0.66, 0.67, 0.68, 0.8, 0.81, 0.8, 0.94, 0.93,
                0.92, 0.89, 0.82, 0.79, 0.58, 0.56, 0.56, 0.64, 0.61, 0.68, 0.76,
                0.86, 0.9, 0.71, 0.62, 0.66, 0.65, 0.73, 0.62, 0.74, 0.79, 0.8,
                0.69, 0.7, 0.76, 0.84, 0.78, 0.67, 0.66, 0.65, 0.54, 0.58, 0.79,
                0.8, 0.75, 0.73, 0.72, 0.62, 0.67, 0.81, 0.63, 0.69, 0.8, 0.43,
                0.8, 0.73, 0.75, 0.71, 0.73, 0.83, 0.72, 0.94, 0.81, 0.81, 0.75,
                0.79, 0.58, 0.59, 0.47, 0.49, 0.47, 0.42, 0.57, 0.62, 0.74, 0.73,
                0.64, 0.63, 0.59, 0.73, 0.79, 0.68, 0.7, 0.81, 0.85, 0.93, 0.91,
                0.69, 0.77, 0.86, 0.74, 0.57, 0.51, 0.67, 0.72, 0.89, 0.95, 0.79,
                0.39, 0.38, 0.34, 0.47, 0.56, 0.71, 0.78, 0.73, 0.82, 0.62, 0.96,
                0.96, 0.46, 0.53, 0.49, 0.76, 0.64, 0.71, 0.84, 0.77, 0.89, 0.82,
                0.84, 0.91, 0.67, 0.95])

X_train, X_test, y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



"""Divided the train batch into n datasets to train separately."""
X_train_batches = divide_array_into_chunks(X_train, size - 1)
y_train_batches = divide_array_into_chunks(y_train, size - 1)


# print("MPI Process starting with total ranks = ", size)
# print("Current Is: ", rank)


if rank == 0:
    print("MPI Process starting with total ranks = ", size)
    print(rank, "There are ", len(X_train_batches), " Batches")
    for i in range(len(X_train_batches)):
        print("Batch Size", len(X_train_batches[i]))
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

    print("Enter your CGPA to know the chances of getting admission: ")
    while(True):
        number = float(input("The Outcome: "))
        if(number == -1):
            break
        else:
            predicted = model.predict([number])
            print(predicted)
    print()

else:
    X_train_b, y_train_b = comm.recv(source=0)
    # print(f"Rank {rank} received chunk: {X_train_b}, {y_train_b}")
    model = LinearRegressionSGD()
    weights, biases, time = model.fit(X_train_b, y_train_b)
    print(rank, weights, biases, time)
    comm.send((weights, biases, time), dest = 0)