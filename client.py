import numpy as np
from sklearn.model_selection import train_test_split
import socket
import pickle
import time
import cProfile
import random

class LinearRegressionSGD:
    def __init__(self, learning_rate=0.001, n_iterations=100, weight_diff_threshold=0.001):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.weight_diff_threshold = weight_diff_threshold

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Stochastic Gradient Descent
        for it in range(self.n_iterations):
            print("Iteration: ", it)
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
            #     print("Training stopped as weight difference is below threshold.")
            #     break

            print(self.weights, self.bias)

        return self.weights, self.bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def client(host, port):
    # Client socket setup
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Connected to server at {host}:{port}")

    # Receive train data from server
    received_data = client_socket.recv(4096)
    data = pickle.loads(received_data)
    X_train, y_train = data['X_train'], data['y_train']
    print(X_train, y_train)

    # Train the model
    start_time = time.time()
    model = LinearRegressionSGD()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Send back trained parameters
    print("FINAL:: ", model.weights, model.bias)
    print("TIME:: (ms) ",training_time * 1000 )
    data_to_send = pickle.dumps((model.weights, model.bias, training_time))
    client_socket.send(data_to_send)
    

    client_socket.close()

if __name__ == "__main__":
    # SERVER_HOST = '0.0.0.0'  # Replace SERVER_IP with the actual IP address of the server
    SERVER_HOST = "192.168.205.186"
    SERVER_PORT = 9999
    profiler = cProfile.Profile()
    profiler.enable()
    client(SERVER_HOST, SERVER_PORT)
    
    profiler.disable()
    profiler.dump_stats(f"profile_{random.randint(0, 1000)}.out")
