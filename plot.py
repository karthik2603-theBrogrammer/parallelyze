import json
import matplotlib.pyplot as plt

# Read data from results.json
with open('results.json', 'r') as file:
    data = json.load(file)

# Extract np, time, and mse values
np_values = list(map(int, data.keys()))
time_values = [data[str(np)]['time'] for np in np_values]
mse_values = [data[str(np)]['mse'] for np in np_values]

# Create Time vs np chart
plt.figure(figsize=(10, 5))
plt.plot(np_values, time_values, marker='o', color='b', label='Time')
plt.title('Time vs Number of Processes')
plt.xlabel('Number of Processes (np)')
plt.ylabel('Time (ms)')
plt.xticks(np_values)
plt.grid(True)
plt.legend()
plt.show()

# Create MSE vs np chart
plt.figure(figsize=(10, 5))
plt.plot(np_values, mse_values, marker='o', color='r', label='MSE')
plt.title('Mean Squared Error (MSE) vs Number of Processes')
plt.xlabel('Number of Processes (np)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(np_values)
plt.grid(True)
plt.legend()
plt.show()
