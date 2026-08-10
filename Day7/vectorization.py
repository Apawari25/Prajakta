import numpy as np

# Sensor values
sensor_values = np.array([10, 20, 30, 40, 50])

print("Original values:")
print(sensor_values)

# Vectorization
normalized = sensor_values / 100

print("Normalized values:")
print(normalized)


# Broadcasting

sensor_values = np.array([10, 20, 30, 40, 50])

added_values = sensor_values + 5
multiplied_values = sensor_values * 2

print("Original:", sensor_values)
print("Add 5:", added_values)
print("Multiply by 2:", multiplied_values)

# Boolean Masking

sensor_values = np.array([10, 20, 30, 40, 50])

mask = sensor_values > 25

print("Boolean mask:")
print(mask)

print("Values greater than 25:")
print(sensor_values[mask])

# Statistics

sensor_values = np.array([10, 20, 30, 40, 50])

print("Mean:", np.mean(sensor_values))
print("Standard Deviation:", np.std(sensor_values))
print("Minimum:", np.min(sensor_values))
print("Maximum:", np.max(sensor_values))

# Axis statistics

data = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Data:")
print(data)

print("Mean of all values:", np.mean(data))

print("Mean by column:")
print(np.mean(data, axis=0))

print("Mean by row:")
print(np.mean(data, axis=1))

print("Maximum by column:")
print(np.max(data, axis=0))

print("Minimum by row:")
print(np.min(data, axis=1))

# Performance comparison

import time

large_data = np.arange(1_000_000)

# Python loop
start = time.time()

loop_result = []
for value in large_data:
    loop_result.append(value * 2)

loop_time = time.time() - start


# NumPy vectorization
start = time.time()

numpy_result = large_data * 2

numpy_time = time.time() - start


print("Loop time:", loop_time)
print("NumPy time:", numpy_time)

print("Results are same:", np.array_equal(
    np.array(loop_result),
    numpy_result
))