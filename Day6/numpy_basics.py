import numpy as np
numbers = np.array([10, 20, 30, 40, 50])
print("Array:", numbers)
print("Shape:", numbers.shape)
print("Dimensions:", numbers.ndim)
print("Data type:", numbers.dtype)


# arange
arr = np.arange(1, 11)
print("Arange:", arr)


# zeros
zeros_array = np.zeros((2, 3))
print("Zeros:")
print(zeros_array)


# ones
ones_array = np.ones((2, 3))
print("Ones:")
print(ones_array)


# 2D array
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("2D Array:")
print(matrix)

print("2D Shape:", matrix.shape)
print("2D Dimensions:", matrix.ndim)

# Indexing
print("First element:", matrix[0, 0])
print("Second row, third column:", matrix[1, 2])

# Slicing
print("First two rows and first two columns:")
print(matrix[0:2, 0:2])

# Reshape
numbers = np.arange(1, 13)

print("Original array:")
print(numbers)

matrix_reshape = numbers.reshape(3, 4)

print("Reshaped array:")
print(matrix_reshape)

print("Reshaped shape:", matrix_reshape.shape)


# 3D Array
array_3d = np.arange(24).reshape(2, 3, 4)

print("3D Array:")
print(array_3d)

print("3D Shape:", array_3d.shape)
print("3D Dimensions:", array_3d.ndim)

# RGB Image Representation
image = np.zeros((100, 200, 3), dtype=np.uint8)

print("RGB Image Shape:", image.shape)
print("RGB Image Dimensions:", image.ndim)
print("RGB Image Data Type:", image.dtype)