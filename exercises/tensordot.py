from fix_pathing import root_dir

import numpy as np


## matrix-vector multiplication from lecture notes

# Define the tensors
v = np.array([1, 2])  # vector
M = np.array([[1, 2], [3, 4]])  # matrix

# Contract the matrix with the vector
result = np.tensordot(M, v, axes=([1], [0]))

# check that the result matches
assert np.allclose(result, M @ v), "matrix-vector multiplication failed!" 


## matrix-matrix multiplication from lecture notes

# Define the tensors
A = np.array([[1, 2], [3, 4]])  # matrix
B = np.array([[5, 6], [7, 8]])  # matrix

# Contract the matrices
result1 = np.tensordot(A, B, axes=([1], [0]))

# Contract the matrices in the opposite order
result2 = np.tensordot(B, A, axes=([0], [1]))
result2 = np.transpose(result2, (1, 0))  # transpose the result to match the order of the indices

# check that the result matches
assert np.allclose(result1, A @ B), "matrix-matrix method 1 failed!"  
assert np.allclose(result2, A @ B), "matrix-matrix method 2 failed!" 


#--------------------
## Excercises ######
#--------------------

A = np.random.randn(10,10) + 1j*np.random.randn(10,10)
B = np.random.randn(10,10) + 1j*np.random.randn(10,10)


# Exercise 1
# Compute the trace of the product of A and B
result = np.tensordot(A, B, axes=([0,1], [1,0]))

# check that the result matches
assert np.allclose(result, np.trace(A @ B)), "trace of product failed!"


# Exercise 2
# Compute the trace of the product of A and B using the opposite order of matrices
result = np.tensordot(B, A, axes=([1,0], [0,1]))

# check that the result matches
assert np.allclose(result, np.trace(A @ B)), "trace of product 2 failed!"


# Exercise 3
# Compute the product A^dag B 
result = np.tensordot(A.conj().T, B, axes=([1],[0]))

# check that the result matches
assert np.allclose(result, A.conj().T @ B), "adjoint product failed!"


# Exercise 4    
# Compute the product A^\dag B with A^* in second slot
result = np.tensordot(B, A.conj(), axes=([0],[0]))
result = np.transpose(result, (1, 0))  # transpose the result to match the order of the indices

# check that the result matches
assert np.allclose(result, A.conj().T @ B), "adjoint product 2 failed!"


print("All exercises passed!")
