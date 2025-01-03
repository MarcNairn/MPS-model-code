"""
 Here we use the MPS class to perform basic operations on tensors:
 We go through the manipulation of tensor indeces and tensor contraction.

"""
# %%
import numpy as np


## Tensor and vector

# Define the tensors
v = np.array([1, 2])  # vector
M = np.array([[1, 2], [3, 4]])  # matrix

# Contract the matrix with the vector
result = np.tensordot(M, v, axes=([1], [0]))

# check that the result matches
assert np.allclose(result, M @ v), "matrix-vector multiplication failed!" 
# %%

# Tensor and tensor

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
# %%


# Rank 3 tensor and rank 3 tensor

# Define the tensors
A = np.random.randn(4, 2, 4)  # (left, physical, right)
B = np.random.randn(4, 2, 4)  # (left, physical, right)

# Contract the tensors along the physical indices
theta = np.tensordot(A, B, axes=([1], [1]))  # (l1,p,r1) * (l2,p,r2) -> (l1,r1,l2,r2)
theta = np.transpose(theta, (0, 2, 1, 3))  # (l1,r1,l2,r2) -> (l1,l2,r1,r2)