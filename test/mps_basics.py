from fix_pathing import root_dir

import numpy as np
from src.mps import MPS
from src.ed import HeisenbergGroundState

L = 6


## Test 1 - random vector -> MPS -> vector
# Generate a random vector
v = np.random.randn(2**L) + 1j*np.random.randn(2**L)
v /= np.linalg.norm(v)

# Create an MPS from the vector
psi = MPS.fromVector(v)

# Create a vector from the MPS
v2 = psi.toVector()

# Check that the vectors match
assert np.allclose(v, v2), "Test 1 failed!"


## Test 2 - Ground State -> MPS -> vector
# Compute the ground state of the Heisenberg Hamiltonian
_, V = HeisenbergGroundState(L)

# Create an MPS from the ground state
psi = MPS.fromVector(V)

# Create a vector from the MPS
V2 = psi.toVector()

# Check that the vectors match
assert np.allclose(V, V2), "Test 2 failed!"


## Test 3 - Product state MPS -> vector
# Create a product state MPS
psi = MPS.productState(4, [0,0,0,0])

# Create a vector from the MPS
v = psi.toVector()

answer = np.zeros(2**4)
answer[0] = 1

# Check that the vectors match
assert np.allclose(v, answer), "Test 3 failed!"


## Test 4 - Product state MPS -> vector
# Create a product state MPS
psi = MPS.productState(4, [0,1,0,1])

# Create a vector from the MPS
v = psi.toVector()

answer = np.zeros(2**4)
answer[5] = 1

# Check that the vectors match
assert np.allclose(v, answer), "Test 3 failed!"


print("All tests passed!")