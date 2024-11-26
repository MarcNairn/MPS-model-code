from fix_pathing import root_dir

from src.mps import *
import numpy as np
from src.mpo import *
from src.ed import *

L = 6

# construct the random state vector
vec = np.random.randn(2**L) + 1j*np.random.randn(2**L)
vec /= np.linalg.norm(vec)

# construct the MPS
psi = MPS.fromVector(vec)

# construct the Hamiltonian matrix and compute the exact energy
H = HeisenbergHamiltonian(L)
E_exact = vec.conj().T @ H @ vec

# construct the MPO and compute the MPS energy
mpo = MPO.Hamiltonian(L)
E_mps = mpo.expectation(psi)

# check that the energies match
assert np.isclose(E_exact, E_mps), "Energies do not match."

