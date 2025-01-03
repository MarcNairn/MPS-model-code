""" 

MAIN FUNCTIONS AND OTHER USEFUL RESOURCES

This file contains the main functions used in the exercises throughout the course

"""

import numpy as np
import scipy.sparse as sp

"""
EXACT DIAGONALIZATION
"""
def HeisenbergHamiltonian(L):
    """
    Construct the Heisenberg Hamiltonian for a 1D chain of length L.
    """
    # Define the spin operators
    s_x = 1/2*np.array([[0, 1], [1, 0]])
    s_y = 1/2*np.array([[0, -1j], [1j, 0]])
    s_z = 1/2*np.array([[1, 0], [0, -1]])

    # Construct the Heisenberg Hamiltonian
    H = np.zeros((2**L, 2**L))
    for i in range(L-1):
        H += np.kron(np.kron(np.kron(np.eye(2**i), s_x), s_x), np.eye(2**(L-i-2)))
        H += np.real(np.kron(np.kron(np.kron(np.eye(2**i), s_y), s_y), np.eye(2**(L-i-2))))
        H += np.kron(np.kron(np.kron(np.eye(2**i), s_z), s_z), np.eye(2**(L-i-2)))

    return H


def HeisenbergGroundState(L):
    """
    Compute the ground state of the Heisenberg Hamiltonian for a 1D chain of length L.
    """
    H = HeisenbergHamiltonian(L)
    E, V = sp.linalg.eigsh(H, k=1, which='SA') # builtin exact diagonalization
    
    return E[0], V[:,0]


""" 

MPS CLASS 

(Note Python classes are equivalent to Julia structs)

This class contains two attributes: "L", the number of sites, and "tensors" which is a list of rank-3 tensors. 
    These tensors are just numpy arrays. The "copy" method is useful in cases we need to make many MPS copies
"""

class MPS:
    """
    Matrix Product State class for 1D quantum systems of spin-1/2 particles.

    Attributes
    ----------
    L : Int 
        number of sites
    tensors : list of np.Array[ndim=3]
        list of tensors. Indices are (left, physical, right)
    """

    def __init__(self, L, tensors):
        self.L = L  # number of sites
        self.tensors = tensors  # list of tensors. Indices are (left, physical, right)
        

    def copy(self):
        return MPS(self.L, [tensor.copy() for tensor in self.tensors])
    


