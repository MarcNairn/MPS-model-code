import numpy as np
import scipy.sparse as sp


def HeisenbergHamiltonian(L):
    """
    Construct the Heisenberg Hamiltonian for a 1D chain of length L.
    """
    # Define the Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Construct the Heisenberg Hamiltonian
    H = np.zeros((2**L, 2**L))
    for i in range(L-1):
        H += np.kron(np.kron(np.kron(np.eye(2**i), sigma_x), sigma_x), np.eye(2**(L-i-2)))
        H += np.real(np.kron(np.kron(np.kron(np.eye(2**i), sigma_y), sigma_y), np.eye(2**(L-i-2))))
        H += np.kron(np.kron(np.kron(np.eye(2**i), sigma_z), sigma_z), np.eye(2**(L-i-2)))

    return H


def HeisenbergGroundState(L):
    """
    Compute the ground state of the Heisenberg Hamiltonian for a 1D chain of length L.
    """
    H = HeisenbergHamiltonian(L)
    E, V = sp.linalg.eigsh(H, k=1, which='SA')
    
    return E, V