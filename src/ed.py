import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm


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
    E, V = sp.linalg.eigsh(H, k=1, which='SA')
    
    return E[0], V[:,0]


def entanglementEntropy(psi, site):
    """
    Compute the entanglement entropy of a quantum state psi across bond between site and site+1.
    """

    psi = psi.copy().reshape((2**(site+1), -1))
    _, S, _ = np.linalg.svd(psi, full_matrices=False)

    return -np.sum(S**2 * np.log(S**2))


def HeisenbergTimeEvolution(L, state, dt, tMax):
    """
    Compute the time evolution of the Heisenberg Hamiltonian for a 1D chain of length L. L is even!
    """
    H = HeisenbergHamiltonian(L)
    psi0 = 1.
    for i in state:
        psi0 = np.kron(psi0, np.array([int(i == 0), int(i == 1)]))
    
    psi = psi0
    nSteps = int(tMax / dt)

    U = expm(-1j*dt*H)
    Z = np.kron(np.kron(np.eye(2**(L//2)), np.array([[1,0],[0,-1]])),np.eye(2**(L//2-1)))

    magnetization = []
    entanglement = []
    for i in range(nSteps):
        psi = U @ psi
        magnetization.append( np.real( psi.conj().T @ Z @ psi ) )
        entanglement.append( entanglementEntropy(psi, L//2-1) )
    
    return magnetization, entanglement