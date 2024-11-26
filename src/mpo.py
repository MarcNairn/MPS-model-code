import numpy as np


class MPO:
    """
    Matrix Product State class for 1D quantum systems of spin-1/2 particles.
    """

    def __init__(self, L, tensors):
        self.L = L  # number of sites
        self.tensors = tensors  # list of tensors. Indices are (left, p_out, p_in, right)

    
    @classmethod
    def Hamiltonian(cls, L):
        """
        Construct the MPO for the 1D Heisenberg Hamiltonian of length L.
        """
        # Define the spin matrices
        identity = np.eye(2)
        Sx = 1/2*np.array([[0, 1], [1, 0]])
        Sy = 1/2*np.array([[0, -1j], [1j, 0]])
        Sz = 1/2*np.array([[1, 0], [0, -1]])

        W = np.zeros((5,2,2,5), dtype=complex)
        W[0, :, :, 0] = identity
        W[0, :, :, 1] = Sx
        W[0, :, :, 2] = Sy
        W[0, :, :, 3] = Sz
        W[1, :, :, 4] = Sx
        W[2, :, :, 4] = Sy
        W[3, :, :, 4] = Sz
        W[4, :, :, 4] = identity

        # Construct the Heisenberg Hamiltonian
        tensors = [W.copy() for _ in range(L)]

        tensors[0] = tensors[0][0, :, :, :].reshape(1, 2, 2, 5)
        tensors[-1] = tensors[-1][:, :, :, 4].reshape(5, 2, 2, 1)

        return cls(L, tensors)
    

    def get_slice(self, psi, i):
        M = np.tensordot(psi.tensors[i], self.tensors[i], axes=([1],[2]))  # (l1, p_in, r1) x (l_mpo, p_out, p_in, r_mpo) -> (l1, r1, l_mpo, p_out, r_mpo)
        M = np.tensordot(M, psi.tensors[i].conj(), axes=([3],[1]))  # (l1, r1, l_mpo, p_out, r_mpo) x (l2, p_out, r2) -> (l1, r1, l_mpo, r_mpo, l2, r2)
        M = M.transpose(0, 2, 4, 1, 3, 5)
        return M


    def expectation(self, psi):
        """
        Compute the (real) expectation value of the (Hermitian) MPS with respect to a state.
        """
        assert psi.L == self.L, "State size does not match MPS size."

        overlap = np.array([1]).reshape((1,1))

        for i in range(self.L):
            M = self.get_slice(psi, i)
            l1, l_mpo, l2, _, _, _ = M.shape

            overlap = overlap @ M.reshape(l1*l_mpo*l2, -1)

        return np.real(overlap[0, 0])