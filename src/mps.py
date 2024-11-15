import numpy as np
import numpy.linalg as la


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


    @classmethod
    def fromVector(cls, vector):
        """
        Create an MPS from a vector of probability amplitudes.

        Parameters
        ----------
        vector : np.ndarray
            Vector of probability amplitudes for a quantum state. Must be of length 2^L, where L is the number of sites.

        Returns
        -------
        MPS
            MPS representation of the quantum state.
        """

        L = int(np.log2(vector.size))

        tensors = []
        R = vector.copy()
        chi = 1
        for i in range(L-1):
            R = R.reshape((chi*2,2**(L-1-i)))  # group left and first physical leg

            U, S, Vdg = la.svd(R, full_matrices=False)
            lp, chi = U.shape[0], U.shape[1]

            tensors.append(U.reshape((lp//2,2,chi)))
            R = np.diag(S) @ Vdg

        tensors.append(R.reshape((2,2,1)))  # last tensor

        return cls(L, tensors)
    

    @classmethod
    def productState(cls, L, state):
        """
        Create a product state MPS on L sites in a given basis state on each site.

        Parameters
        ----------
        L : int
            Number of sites.
        state : list of ints
            List of basis states for each site. E.g. [0, 1, 0, 1] for a 4-site system.
        """

        tensors = []
        for s in state:
            assert s in [0, 1], "Basis states must be 0 or 1."

            if s == 0:
                tensors.append(np.array([1,0]).reshape((1,2,1)))
            else:
                tensors.append(np.array([0,1]).reshape((1,2,1)))

        return cls(L, tensors)
    

    def toVector(self):
        """
        Returns the probability amplitudes of the MPS as a vector.
        """

        vector = self.tensors[0]
        
        for i in range(1, self.L):
            vector = np.tensordot(vector, self.tensors[i], axes=(2,0))
            l, p1, p2, r = vector.shape
            vector = vector.reshape((l, p1*p2, r))

        vector = vector.flatten()

        return vector