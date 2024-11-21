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

    def __init__(self, L, tensors, centre):
        self.L = L  # number of sites
        self.tensors = tensors  # list of tensors. Indices are (left, physical, right)
        self.centre = centre  # position of the orthogonality centre
        

    def copy(self):
        return MPS(self.L, [tensor.copy() for tensor in self.tensors], self.centre)


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

        return cls(L, tensors, L-1)
    

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

        return cls(L, tensors, 0)
    

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
    

    def move_centre_left(self):
        """
        Move the orthogonality centre to the left. This does not truncate the bond dimension.
        """
        if self.centre == 0:
            return

        tensor_left = self.tensors[self.centre-1]
        tensor_right = self.tensors[self.centre]
        chi = tensor_left.shape[2]

        theta = np.tensordot(tensor_left, tensor_right, axes=(2,0))
        l, p1, p2, r = theta.shape
        theta = theta.reshape((l*p1, p2*r))

        U, S, Vdg = la.svd(theta, full_matrices=False)
        U = U[:, :chi]
        S = S[:chi]
        Vdg = Vdg[:chi, :]

        tensor_left = U @ np.diag(S)
        tensor_right = Vdg

        self.tensors[self.centre-1] = tensor_left.reshape((l, p1, chi))
        self.tensors[self.centre] = tensor_right.reshape((chi, p2, r))
        self.centre -= 1


    def move_centre_right(self):
        """
        Move the orthogonality centre to the right. This does not truncate the bond dimension.
        """
        if self.centre == self.L-1:
            return

        tensor_left = self.tensors[self.centre]
        tensor_right = self.tensors[self.centre+1]
        chi = tensor_left.shape[2]

        theta = np.tensordot(tensor_left, tensor_right, axes=(2,0))
        l, p1, p2, r = theta.shape
        theta = theta.reshape((l*p1, p2*r))

        U, S, Vdg = la.svd(theta, full_matrices=False)
        U = U[:, :chi]
        S = S[:chi]
        Vdg = Vdg[:chi, :]

        tensor_left = U
        tensor_right = np.diag(S) @ Vdg

        self.tensors[self.centre] = tensor_left.reshape((l, p1, chi))
        self.tensors[self.centre+1] = tensor_right.reshape((chi, p2, r))
        self.centre += 1


    def move_centre_to(self, i):
        """
        Move the orthogonality centre to site i.
        """

        while self.centre > i:
            self.move_centre_left()

        while self.centre < i:
            self.move_centre_right()


    def expectation(self, O, site):
        """
        Compute the expectation value of an operator at a given site.

        Parameters
        ----------
        O : np.ndarray (shape=(2,2))
            Operator acting on a single site.
        site : int
            Site at which to compute the expectation value.
        """

        self.move_centre_to(site)

        tensor = self.tensors[self.centre]

        expectation = np.tensordot(tensor, O, axes=([1],[1]))  # (l, q, r)*(p, q) -> l, r, p
        expectation = np.tensordot(expectation, np.conj(tensor), axes=([0,2,1],[0,1,2]))  # (l, r, p) (l, p, r) -> ()

        return np.real(expectation)


    def entropy(self, site):
        """
        Compute the bipartite entanglement entropy of the bond between site and site+1.
        """
        # Move the centre to the site of interest
        self.move_centre_to(site)

        C = self.tensors[site]
        B = self.tensors[site+1]

        # Contract the tensors
        theta = np.tensordot(C, B, axes=([2], [0]))  # (l, p, j) (j, q, r) -> (l, p, q, r)
        l, p, q, r = theta.shape
        theta = theta.reshape((l*p, q*r))

        _, S, _ = la.svd(theta, full_matrices=False)

        return -np.sum(S**2 * np.log(S**2))