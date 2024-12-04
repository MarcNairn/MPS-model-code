import numpy as np

def overlap(psi1, psi2):
    """
    Compute the overlap between psi1 and psi2. Specifically, <psi1 | psi2>.
    """

    assert psi1.L == psi2.L, "MPS must have the same number of sites."

    overlap = np.array([1.0], dtype=complex)
    overlap = overlap.reshape((1,1,1,1))  # (l1, l2, r1, r2)

    for i in range(psi1.L):
        tensor1 = psi1.tensors[i]
        tensor2 = psi2.tensors[i]

        overlap_slice = np.tensordot(tensor1.conj(), tensor2, axes=([1],[1]))  # (l1, p, r1)*(l2, p, r2) -> (l1, r1, l2, r2)
        overlap = np.tensordot(overlap, overlap_slice, axes=([2,3],[0,2]))

    overlap = overlap.flatten()[0]
    
    return overlap