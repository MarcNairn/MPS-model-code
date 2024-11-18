import numpy as np
import numpy.linalg as la

def svd_truncated(M, chiMax, threshold):

    U, S, Vdg = la.svd(M, full_matrices=False)
    
    if (chiMax is not None) and (threshold is not None):
        # truncate the singular values
        chi = len(S)
        if chiMax is not None:
            chi = min(chi, chiMax)
        if threshold is not None:
            chi = min(chi, sum(np.cumsum(S**2)/sum(S**2) <= 1-threshold)+1)

        U = U[:, :chi]
        S = S[:chi]
        Vdg = Vdg[:chi, :]
    
    # normalize the singular values
    S = S / la.norm(S)

    return U, S, Vdg