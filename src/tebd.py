import numpy as np
import scipy as sp

from .svd import svd_truncated

PauliX = np.array([[0,1],[1,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1,0],[0,-1]])
XX = np.kron(PauliX, PauliX)
YY = np.kron(PauliY, PauliY)
ZZ = np.kron(PauliZ, PauliZ)
heisenberg_term = 1/4*(XX + YY + ZZ)

def heisenbergGate(dt):
    """
    Compute the two-site Heisenberg gate for a given time step dt.
    """

    return sp.linalg.expm(-1j*dt*heisenberg_term)


def applyGate(psi, site, dt, chiMax, tol):
    """
    Apply a two-site gate for the Heisenberg model on site and site+1.
    """

    gate = heisenbergGate(dt)

    psi.move_centre_to(site)

    tensor_left = psi.tensors[psi.centre]
    tensor_right = psi.tensors[psi.centre+1]

    theta = np.tensordot(tensor_left, tensor_right, axes=(2,0))  # (l1, p1, r) * (r, p2, r2) -> (l1, p1, p2, r2)
    l1, p1, p2, r2 = theta.shape
    theta = theta.reshape((l1, p1*p2, r2))

    theta = np.tensordot(gate, theta, axes=([1],[1]))  # (out, in) * (l1, in, r2) -> (out, l1, r2)
    out, l1, r2 = theta.shape
    theta = theta.transpose((1,0,2)).reshape(l1*2, 2*r2)

    U, S, Vdg = svd_truncated(theta, chiMax, tol)
    chi = U.shape[1]

    tensor_left = U.reshape((l1, 2, chi))
    tensor_right = np.diag(S) @ Vdg
    tensor_right = tensor_right.reshape((chi, 2, r2))

    psi.tensors[psi.centre] = tensor_left
    psi.tensors[psi.centre+1] = tensor_right
    psi.centre += 1



def TEBD_step(psi, dt, chiMax, tol):
    """
    Perform a single time step of the time-evolution block decimation (TEBD) algorithm. First-order trotterization of the Heisenberg Hamiltonian.

    Parameters
    ----------
    psi : MPS
        Matrix Product State object.
    dt : float
        Time step.
    """

    L = psi.L

    # apply the Heisenberg gate to each pair of even sites
    for i in range(0,L-1,2):
        applyGate(psi, i, dt, chiMax=chiMax, tol=tol)

    for i in range(1,L-1,2):
        applyGate(psi, i, dt, chiMax=chiMax, tol=tol)