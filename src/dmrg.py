import numpy as np
from .mpo import *
from .mps import *
from .svd import svd_truncated

import scipy.sparse as sp


def dmrg(psi, H_mpo, chiMax, tol=1E-12, nSweeps=5):
    """
    Perform density matrix renormalization group (DMRG) for a matrix product state (MPS).

    Parameters
    ----------
    psi : MPS
        Initial matrix product state.
    H_mpo : MPO
        Hamiltonian as a matrix product operator.
    chiMax : int
        Maximum bond dimension.
    tol : float, optional
        Convergence tolerance.
    nSweeps : int, optional
        Number of DMRG sweeps.

    Returns
    -------
    psi : MPS
        Ground state matrix product state.
    E : float
        Ground state energy.
    """
    
    L = psi.L
    assert L == H_mpo.L, "MPS and MPO sizes do not match."
    
    # move centre to right end (won't do anything for product state)
    psi.move_centre_to(L-1)  

    # build left environment list
    L_envs = [np.array([1]).reshape((1,1,1))]
    for i in range(L-2):
        M = H_mpo.get_slice(psi, i)
        L_envs.append(np.tensordot(L_envs[-1], M, axes=([0,1,2], [0,1,2])))

    # build right environment
    R_envs = [np.array([1]).reshape((1,1,1))]

    E_list = []

    for _ in range(nSweeps):

        # sweep right to left in dmrg
        for i in range(L-2, 0, -1):
            psi.move_centre_to(i+1)  # Shouldn't actually do anything (but just to be safe)

            H_block, chi_left, chi_right  = construct_H_block(H_mpo, i, L_envs, R_envs)

            _, V = sp.linalg.eigsh(H_block, k=1, which='SA')

            V = V.reshape(chi_left*2, 2*chi_right)

            U, S, Vdg = svd_truncated(V, chiMax, tol)
            chi = U.shape[1]
            U = U @ np.diag(S)
            psi.centre = i

            psi.tensors[i] = U.reshape(chi_left, 2, chi)
            psi.tensors[i+1] = Vdg.reshape(chi, 2, chi_right)

            # update environments
            L_envs.pop()

            M = H_mpo.get_slice(psi, i+1)
            R_envs.append(np.tensordot(M, R_envs[-1], axes=([3,4,5], [0,1,2])))
        
        # sweep left to right in dmrg
        for i in range(L-2):
            psi.move_centre_to(i)

            H_block, chi_left, chi_right = construct_H_block(H_mpo, i, L_envs, R_envs)

            _, V = sp.linalg.eigsh(H_block, k=1, which='SA')

            V = V.reshape(chi_left*2, 2*chi_right)

            U, S, Vdg = svd_truncated(V, chiMax, tol)
            chi = U.shape[1]
            Vdg = np.diag(S) @ Vdg
            psi.centre = i+1

            psi.tensors[i] = U.reshape(chi_left, 2, chi)
            psi.tensors[i+1] = Vdg.reshape(chi, 2, chi_right)

            # update environments
            R_envs.pop()

            M = H_mpo.get_slice(psi, i)
            L_envs.append(np.tensordot(L_envs[-1], M, axes=([0,1,2], [0,1,2])))

        # compute final energy
        E_list.append(H_mpo.expectation(psi))

    return psi, E_list


def construct_H_block(H_mpo, i, L_envs, R_envs):
    H_block = np.tensordot(L_envs[-1], H_mpo.tensors[i], axes=([1], [0])) 
    H_block = np.tensordot(H_block, H_mpo.tensors[i+1], axes=([4], [0]))
    H_block = np.tensordot(H_block, R_envs[-1], axes=([6], [1]))
    H_block = H_block.transpose(1, 2, 4, 7, 0, 3, 5, 6)
    chi_left, d2, d3, chi_right, d5, d6, d7, d8 = H_block.shape
    H_block = H_block.reshape(chi_left*d2*d3*chi_right, d5*d6*d7*d8)
    return H_block, chi_left, chi_right