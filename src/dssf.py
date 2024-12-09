from src.tebd import *
from src.dmrg import *
from src.overlap import overlap

import math
import pickle
from matplotlib import pyplot as plt

from multiprocessing import Manager, Pool


def correlator(procid, return_dict, psi, E_0, j, dt, tMax, chiMax, tol, entropy=False):
    """
    Compute the time evolution of the correlator <psi0|Z_j(t) Z_N/2(0)|psi0> using TEBD, for a single site j.
    Includes flag to return the half-chain entropy at each time step.

    Args:
    psi (MPS): mps ground state
    E_0 (float): ground state energy
    j (int): site index
    dt (float): time step
    tMax (float): maximum time
    chiMax (int): maximum bond dimension
    tol (float): convergence tolerance
    entropy (bool): flag to compute entropy

    Writes date to file.
    """

    L = psi.L

    nSteps = int(tMax / dt / 2)  # number of time steps

    t_list = np.arange(2*nSteps+1)*dt

    psi_0 = psi.copy()
    psi_j = psi.copy()

    psi_0.applyOperator(PauliZ, math.ceil(L/2)-1)
    psi_j.applyOperator(PauliZ, j)

    entropy_list = []
    correlator_list = [overlap(psi_j,psi_0)]
    for ii in range(nSteps):
        TEBD_step(psi_0, dt, chiMax, tol=tol)

        correlator_list.append(np.exp(1j*E_0*dt*(2*ii+1))*overlap(psi_j,psi_0))
       
        TEBD_step(psi_j, -dt, chiMax, tol=tol)

        correlator_list.append(np.exp(1j*E_0*dt*(2*ii+2))*overlap(psi_j,psi_0))

        if entropy:
            entropy_list.append(psi_j.entropy(L//2))

    if entropy:
        return_dict[procid] = (t_list, correlator_list, entropy_list)
    else:
        return_dict[procid] = (t_list, correlator_list)
    

def computeCorrelator(L, dt, tMax, chiMax, tol, entropy=False):
    """
    Compute the correlators <Z_j(t) Z_N/2(0)> for all sites j using TEBD.
    Starts by computing the ground state using DMRG.
    """

    nSteps = int(tMax / dt / 2)  # number of time steps

    t_list = np.arange(2*nSteps+1)*dt
    H_mpo = MPO.Hamiltonian(L)  # Hamiltonian MPO

    psi = MPS.productState(L, [0]*int(L))  # product state MPS initialization

    # DMRG to find ground state and E_0
    psi, E_list = dmrg(psi, H_mpo, chiMax, tol=tol, nSweeps=3)
    E_0 = E_list[-1]

    print("Found the ground state!")

    processes = []
    manager = Manager()
    return_dict = manager.dict()

    pool = Pool()

    # Create len(nums) processes
    for procid, j in enumerate(range(L)):
        pool.apply_async(correlator, args=(procid, return_dict, psi.copy(), E_0, j, dt, tMax, chiMax, tol, entropy,))
    pool.close()
    pool.join()
    
    t_list = return_dict[0][0]
    correlator_list = [return_dict[val][1] for val in range(L)]
    correlator_array = np.array(correlator_list)
    if entropy:
        entropy_list = [return_dict[val][2] for val in range(L)]


    # save the correlator data to file using pickle
    data = {'L': L, 't_list': t_list, 'correlator': correlator_array}
    if entropy:
        data['entropy'] = entropy_list

    with open(f'data/correlator_L{L}_chi{chiMax}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_dssf(filename, omega):
    """
    Load the data from file and plot the dynamical structure factor.
    """

    with open('data/'+filename, 'rb') as handle:
        data = pickle.load(handle)

    L = data['L']
    t_list = data['t_list']
    correlator = data['correlator']
    n_sites = len(correlator)

    # define the window function
    window_function = np.cos(t_list/t_list[-1]*np.pi/2)**2

    # construct j->k Fourier Transform as a matrix
    k_list = np.array([2*np.pi*j/(L) for j in range(L)])
    k_list = np.tile(k_list, (n_sites,1)).T

    j_list = np.array([j - math.ceil(L/2)-1 for j in range(n_sites)])
    j_list = np.tile(j_list, (L,1))

    exp_kj = np.exp(-1j*k_list*j_list)

    # t->omega Fourier Transform
    t_list_rep = np.tile(t_list, (len(omega),1)).T
    omega_rep = np.tile(omega, (len(t_list),1))
    exp_omega = np.exp(1j*omega_rep*t_list_rep)

    # Fourier Transform of the correlator with window function
    DSSF = 8*t_list[1]*exp_kj @ np.real((correlator * window_function) @ exp_omega)

    # plot the DSSF
    plt.imshow((np.abs(DSSF)).T, aspect='auto', origin='lower', cmap='magma_r', extent=[0, 2*np.pi, 0, omega[-1]])
    plt.xlabel('k')
    plt.ylabel('$\omega$')
    plt.xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])
    plt.clim([0,50])
    plt.colorbar()