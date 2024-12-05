from src.tebd import *
from src.dmrg import *
from src.overlap import overlap

import math
import pickle
from matplotlib import pyplot as plt

def correlator(psi, E_0, j, dt, tMax, chiMax, tol, entropy=False):

    L = psi.L

    nSteps = int(tMax / dt / 2)  # number of time steps

    t_list = np.arange(2*nSteps+1)*dt

    psi_0 = psi.copy()
    psi_j = psi.copy()

    psi_0.applyOperator(PauliZ, math.ceil(L/2)-1)
    psi_j.applyOperator(PauliZ, j)

    entropy_list = []
    correlator = [psi_0.overlap(psi_j)]
    for ii in range(nSteps):
        TEBD_step(psi_0, dt, chiMax, tol=tol)

        correlator.append(np.exp(1j*E_0*dt*(2*ii+1))*overlap(psi_j,psi_0))
       
        TEBD_step(psi_j, -dt, chiMax, tol=tol)

        correlator.append(np.exp(1j*E_0*dt*(2*ii+2))*overlap(psi_j,psi_0))

        if entropy:
            entropy_list.append(psi_j.entropy(L//2))

    if entropy:
        return t_list, correlator, entropy_list
    else:
        return t_list, correlator
    

def computeCorrelator(L, dt, tMax, chiMax, tol):
    
    nSteps = int(tMax / dt / 2)  # number of time steps

    t_list = np.arange(2*nSteps+1)*dt
    H_mpo = MPO.Hamiltonian(L)  # Hamiltonian MPO

    psi = MPS.productState(L, [0]*int(L))  # product state MPS initialization

    # DMRG to find ground state and E_0
    psi, E_list = dmrg(psi, H_mpo, chiMax, tol=tol, nSweeps=3)
    E_0 = E_list[-1]

    print("Found the ground state!")

    correlator_list = []
    for j in range(L):
        print("site", j)

        # compute the correlator at site j
        t_list, correlator = correlator(psi, E_0, j, dt, tMax, chiMax, tol)
        correlator_list.append(correlator)

    correlator = np.array(correlator_list)

    # save the correlator data to file using pickle
    data = {'L': L, 't_list': t_list, 'correlator': correlator}

    with open(f'../data/correlator_L{L}_chi{chiMax}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_dssf(filename):

    with open('../data/'+filename, 'rb') as handle:
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

    j_list = np.array([j for j in range(n_sites)])
    j_list = np.tile(j_list, (L,1))

    exp_kj = np.exp(-1j*k_list*j_list)

    # t->omega Fourier Transform
    omega = np.linspace(0, 15, 2*L)
    t_list_rep = np.tile(t_list, (len(omega),1)).T
    omega_rep = np.tile(omega, (len(t_list),1))
    exp_omega = np.exp(1j*omega_rep*t_list_rep)

    # Fourier Transform of the correlator with window function
    DSSF = exp_kj @ np.real((correlator * window_function) @ exp_omega)

    # plot the DSSF
    plt.imshow((np.abs(DSF)[:,:]).T, aspect='auto', origin='lower', cmap='magma_r', extent=[0, 2*np.pi, 0, omega[-1]/4])
    plt.xlabel('k')
    plt.ylabel('$\omega$')
    plt.xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])
    plt.clim([0,100])
    plt.colorbar()
    plt.show()