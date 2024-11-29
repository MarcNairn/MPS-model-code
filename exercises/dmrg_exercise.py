from fix_pathing import root_dir

import numpy as np
from src.mps import *
from src.mpo import *
from src.dmrg import *
from src.ed import *

from matplotlib import pyplot as plt

L = 100
chiMax = 16

def plot_entropy(L, ax):
    mpo = MPO.Hamiltonian(L)

    psi = MPS.productState(L, [0]*L)

    psi, _ = dmrg(psi, mpo, chiMax, tol=None, nSweeps=5)

    S_list = []
    for site in range(L-1):
        S_list.append(psi.entropy(site))

    ax.plot(S_list, 'o-')
    ax.set(title=f'L = {L}', xlabel='Bond', ylabel='Entropy', ylim=[0.4,1])


fig, (ax1, ax2) = plt.subplots(1,2)  # create a figure with two subplots

plot_entropy(L, ax1)

plot_entropy(L+1, ax2)

plt.show()