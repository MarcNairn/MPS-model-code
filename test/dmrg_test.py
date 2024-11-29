from fix_pathing import root_dir

import numpy as np
from src.mps import *
from src.mpo import *
from src.dmrg import *
from src.ed import *

from matplotlib import pyplot as plt

L = 10
chiMax = 32
tol = 1e-14

H = HeisenbergHamiltonian(L)
E_exact, psi_exact = HeisenbergGroundState(L)

mpo = MPO.Hamiltonian(L)

for chiMax in [2, 4, 8, 16, 32]:

    psi = MPS.productState(L, [0]*L)

    psi, E_list = dmrg(psi, mpo, chiMax, tol=tol, nSweeps=5)

    plt.plot(np.abs(E_list-E_exact), 'o-', label=f'chiMax={chiMax}')

plt.xlabel('Iteration')
plt.ylabel('Error in Energy')
plt.yscale('log')
plt.legend()
plt.show()
