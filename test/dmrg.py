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
mpo = MPO.Hamiltonian(L)

E_exact, psi_exact = HeisenbergGroundState(L)

psi = MPS.productState(L, [0, 1]*int(L/2))

psi, E_list = dmrg(psi, mpo, chiMax, tol=tol, nSweeps=5)

plt.plot(np.abs(E_list-E_exact), 'o-')
plt.xlabel('Iteration')
plt.ylabel('Error in Energy')
plt.yscale('log')
plt.show()


L = 50
chiMax = 8
tol = 1e-14

mpo = MPO.Hamiltonian(L)

psi = MPS.productState(L, [0, 1]*int(L/2))

psi, E_list = dmrg(psi, mpo, chiMax, tol=tol, nSweeps=5)

plt.plot(np.abs(E_list), 'o-')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.yscale('log')
plt.show()