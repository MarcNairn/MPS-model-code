from fix_pathing import root_dir

from src.mps import *
import numpy as np
from src.tebd import *
from src.ed import *

from matplotlib import pyplot as plt

# Define the parameters of the simulation
L = 10  # number of sites
dt = 0.05  # time step
tMax = 6  # maximum time
nSteps = int(tMax / dt)  # number of time steps

t_list = np.arange(nSteps)*dt

magnetization_ED, entanglement_ED = HeisenbergTimeEvolution(L, [0, 1]*int(L/2), dt, tMax)

fig, (ax1, ax2) = plt.subplots(2)  # create a figure with two subplots

ax1.plot(t_list, magnetization_ED, '--', color='k', linewidth=2, label='ED')
ax2.plot(t_list, entanglement_ED, '--', color='k', linewidth=2, label='ED')

for chi in [2,4,8,16]:  # maximum bond dimension

    # Create a product state MPS with alternating spins
    mps = MPS.productState(L, [0, 1]*int(L/2))

    magnetization = []
    entanglement = []
    # Perform the time evolution
    for i in range(nSteps):
        TEBD_step(mps, dt, chiMax=chi, tol=None)
        magnetization.append(mps.expectation(PauliZ, L//2))
        entanglement.append(mps.entropy(L//2-1))

    # Plot the magnetization as a function of time

    ratio = 1 - 0.6*np.log(chi)/np.log(16)
    color = (0.5*ratio,0.8*ratio,ratio,1)

    ax1.plot(t_list, magnetization, color=color, label=f'TEBD chi={chi}')
    ax2.plot(t_list, entanglement, color=color, label=f'TEBD chi={chi}')


plt.legend()
ax1.set(xlabel='', ylabel='Magnetization')
ax2.set(xlabel='Time', ylabel='Entanglement entropy')
plt.show()