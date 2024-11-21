from fix_pathing import root_dir

from src.mps import *
import numpy as np
from src.tebd import *
from src.ed import *

from matplotlib import pyplot as plt

# Define the parameters of the simulation
L = 51  # number of sites
dt = 0.1  # time step
tMax = 25  # maximum time
nSteps = int(tMax / dt)  # number of time steps

t_list = np.arange(nSteps)*dt

chi = 8 # maximum bond dimension

# Create a product state MPS with alternating spins
state_list = [0]*L
state_list[L//4] = 1
state_list[L//2] = 1
state_list[3*L//4] = 1

mps = MPS.productState(L, state_list)

magnetization = []
# Perform the time evolution
for i in range(nSteps):
    TEBD_step(mps, dt, chiMax=chi, tol=None)
    magneization_slice = []
    for j in range(L):
        magneization_slice.append(mps.expectation(PauliZ, j))
    magnetization.append(magneization_slice)

print(mps.tensors[L//2].shape)

# Image plot the magnetization as a function of time
magnetization = np.array(magnetization)
plt.imshow(magnetization, aspect='auto', origin='lower', cmap='RdBu', extent=[-0.5, L+0.5, 0, tMax])
plt.colorbar()
plt.ylabel('Time')
plt.xlabel('Site')
plt.title('Magnetization')
plt.show()
