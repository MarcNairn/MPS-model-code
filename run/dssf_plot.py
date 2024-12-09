from fix_pathing import root_dir
from src.dssf import plot_dssf

import numpy as np

L = 100
chi = 32

filename = f'correlator_L{L}_chi{chi}.pickle'

omega = np.linspace(0, 4, 2*L)

plot_dssf(filename, omega)


