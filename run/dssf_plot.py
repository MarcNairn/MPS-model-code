from fix_pathing import root_dir
from src.dssf import plot_dssf

import numpy as np
from matplotlib import pyplot as plt

L = 101
chi = 8

filename = f'correlator_L{L}_chi{chi}.pickle'

omega = np.linspace(0, 4, 2*L)

plot_dssf(filename, omega)

plt.savefig(f"data/dssf_L{L}_chi{chi}.png", dpi=300, bbox_inches='tight')
plt.show()

