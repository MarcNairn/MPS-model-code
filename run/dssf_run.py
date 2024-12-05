from fix_pathing import root_dir
from src.dssf import computeCorrelator

L = 30
chiMax = 8
tol = 1e-14

dt = 0.2
tMax = 8  ## should be approximately L/4. Don't want to hit the boundary.

computeCorrelator(L, dt, tMax, chiMax, tol, entropy=False)