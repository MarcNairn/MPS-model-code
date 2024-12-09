from os import environ
environ['OMP_NUM_THREADS'] = '1'

from fix_pathing import root_dir
from src.dssf import computeCorrelator
from multiprocessing import freeze_support

import time

if __name__ == '__main__':
    freeze_support()
    L = 101
    chiMax = 8
    tol = 1e-14

    dt = 0.1
    tMax = 25  ## should be approximately L/4. Don't want to hit the boundary.

    t1 = time.time()
    computeCorrelator(L, dt, tMax, chiMax, tol, entropy=False)
    t2 = time.time()
    print(f"Time taken: {t2-t1:.2f} s")
