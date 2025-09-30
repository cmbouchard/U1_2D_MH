### Script to generate quenched U(1) gauge fields in 2D 

from __future__ import print_function
import os, sys, string
import numpy as np
from gauge_latticeqcd import *


### settings
Nx = 16
Ny = 16
startcfg = 500000       # warm start (0) or existing cfg number to start the Markov chain
Ncfg = 500002          # number of lattices to generate
action = 'W'       # W = Wilson
betas = [7.0, 9.0]      # betas to be generated, beta = 1/e^2
Nhits = 10         # hits between each update
epsilon = 0.55      # U -> U' = U * exp(i * r * epsilon * pi), random -1 < r < 1
threads = 1        # threads used in multiprocessing


### generate lattices
for b in betas:

    dir_name = action + '_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(b * 100))
    
    ### create output directory if it does not exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name) 
    else:
        print("Directory exists for beta ", b)

    generate(beta=b, action=action, Nx=Nx, Ny=Ny, startcfg=startcfg, Ncfg=Ncfg, Nhits=Nhits, epsilon=epsilon)

### initialize multiprocessing
#p = Pool(threads)
### function to be calculated needs to use functools to work with map
#func = functools.partial(generate, action=action, Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz, startcfg=startcfg, Ncfg=Ncfg, Nhits=Nhits, epsilon=epsilon)
#p.map(func, betas) # call multiprocessing map function
#p.terminate()      # terminate multiprocessing
