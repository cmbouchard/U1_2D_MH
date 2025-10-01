import os, sys, string
import numba
import numpy as np
import gauge_latticeqcd as gl

### Script to calculate the evolution of the action as a function of Monte Carlo time
Nstart = 0
Nend = 12000
Nx = 6
Ny = 6
action = 'W'
beta = 2.0

def calc_plaq(U):
    return 

### output data vs cfg
### * allow plots of evolution of action with configuration number
### * divide action into contributions from:
###   - flat spacetime QCD (standard LQCD action), leading order

dir = './' + action + '_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(beta * 100)) + '/'
U_infile = dir + 'link_' + action + '_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(beta * 100)) + '_'

### prepare output file
outfile = './P_v_cfg_' + str(int(beta * 100)) + '_' + str(Nx) + 'x' + str(Ny) + '_' + action + '_' + str(Nstart) + '-' + str(Nend) + '.dat'

fout = open(outfile, 'w')
fout.write('#1:cfg  2:ReP 3:ImP\n')

for Ncfg in range(Nstart, Nend + 1):

    ### load lattice data
    U = np.load(U_infile + str(Ncfg))

    ### calculate action
    ReP, ImP = gl.fn_average_plaquette(U)
    fout.write(str(Ncfg) + ' ' + str(ReP) + ' ' + str(ImP) + '\n' )

#end Ncfg
fout.close()
