import os, sys, string
import numba
import numpy as np
import gauge_latticeqcd as gl

### Script to calculate the evolution of the action as a function of Monte Carlo time
Nstart = 0
Nend = 1000000
Nx = 16
Ny = 16
action = 'W'
beta = 9.0

def calc_S(U):
    Nx = len(U)
    Ny = len(U[0])
    S = 0.
    for x in range( Nx ):
        for y in range( Ny ):
            S += gl.fn_eval_point_S(U, x, y, beta)
        #end y
    #end x
    return S

### output data vs cfg
### * allow plots of evolution of action with configuration number
### * divide action into contributions from:
###   - flat spacetime QCD (standard LQCD action), leading order

dir = './' + action + '_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(beta * 100)) + '/'
U_infile = dir + 'link_' + action + '_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(beta * 100)) + '_'

### prepare output file
outfile = './S_v_cfg_' + str(int(beta * 100)) + '_' + str(Nx) + 'x' + str(Ny) + '_' + action + '_' + str(Nstart) + '-' + str(Nend) + '.dat'

fout = open(outfile, 'w')
fout.write('#1:cfg  2:S\n')

for Ncfg in range(Nstart, Nend + 1):

    ### load lattice data
    U = np.load(U_infile + str(Ncfg))

    ### calculate action
    S = calc_S(U)
    fout.write(str(Ncfg) + ' ' + str(S) + '\n' )

#end Ncfg
fout.close()
