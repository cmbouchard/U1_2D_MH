from __future__ import print_function
import numba
import numpy as np
import sys
import lattice_collection as lc
import datetime

### File with Lattice class to sweep and generate lattices and functions: 
### plaquette, average plaquette, polyakov, planar and non-planar wilson loops, wilson action,
### operator density and operator sum helper functions, topological charge included.


#-------------Measurment code -------------------
### some functions are reproduced here, outside of Lattice class, to be accessible via function call.
### - a bit redundant
def fn_periodic_link(U, xy, direction):
    Nx, Ny = len(U), len(U[0])
    return U[xy[0] % Nx][xy[1] % Ny][direction]

def fn_move_forward_link(U, xy, direction):
    link = fn_periodic_link(U, xy, direction)
    new_xy = xy[:]
    new_xy[direction] += 1
    return link, new_xy

def fn_move_backward_link(U, xy, direction):
    new_xy = xy[:]
    new_xy[direction] -= 1
    link = np.conj( fn_periodic_link(U, new_xy, direction) )
    return link, new_xy

def fn_line_move_forward(U, line, xy, direction):
    link, new_xy = fn_move_forward_link(U, xy, direction)
    new_line = line * link
    return new_line, new_xy

def fn_line_move_backward(U, line, xy, direction):
    link, new_xy = fn_move_backward_link(U, xy, direction)
    new_line = line *  link
    return new_line, new_xy

### plaquette calculation
def fn_plaquette(U, x, y, mu, nu):
    Nx = len(U)
    Ny = len(U[0])
    start_xy = [x, y]
    result = 1. + 0. * 1J
    result, next_xy = fn_line_move_forward(U, result, start_xy, mu)
    result, next_xy = fn_line_move_forward(U, result, next_xy, nu)
    result, next_xy = fn_line_move_backward(U, result, next_xy, mu)
    result, next_xy = fn_line_move_backward(U, result, next_xy, nu)    
    return result

### calculate average plaquette
### return real(P), imag(P)
#@numba.njit
def fn_average_plaquette(U):
    Nx, Ny = map(len, [U, U[0]])
    res = 0. + 0. * 1J
    for x in range(Nx):
        for y in range(Ny):
            #for mu in range(1, 2):
            #    for nu in range(mu):
            mu, nu = 1, 0
            res += fn_plaquette(U, x, y, mu, nu)
    return np.real(res) / (Nx * Ny), np.imag(res) / (Nx * Ny)

### Wilson action at a specific point
### S = beta \sum_x \sum_{\mu > \nu} (1 - Re P_{\mu\nu}(x))
###   * P_{\mu\nu}(x) = U_\mu(x) U_\nu(x + \hat\mu) U^\dagger_mu(x + \hat\nu) U^\dagger_\nu(x)
###   * U_\mu(x) = exp(iae A_\mu(x))
###   * fn_plaquette(U, x, y, mu, nu) returns the product of links around the plaquette, P_{\mu\nu}(x)
###   * beta = 1 / e^2
def fn_eval_point_S(U, x, y, beta):
    #tmp = 0.
    #for mu in range(1, 2):  #sum over \mu > \nu spacetime dimensions
    #    for nu in range(mu):
    mu, nu = 1, 0    
    tmp = 1. - np.real( fn_plaquette(U, x, y, mu, nu) )
    return beta * tmp

### Calculate density for given operator.
### Requires lattice and operator to calculate along with all arguments that need to be passed to operator.
def fn_operator_density(U, operator_function, *args):
    Nx, Ny = map(len, [U, U[0]])
    tmp = [[0 for y in range(Ny)] for x in range(Nx)]
    for x in range(Nx):
        for y in range(Ny):
            tmp[x][y] = operator_function(U, x, y, *args)
    return tmp


### Calculate sum for given operator over whole lattice.
### Requires lattice and operator to calculate along with all arguments that need to be passed to operator.
def fn_sum_over_lattice(U, operator_function, *args):
    Nx, Ny = map(len, [U, U[0]])
    sum_lattice = 0.
    for x in range(Nx):
        for y in range(Ny):
            sum_lattice += operator_function(U, x, y, *args)
    return sum_lattice

### field strength from plaquette, returns a^2 F_\mu\nu(x)
### P_\mu\nu(x) = 1 + ia^2e F_\mu\nu(x) - a^4e^2 F_\mu\nu(x)^2 / 2 + ...
def fn_F_munu(U, x, y, mu, nu, beta):
    Pmunu = fn_plaquette(U, x, y, mu, nu)
    return np.imag(Pmunu) * beta**0.5


#-------------Generation code -------------------
### function called by multiprocessor in generate script
def generate(beta, action, Nx, Ny, startcfg, Ncfg, Nhits, epsilon):    
    
    ### loop over (x,y) and mu and set initial collection of links
    ### Either:
    ###  1. initialize to cold start by initializing U(1) to 1, or
    ###  2. read in a previously generated configuration and continue with that Markov chain.

    name = action +'_' + str(Nx) + 'x' + str(Ny) + '_b' + str(int(beta * 100))

    print('simulation parameters:')
    print('      action: ' + action)
    print('Nx,Ny = ' + str(Nx) + ',' + str(Ny))
    print('       beta = ' + str(beta))
    print('      Nhits = ' + str(Nhits))
    print('      start = ' + str(startcfg))
    print('     sweeps = ' + str(Ncfg))

    if startcfg == 0:
        U = lattice(Nx, Ny, beta)
    else:
        U = lc.fn_load_configuration(action, Nx, Ny, beta, startcfg, "./")
        U = lattice(Nx, Ny, beta, U)
    
    print('Continuing from cfg: ', startcfg)
    print('... generating lattices')
    acceptance = U.markov_chain_sweep(Ncfg, startcfg, name, Nhits, action, epsilon)
    print("acceptance:", acceptance)


### LATTICE CLASS
class lattice():
    ### Lattice initialization.
    ### If U not passed, lattice of identities returned.
    ### Class to avoid use of incorrect initialization and passing a lot of variables
    #@numba.njit
    def __init__(self, Nx, Ny, beta, U=None):
        if None == U:
            # initialize to 1, giving S=0, cold start
            U = [[[1. + 0. * 1J for mu in range(2)] for y in range(Ny)] for x in range(Nx)]
        # convert to numpy arrays -> significant speed up
        self.U = np.array(U)
        self.beta = beta
        self.Nx = Nx
        self.Ny = Ny
        
    ### calculate link imposing periodic boundary conditions
    def periodic_link(self, xy, direction):
        return self.U[xy[0] % self.Nx, xy[1] % self.Ny, direction]
    
    def move_forward_link(self, xy, direction):
        link = self.periodic_link(xy, direction)
        new_xy = xy[:]
        new_xy[direction] += 1
        return link, new_xy

    def move_backward_link(self, xy, direction):
        new_xy = xy[:]
        new_xy[direction] -= 1
        link = np.conj( self.periodic_link(new_xy, direction) )
        return link, new_xy

    def line_move_forward(self, line, xy, direction):
        link, new_xy = self.move_forward_link(xy, direction)
        new_line = line * link
        return new_line, new_xy

    def line_move_backward(self, line, xy, direction):
        link, new_xy = self.move_backward_link(xy, direction)
        new_line = line * link
        return new_line, new_xy

    ###WILSON ACTION staple
    #@numba.njit
    def dS_staple(self, x, y, mu):
        tmp1, tmp2 = 0. + 0. * 1J, 0. + 0. * 1J
        for nu in range(2):
            if nu != mu:

                #Determine required points for the calculation of the action
                start_xy = [x, y]
                start_xy[mu] += 1

                ### staple 1
                line1 = 1. + 0. * 1J
                line1, next_xy = self.line_move_forward(line1, start_xy, nu)
                line1, next_xy = self.line_move_backward(line1, next_xy, mu)
                line1, next_xy = self.line_move_backward(line1, next_xy, nu)
                tmp1 += line1
                
                ### staple 2, opposite orientation to staple 1
                line2 = 1. + 0. * 1J
                line2, next_xy = self.line_move_backward(line2, start_xy, nu)
                line2, next_xy = self.line_move_backward(line2, next_xy, mu)
                line2, next_xy = self.line_move_forward(line2, next_xy, nu)
                tmp2 += line2
        
        return tmp1, tmp2
    

    ### Difference of action at a point for fixed staple. Gets link, updated link, and staples A1, A2
    ### Need (U'-U).A1 + [(U'-U).A2]^\dagger, to correct for orientation of A2
    def deltaS(self, link, updated_link, staple1, staple2):
        tmp = (updated_link - link) * staple1 + np.conj((updated_link - link) * staple2)
        return -self.beta * np.real( tmp )


    #@numba.njit
    def plaquette(self, x, y, mu, nu):
        Nx, Ny = self.Nx, self.Ny
        start_xy = [x, y]
        result = 1. + 0. * 1J
        result, next_xy = self.line_move_forward(result, start_xy, mu)
        result, next_xy = self.line_move_forward(result, next_xy, nu)
        result, next_xy = self.line_move_backward(result, next_xy, mu)
        result, next_xy = self.line_move_backward(result, next_xy, nu)
        return result
    
    ### return real(P), imag(P)
    #@numba.njit
    def average_plaquette(self):
        Nx, Ny = self.Nx, self.Ny
        res = 0. + 0. * 1J
        for x in range(Nx):
            for y in range(Ny):
                mu, nu = 1, 0
                res += self.plaquette(x, y, mu, nu)
        return np.real(res) / (Nx * Ny), np.imag(res) / (Nx * Ny)
    

    ### Markov chain sweep. Requires: 
    ###   number of cfgs,
    ###   initial cfg,
    ###   save name (if given, otherwise will not save),
    ###   hits per sweep,
    ###   action-> W for Wilson 
    def markov_chain_sweep(self, Ncfg, initial_cfg=0, save_name='', Nhits=10, action='W', epsilon=0.2):
        ratio_accept = 0.
        if save_name:
            output = save_name + '/link_' + save_name + '_'
        
        ### loop through number of configurations to be generated
        for i in range(Ncfg - 1):
            print('starting sweep ' + str(i) + ':  ' + str(datetime.datetime.now()))

            ### loop through spacetime dimensions
            for x in range(self.Nx):
                for y in range(self.Ny):
                    ### loop through directions
                    for mu in range(2):
                        A1, A2 =  self.dS_staple(x, y, mu) #Wilson staples
                        ### loop through hits
                        for j in range( Nhits ):
                            ### let small be a random, small fraction of 2pi
                            small = np.random.uniform(-1, 1) * epsilon * np.pi
                            r = np.cos(small) + 1J * np.sin(small)
                            ### create U'
                            Uprime = r * self.U[x, y, mu]
                            ### calculate staple
                            dS = self.deltaS(self.U[x, y, mu], Uprime, A1, A2)
                            ### check if U' accepted
                            if (np.exp(-1. * dS) > np.random.uniform(0, 1)):
                                self.U[x, y, mu] = Uprime
                                ratio_accept += 1
                                        

            ### save if name given
            if (save_name):
                idx = int(i) + initial_cfg
                output_idx = output + str(int( idx ))
                file_out = open(output_idx, 'wb')
                np.save(file_out, self.U)  #NOTE: np.save without opening first appends .npy
                sys.stdout.flush()
        
        ratio_accept = float(ratio_accept) / Ncfg / self.Nx / self.Ny / 2. / Nhits
        return ratio_accept
