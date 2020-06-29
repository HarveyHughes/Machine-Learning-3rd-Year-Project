import numpy as np
from CartPole import *
import matplotlib.pyplot as plt
import random
from scipy import stats
from Task_1_Functions import *
#from Non_linear_modelling import *
import sobol_seq



def get_xy_pair(n,system,force=0,f=False,noise=None):  ## returns the nth iternation and the difference bween nth and n+1th for an intialised system
    # x and y both 4x1 matrices
    v=4
    if f:
        v=5

    for i in range(n):
        system.performAction(force)  ## runs for 0.1 secs with no force
    x= np.zeros(v)
    x[0:4] = system.getState()
    if f:
        x[4] = force
    system.performAction(force)
    #system.remap_angle()
    state=system.getState()
    if noise!= None:
        for i in range(4):
            state[i] += random.normalvariate(0,noise)
    y=  state- x[0:4]
    return x[0:v],y

def get_random_data_pairs(n,visual=False,stable_equ=False,ext=False, quasi = False, f = False,noise=None):
    # generates n random state initialisations and their change in state
    # stable_equ if generation is only about theta = pi
    # returns 4xn matrices xs and ys for the initial state and change in state respectivvly
    system = CartPole(visual)

    # initial_state = random_init(initial_state,stable_equ)
    # system.setState(initial_state)
    v=4
    force=0
    if f:
        v=5
    initial_state = np.zeros(v)
    xs = np.zeros((v,n))
    ys = np.zeros((4,n))
    seed=1
    for i in range(n):
        if quasi == False:
            initial_state = random_init(initial_state,stable_equ,ext,f=f)
        else:
            initial_state, seed = quasi_init(seed,f=f)

        system.setState(initial_state[0:4])
        if f:
            force=initial_state[4]
        xs[:,i],ys[:,i] = get_xy_pair(0,system,f=f,force=force,noise=noise)    ## each column is a sample
    return xs,ys


def linear_regression(x,y, f=False):
    #x is (vxn) matrix of states, y is (4xn) matrix of state changes
    # minimise Y - CX
    # returns the 4xv coefficent matrix c
    # plots state predictions for y against true y
    c = np.linalg.lstsq(x.T,y.T,rcond=None)[0]
    c=c.T
    predicted_y = np.matmul(c,x)
    v=4
    if f:
        v=5

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.scatter(y[i,:],predicted_y[i,:], marker = 'x')
        plt.ylabel('Predicted next step')
        plt.xlabel('Actual next step')
        plt.title (labels[i])

        grad,int,r,p,se = stats.linregress(y[i,:],predicted_y[i,:])
        print(labels[i] , ' gradient = ', grad)
        print(labels[i], ' inter = ', int)
        print(labels[i], ' r = ', r)
        print(labels[i], ' r^2 = ', r**2)
    plt.show()


    return c



