import numpy as np
from CartPole import *
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import sobol_seq

labels = [ "Position" , "Velocity" , "Angle" , "Angular Velocity"]
ranges = [50.,20.,2*np.pi,30. , 20] ## the size of suitable ranage
#ranges = [100.,40.,2*np.pi,40. , 20]


def test_force_start(f=0,n=30, all=False):
    system = CartPole(visual=False)
    system.setState([0, 0, np.pi, 0])  ## initialise
    state_history = np.empty((n + 1, 4,n))
    for i in range(n):
        state_history[0, :,i] = [0, 0, np.pi,0]
    time = np.arange(0, (n + 0.5) * 0.1, 0.1)
    for fn in range(n):
        system.setState([0, 0, np.pi, 0])
        for i in range(n):  # update dynamics n times
            if i < fn+1:
                system.performAction(f)  ## runs for 0.1 secs with no
            else:
                system.performAction(0)
            system.remap_angle()
            state_history[i + 1, :,fn] = system.getState()#
        if fn%3==0:
            plt.plot(time, state_history[:, 2,fn], Label=str(fn+1) + ' Steps')
            if all:
                plt.plot(time, state_history[:, 0, fn], Label=str(fn + 1) + ' Steps-x')
                plt.plot(time, state_history[:, 1, fn], Label=str(fn + 1) + ' Steps-v')
                plt.plot(time, state_history[:, 3, fn], Label=str(fn + 1) + r' Steps-$\dot{\theta}$')
    plt.xlabel('Time')
    plt.ylabel(r'$\theta$')
    plt.legend(loc='lower left')
    plt.title('Constant force effect for start')
    plt.show()


def rollout (initial_state, n = 100, visual =False,f=0,fn=0):
    ## takes an intitial state array for the initial cart and angular velocity,
    ## itialises at stable equilibrium theta = pi , x =0
    ## runs for n iterations of euler dynamics each iteration is 0.1 seconds
    ## no applied force
    ## plots state variables

    system = CartPole(visual)
    system.setState([0, initial_state[0], np.pi, initial_state[1]])  ## initialise
    state_history= np.empty((n+1,4))
    state_history[0,:] = [0, initial_state[0], np.pi, initial_state[1]]
    for i in range(n): #update dynamics n times
        if i<fn:
            system.performAction(f)  ## runs for 0.1 secs with no
        else: system.performAction(0)
        system.remap_angle()
        state_history[i+1,:] = system.getState()
    time=np.arange(0,(n+0.5)*0.1,0.1)


    plt.plot(time,state_history[:,0], Label = 'Position')
    plt.plot(time,state_history[:,1], Label = 'Velocity')
    plt.plot(time,state_history[:,2], Label = 'Angular Position')
    plt.plot(time,state_history[:,3], Label = 'Angular Velocity')
    plt.xlabel('Time')
    plt.legend(loc='upper right')
    plt.title('Initial Velocity : '+ str(initial_state[0])+ ' , Initial Angular velocity : '+ str(initial_state[1]))
    plt.show()

    # plt.plot(state_history[:,0], state_history[:, 1])
    # plt.xlabel('Position')
    # plt.ylabel('Velocity')
    # plt.show()

def quasi_init(seed,f=False):
    v =4
    if f:
        v=5
    x, seed = sobol_seq.i4_sobol(v, seed)
    for i in range(v):
        x[i] = x[i]* ranges[i] - ranges[i]/2 # produces rando float in the ranges band described above
    return x,seed

def random_init(x,stable_equ=False,ext =False,f=False):
    # generates a random start state in the correct range
    v=4
    if f:
        v=5
    for i in range(v):
        x[i] = random.random() * ranges[i] - ranges[i]/2 # produces rando float in the ranges band described above
    if stable_equ==True:
        x[2] = random.random() * 0.2
        if x[2] > 0.1:
            x[2]=np.pi-x[2]
        else: x[2] = -np.pi+x[2]
    elif ext == True: ## get 2 times the range
            x[0] = x[0]*2
            x[1] = x[1]*2
            x[3] = x[3]*2
    return x

def scan_step(to_scan,n,visual=False,f=False):
    # scnes through one state variable with n points scan variable given by to_scan = [0,3]
    # plots the next state vector


    system = CartPole(visual)
    initial_state = np.zeros(4)
    initial_state = random_init(initial_state,f=f)

    initial_state[to_scan] = 0 - ranges[to_scan]/2   ##makes it scan from the start of a variables range
    scanned_values = np.zeros(n)
    reached_states = np.zeros((n,4))
    step = ranges[to_scan]/n
    for i in range(n): ##number of scanning variables
        x = initial_state.copy() ##takes a copy of this state
        x[to_scan] = x[to_scan]+ i * step ##changes one state variable
        system.setState(x)
        if to_scan == 2:
            system.remap_angle()              ##remaps the angle
            x[to_scan] = remap_angle(x[to_scan])
        scanned_values[i] = x[to_scan]

        system.performAction(x[4])
        reached_states[i,:] = system.getState()

    plt.plot(scanned_values, reached_states[:, 0], Label='Position')
    plt.plot(scanned_values, reached_states[:, 1], Label='Velocity')
    plt.plot(scanned_values, reached_states[:, 2], Label='Angle')
    plt.plot(scanned_values, reached_states[:, 3], Label='Angular Velocity')


    plt.xlabel(labels[to_scan])
    plt.ylabel('Reached state')
    plt.legend(loc='upper left')
    plt.title(
            'Initial state: ' + str(np.round(initial_state,2)) + ' Scan through ' + labels[to_scan] )
    plt.show()

def scan_all(n,type=0,visual=False, c =None,f=False,nlm=False):  ##type 0 is a regular scan, type 1 is a change in variable scan
    ## c iks a 4x4 linear coefficent matrix for plotting model scans
    # scans through all state variables and plots either the next state or the change in state

    system = CartPole(visual)
    if f:
        initial_state = np.zeros(5)
    else:
        initial_state = np.zeros(4)
    initial_state = random_init(initial_state,f=f)
    line_lables=[None,None,None,None,None,None,None,None]
    fig = plt.figure()
    for to_scan in range(4):
        plot_no=to_scan
        scanned_state = np.zeros((4,n))
        if nlm:
            if c.v==5:
                to_scan+=1
                scanned_state = np.zeros((5, n))
        initial_state[to_scan] = 0 - ranges[to_scan]/2   ##makes it scan from the start of a variables range
        scanned_values = np.zeros(n)
        reached_states = np.zeros((n,4))

        Y = np.zeros((n,4))
        step = ranges[to_scan]/n
        for i in range(n): ##number of scanning variables
            x = initial_state.copy() ##takes a copy of this state
            x[to_scan] = x[to_scan]+ i * step ##changes one state variable
            system.setState(x)
            if to_scan == 2:
                system.remap_angle()              ##remaps the angle
                x[to_scan] = remap_angle(x[to_scan])
            scanned_values[i] = x[to_scan]
            scanned_state[:,i] = x

            if f:
                system.performAction(x[4])
            else:
                system.performAction(0)
            reached_states[i,:] = system.getState()
            Y[i,:] = reached_states[i,:] - x[:4]

        fig.add_subplot(2,2,plot_no+1)
        plt.subplots_adjust(hspace=0.3)
        if plot_no == 3: #add labels
            line_lables=['x','v',r"$\theta$",r"$\dot{\theta}$",'Predicted x','Predicted v',r"Predicted $\theta$",r" Predicted $\dot{\theta}$"]

        if type == 0 : #plot regular scan
            plt.plot(scanned_values, reached_states[:, 0], Label=line_lables[0])
            plt.plot(scanned_values, reached_states[:, 1], Label=line_lables[1])
            plt.plot(scanned_values, reached_states[:, 2], Label=line_lables[2])
            plt.plot(scanned_values, reached_states[:, 3], Label=line_lables[3])
            plt.ylabel('Reached state')

        elif type ==1: #plot change
            if np.any(c)!=None:
                if nlm:
                    scanned_state=c.transform_x(scanned_state)
                    predictions = np.matmul(c.alpha.T,scanned_state)
                else:
                    predictions = np.matmul(c,scanned_state)

                plt.plot(scanned_values[:], predictions[0, :],linestyle=':',color= 'b', Label=line_lables[4])
                plt.plot(scanned_values[:], predictions[1, :],linestyle=':',color= 'orange', Label=line_lables[5])
                plt.plot(scanned_values[:], predictions[2, :],linestyle=':',color= 'g', Label=line_lables[6])
                plt.plot(scanned_values[:], predictions[3, :],linestyle=':',color= 'r', Label=line_lables[7])

            plt.plot(scanned_values[:], Y[:, 0],color= 'b', Label=line_lables[0])
            plt.plot(scanned_values[:], Y[:, 1],color= 'orange', Label=line_lables[1])
            plt.plot(scanned_values[:], Y[:, 2],color= 'g', Label=line_lables[2])
            plt.plot(scanned_values[:], Y[:, 3],color= 'r', Label=line_lables[3])
            plt.ylabel('Change in State Variable')


        labels = [ "Position" , "Velocity" , "Angle" , "Angular Velocity", "Force"]
        plt.xlabel(labels[to_scan])
        #plt.legend(loc='upper left')
        plt.title(
                'Initial state: ' + str(np.round(initial_state,2)) + ' Scan through ' + labels[to_scan] )
    fig.legend(loc='lower center', ncol=4)
    plt.show()


def contour_plot(n,x_var, y_var , cont, visual = False , model = None,error=False,f=False):
    # plots a 3d surface plot and contoru plot where x_var , Y_var are scanned through the suitable range
    # cont is the variable to be plotted on the contours
    #n is the number of points to evaluate at in the range
    # x_var,y_var and cont are either 0,1,2,3
    v=4
    if f:
        v=5
    system = CartPole(visual)
    initial_state = np.zeros(v)
    initial_state = random_init(initial_state,f=f)
    force=0
    if f:
        force = initial_state[4]

    scanned_x = np.arange( - ranges[x_var]/2, ranges[x_var]/2, ranges[x_var]/n )  ##scanning variable 1yield
    scanned_y = np.arange( - ranges[y_var]/2, ranges[y_var]/2, ranges[y_var]/n )  ##scanning variable 1yield

    X, Y = np.meshgrid(scanned_x, scanned_y)

    rav_X=np.ravel(X)
    rav_Y=np.ravel(Y)
    rav_Z=np.zeros(rav_X.shape[0])
    for i in range(rav_X.shape[0]):
        x = initial_state.copy()
        x[x_var] = rav_X[i]
        x[y_var] = rav_Y[i]

        system.setState(x)

        system.performAction(force)
        rav_Z[i] = system.getState()[cont]

        if error ==True:
            change = np.zeros((1,v))
            change[0,0:4] = np.matmul(model.alpha.T, model.transform_x(x)).T

            next_state = x + change

            rav_Z[i] = abs(next_state[0,cont] - rav_Z[i])

    Z = rav_Z.reshape(X.shape)

    colour = 'inferno'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1 , cmap=colour)
    # Add a color bar which maps values to colors.
    cbar =fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label(labels[cont])
    plt.xlabel(labels[x_var])
    plt.ylabel(labels[y_var])
    ax.set_zlabel(labels[cont])
    plt.title('Initial state: ' + str(np.round(initial_state,2)) + '\n Scan through ' + labels[x_var]+ ' and ' + labels[y_var] )

    if error == True:
        ax.set_zlabel('Error in ' + labels[cont])
        plt.scatter(model.basis[x_var,:],model.basis[y_var,:],np.ones(model.basis[x_var,:].shape[0])*0.3)
    else:
        cont_plot = plt.tricontourf(rav_X, rav_Y, rav_Z, levels=14, cmap=colour, offset = np.amin(rav_Z))

    plt.show()

    cont_plot = plt.tricontourf(rav_X,rav_Y,rav_Z,levels=14, cmap=colour)
    cbar = plt.colorbar(cont_plot,shrink=0.5, aspect=5)
    cbar.set_label(labels[cont])
    plt.xlabel(labels[x_var])
    plt.ylabel(labels[y_var])
    plt.title(
        'Initial state: ' + str(np.round(initial_state, 2)) + '\n Scan through ' + labels[x_var] + ' and ' + labels[
            y_var])

    plt.show()



