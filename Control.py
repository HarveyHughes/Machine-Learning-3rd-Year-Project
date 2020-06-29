import autograd.numpy as np
from autograd import grad
from CartPole import *
import matplotlib.pyplot as plt
import random
from scipy import stats
from Task_1_Functions import *
from Linear_Modeling import *
from Non_linear_modelling import *
import random
from scipy import optimize


p_range = [30.,30.,30.,30.]
training_model = None #model



def loss_trajectory(initial,n,p,model=None,visual=False,nlp=None):
    loss = 0
    if model == None:
        system = CartPole(visual)
        system.setState(initial)
        loss += system.loss()
    else:
        loss += loss_pos(initial)
        x = initial[0:4]
        x_extended =np.zeros(5)

    for i in range(n):
        if model == None:  #using model dynamics for initial scans
            x = system.getState()
            force = np.matmul(p,x)
            system.performAction(force)
            system.remap_angle()
            loss += system.loss()
        else:  # using non linear modeelling for training
            if nlp==None:
                force = np.matmul(p, x)
            else:
                force = np.matmul(p,nlp.transform_x(x))
            x_extended[0:4] = x
            x_extended[4] = force
            change = np.matmul(model.alpha.T, model.transform_x(x_extended)).T
            x +=change
            x[2] = remap_angle(x[2])
            loss += loss_pos(x)
    return loss



def scan_policy(pos_1,pos_2=None,n=20,points=30, model=None,top=False,bot=False,nlp=None):

    x0= np.zeros(4)
    x0 = random_init(x0)
    x0[0]= 0

    if top==True:
        x0=np.array([random.normalvariate(0,0.03),random.normalvariate(0,0.03),random.normalvariate(0,0.03),random.normalvariate(0,0.03)])
    if bot:
        x0 = np.array([random.normalvariate(0, 0.03), random.normalvariate(0, 0.03), np.pi - random.normalvariate(0, 0.03),
                       random.normalvariate(0, 0.03)])

    p=np.zeros(4)

    # p[1]=5
    # p[3]=-8

    loop=False
    osc=False

    if loop == True:
        x0 = np.array([0,0,np.pi,15])
    elif osc == True:
        x0 = np.array([0, 0, np.pi, 5])

    scanned_x = np.arange(- p_range[pos_1] / 2, p_range[pos_1] / 2, p_range[pos_1] / points)  ##scanning variable 1yield


    if pos_2 != None: ## 2d contour plots
        scanned_y = np.arange(- p_range[pos_2] / 2, p_range[pos_2] / 2, p_range[pos_2] / points)  ##scanning variable 1yield
        X, Y = np.meshgrid(scanned_x, scanned_y)
        rav_X = np.ravel(X)
        rav_Y = np.ravel(Y)
        rav_Z = np.zeros(rav_X.shape[0])
        for i in range(rav_X.shape[0]):
            p[pos_1] = rav_X[i]
            p[pos_2] = rav_Y[i]

            rav_Z[i] = loss_trajectory(x0,n,p,model,nlp)

        Z = rav_Z.reshape(X.shape)

        colour = 'inferno'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colour,vmin=0, vmax=50)
        # Add a color bar which maps values to colors.
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_label('Loss')
        plt.xlabel('p[' + str(pos_1) +']')
        plt.ylabel('p[' + str(pos_2) +']')
        ax.set_zlabel('Loss')
        plt.title(
            'Initial state: ' + str(np.round(x0, 2)) + '\n Scan through ' + 'p[' + str(pos_1) +'] and p[' + str(pos_2) +']')
        plt.show()

        cont_plot = plt.tricontourf(rav_X, rav_Y, rav_Z, levels=14, cmap=colour,vmin=0, vmax=50 )
        cbar = plt.colorbar(cont_plot, shrink=0.5, aspect=5)
        cbar.set_label('Loss')
        plt.xlabel('p[' + str(pos_1) +']')
        plt.ylabel('p[' + str(pos_2) +']')
        plt.title(
            'Initial state: ' + str(np.round(x0, 2)) + '\n Scan through ' + 'p[' + str(pos_1) + '] and p[' + str(
                pos_2) + ']')

        plt.show()

def loss_function(p):
    model = non_linear_model(10, f=True)
    #model.read_from_file('m=500n=10000')
    model.read_from_file('m=1000n=10000')

    loss = 0
    #initial =np.array([random.normalvariate(0,0.03),random.normalvariate(0,0.03),random.normalvariate(0,0.03),random.normalvariate(0,0.03)]) #+ random.normalvariate(0,0.01)
    initial=np.array([0,0.2,0.05,-0.2])#*10
    n = 20
    loss =loss + loss_pos(initial)
    x=np.zeros((4,1))
    x[:,0] = initial[0:4]
    x_extended = np.zeros((5))

    system = CartPole(visual=False)
    system.setState(initial)

    for i in range(n):
       # # using non linear modeelling for training
       change = np.zeros((1,4))
       if isinstance(p, (list, tuple, np.ndarray)):
           force = np.matmul(p,x)
       else:
           force = np.matmul(p._value, x)

       x_extended[0:4] = x[:,0]
       x_extended[4] = force
       test = np.matmul(model.alpha.T, model.transform_x(x_extended))
       x =x + test



       # x = system.getState()
       #
       # if isinstance(p, (list, tuple, np.ndarray)):
       #       force = np.matmul(p,x)
       # else:
       #       force = np.matmul(p._value, x)
       #
       # #force = np.matmul(p, x)
       # system.performAction(force)
       # x= system.getState()

       x[2] = remap_angle(x[2])
       loss += loss_pos(x)
    return loss



def compute_gradient(p):
    dp = 0.01

    gradient = np.zeros((p.shape))

    loss_original = loss_function(p)
    p = p + dp
    for i in range(p.shape[0]):
        p_new = p.copy()
        p_new[i] = p[i]+dp
        gradient[i] = (loss_function(p_new) - loss_original)/dp
    return gradient
def grad_decent(n,model,p):

    training_model = model
    #loss_function_grad = grad(loss_function)
    a=0.01
    m = 0.01
    #print(loss_function(p))
    loss_history = np.zeros((n))
    grad_old=np.zeros(p.shape)
    for i in range(n): #do n loops of gradient decent
        #gradient = loss_function_grad(p)
        gradient = compute_gradient(p)
        #print(gradient)
        #p = p - a* gradient
        grad_old = a*gradient
        p-= a* gradient +grad_old*m
        loss_history[i] = loss_function(p)

    plt.plot(loss_history)
    plt.show()

    print(p)
    print(loss_function(p))
    return p

def test_policy(p,n,visible=False,sd=0.03,bottom=False,nlm=None,Top=False):
    system = CartPole(visual=visible)
    #system.setState(np.array([0,0.01,-0.01,-0.01]))
    system.setState(np.array([random.normalvariate(0,sd),random.normalvariate(0,sd),random.normalvariate(0,sd),random.normalvariate(0,sd)]))
    if nlm != None and Top==False: bottom =True
    if bottom:
        system.setState(np.array([0,0,np.pi,0]))
        #sd=1
        #system.setState(np.array([random.normalvariate(0, sd), random.normalvariate(0, sd), np.pi ,random.normalvariate(0, sd)]))
        #system.setState(np.array([-5.5,-3.36,-0.56,0.42]))

    state_history = np.empty((n,4))
    #system.setState(np.array([0, 0, 0,5.1]))
    #system.setState(np.array([0, 0.2, 0.05, -0.2])*5)
    # system.setState(np.array([0, 0.2, 1, -4]) ) #get it to 1 rad at -4rads^1 and itll settle
    for i in range(n):
        x = system.getState()
        if x.ndim == 1:
            state_history[i,:]=x
        else:
            state_history[i, :] = x.reshape((4))

        if nlm==None:
            force = np.matmul(p, x)
        else:
            x_ext = nlm.transform_x(x)
            force = np.matmul(p, x_ext)
            #state_history[i,2]=force

        system.performAction(force)
        system.remap_angle()

    plt.plot(state_history[:,2],label=r'$\theta$')
    plt.plot(state_history[:, 1],label='v')
    plt.plot(state_history[:, 3],label=r'$\dot{\theta}$')
    plt.plot(state_history[:, 0], label='x')
    plt.xlabel('Iteration')
    #plt.ylabel(r'$\theta$')
    plt.legend(loc='lower left')
    plt.title('P=' + str(p))
    #plt.ylim([-0.2,0.2])
    plt.show()


def non_linear_loss_function(p):
    model = non_linear_model(10, f=True)
    #model.read_from_file('m=500n=10000')
    model.read_from_file('m=1000n=10000')

    policy = non_linear_model(10,policy=True)
    #policy.read_from_file('nonzero')  ### change at somepoint so that sd's are part of the p input?
    #policy.read_from_file('justtop')
    policy.read_from_file('wholetest')
   # policy.read_from_file('flickup')

    #different optmisation situations
    optimise_loc = False
    sym = False
    top_constant = False
    bot_constant = False
    bounce_back = True

    if optimise_loc == True:
        m = int((p.shape[0]-2)/4) #policy.m
        #policy = non_linear_model(m,policy=True)
        #policy.sd = np.ones((4,m))
        policy.basis[:, 0] = p[: 4].T
        policy.basis[:, 1] = -policy.basis[:, 0]
        policy.basis[:, 2] = p[4:4 + 4].T
        policy.basis[:, 3] = -policy.basis[:, 2]
        p = [-27.669,27.669,-10.98,10.98]
        policy.sd[0, :] = [30, 30, 30, 30]
        policy.sd[1, :] = [30, 30, 30, 30]

    if sym:
        policy.basis[:,0] = p[2:6].T
        policy.basis[:,1]=-policy.basis[:,0]
        policy.basis[:, 2] = p[6:10].T
        policy.basis[:, 3] = -policy.basis[:, 2]
        #print(policy.basis)
        p = [p[0],-p[0],p[1],-p[1]]

    if top_constant:
        p=np.array([-27.669,27.669,-10.98,10.98,p[0],p[1],0,p[2]]) #for updating 3 on lhs

    if bot_constant:
        p = [p[0], -p[0], p[1], -p[1],76, -77,0, -37.025]

    if bounce_back:
        p=np.array([-27.669,27.669,-10.98,10.98,76,-77,p[0],-37.025,-p[0],p[1],-p[1]])


    loss = 0
    #initial=np.array([0,0,np.pi,0])
    #initial = np.array([0, 0.2, 0.05, -0.2])*5
    initial = np.array([-5.5,-3.36,-0.56,0.42])
    n = 30
    loss =loss + loss_pos(initial)
    x=np.zeros((4,1))
    x[:,0] = initial[0:4]
    x_extended = np.zeros((5))

    system = CartPole(visual=False)
    system.setState(initial)

    for i in range(n):
       # # using non linear modeelling for training
       change = np.zeros((1,4))
       x_policy = policy.transform_x(x)
       if isinstance(p, (list, tuple, np.ndarray)):
           force = np.matmul(p,x_policy)
       else:
           force = np.matmul(p._value, x_policy)

       x_extended[0:4] = x[:,0]
       x_extended[4] = force
       test = np.matmul(model.alpha.T, model.transform_x(x_extended))
       x =x + test
       x[2] = remap_angle(x[2])

       # x = system.getState()
       # x_policy = policy.transform_x(x)
       # if isinstance(p, (list, tuple, np.ndarray)):
       #       force = np.matmul(p,x_policy)
       # else:
       #       force = np.matmul(p._value, x_policy)
       #
       # #force = np.matmul(p, x)
       # system.performAction(force)
       # x= system.getState()


       loss += loss_pos(x)
    return loss





def generate_non_linear_policy(m):


    # m=11

    #m=10
    #m=4
    m=11
    #m=8
    nlp = non_linear_model(m,policy=True)

    basis = np.zeros((4,m)) #no force nessary
    sd = np.zeros((4,m))

    #top section
    # to slow down
    basis[:, 0] = [0, 0, 0.25, 2.5]  # this is the overall aim
    #basis[:,0]=[-0.00988998, -0.03803768, 0.57465364, 2.65458026]
    sd[:, 0] = [30, 30, .25, 2.5]  # small sd     #change velocity sd to huge, and x to hughe
    basis[:, 1] = [0, 0, -0.25, -2.5]  # this is the overall aim
    #basis[:,1]=[-0.00988998, -0.03803768, 0.57465364, 2.65458026]
    sd[:, 1] = [30, 30, .25, 2.5]  # small sd

    # to push back
    basis[:, 2] = [0, 0, 0.25, -2.5]  # this is the overall aim
    #basis[:,2]=[0.00525632, 0.09308213, 0.88202638, -1.32182546]
    sd[:, 2] = [30, 30, .25, 2.5]  # small sd
    basis[:, 3] = [0, 0, -0.25, 2.5]  # this is the overall aim
    sd[:, 3] = [30, 30, .25, 2.5]  # small sd
    #basis[:,3]=[0.00525632, 0.09308213, 0.88202638, -1.32182546]


    #bottom  , try and increase oscilation height
    basis[:, 4] = [0, 0, np.pi - 0.3, -1.5]  # in start position still # generates the initial force
    sd[:, 4] = [7., 7., .5, 5.]

    basis[:, 5] = [0, 0, -np.pi + 0.3, 1.5]  # in start position still # generates the initial force
    sd[:, 5] = [7., 7., .5, 5.]

    # #just over a quarter
    # basis[:, 6] = [0, -2, np.pi/2 - 0.05, -5]  # in start position still # generates the initial force
    # sd[:, 6] = [2., 3., 0.3, 3.]#small angle sd as its like an impulse
    #  #not needed


    #push it back (from -ve x)
    basis[:, 6] = [-6, 0.25, 0, 0]  # in start position still # generates the initial force
    sd[:, 6] = [.2, .5, 0.2, 3.]  # small angle sd as its like an impulse


    #2/3 round
    basis[:, 7] = [0, -1, 1.2, -10]
    sd[:, 7] = [5., 5., 0.2, 8]

    # push it back (from +ve x)
    basis[:, 8] = [6, -0.25, 0, 0]  # in start position still # generates the initial force
    sd[:, 8] = [.2, .5, 0.2, 3.]  # small angle sd as its like an impulse

    # slow down  (from +ve x)
    basis[:, 9] = [-0.5, -5, 0, 0]  # in start position still # generates the initial force
    sd[:, 9] = [1., 3, 0.2, 3.]  # small angle sd as its like an impulse

    # slow dow (from -ve x)
    basis[:, 10] = [0.5, 5, 0, 0]  # in start position still # generates the initial force
    sd[:, 10] = [1., 3, 0.2, 3.]  # small angle sd as its like an impulse









    #try with less about 0 velocity
    #bot
    # basis[:, 0] = [0, 0, np.pi-0.3, 0]  # in start position still # generates the initial force
    # sd[:, 0] = [3., 3., .75, 3.]
    # basis[:, 1] = [0, 0, -np.pi + 0.3, 0]  # in start position still # generates the initial force
    # sd[:, 1] = [3., 3., .75, 3.]
    #
    # #part of the way from bottom , force needs to change direction
    # basis[:, 2] = [0, 2.5, 2, -4]  # in start position still # generates the initial force
    # sd[:, 2] = [2., 3., 1, 2.]
    # basis[:, 3] = [0, -2.5, -2, 4]  # in start position still # generates the initial force
    # sd[:, 3] = [2., 3., 1, 2.]
    #
    # # ## a third
    # basis[:, 4] = [0, -1, 1.2, -2]
    # sd[:, 4] = [5., 5., 1, 2.]
    # basis[:, 5] = [0, 1, -1.2, 2]
    # sd[:, 5] = [5., 5., 1, 2.]

    # #top keep it between these two
    # basis[:,0] = [0,0.5,0.25,0.5]   #this is the overall aim
    # sd[:,0] = [3,.5,.25,.5]  # small sd
    # basis[:, 1] = [0, -0.5, -0.25, -0.5]  # this is the overall aim
    # sd[:, 1] = [3, .5, .25, .5]  # small sd




    # #changed loc
    # basis[:, 0] = [0, 0.42, 0.3, 0.43]  # this is the overall aim
    # sd[:, 0] = [3, .5, .25, .5]  # small sd
    # basis[:, 1] = [0, -0.42, -0.3, -0.43]  # this is the overall aim
    # sd[:, 1] = [3, .5, .25, .5]  # small sd


    # #top push inwards or slow down?
    # basis[:, 8] = [0, -1, 0.5, -1.5]  # this is the overall aim
    # sd[:, 8] = [3, 3, .4, 1]  # small sd
    # basis[:, 9] = [0, 1, -0.5, 1.5]  # this is the overall aim
    # sd[:, 9] = [3, 3, .4, 1]  # small sd


    #most are around 0 velocity etc to try and limit the number of points, instead the sd is being increased

    # #top
    # basis[:,0] = [0,0,0.25,0]   #this is the overall aim
    # sd[:,0] = [3,.2,.2,.2]  # small sd
    # basis[:, 1] = [0, 0, -0.25, 0]  # this is the overall aim
    # sd[:, 1] = [3, .2, .2, .2]  # small sd
    #
    # # basis[:, 1] = [0, 0, 0, 0]  # in correct posiiton but moving therefore wider sd
    # # sd[:, 1] = [3., 5., 1., 5.]  # larger sd, more sd on force    # maybe move to be a few located around the centre?
    # # basis[:, 1] = [0, 0, 0, 0]
    # # sd[:, 1] = [3., 5., 1., 5.]
    #
    # #bot
    # basis[:, 2] = [0, 0, np.pi-0.3, 0]  # in start position still # generates the initial force
    # sd[:, 2] = [3., 3., .3, 3.]
    # basis[:, 3] = [0, 0, -np.pi + 0.3, 0]  # in start position still # generates the initial force
    # sd[:, 3] = [3., 3., .3, 3.]
    #
    # #bot and moving
    # basis[:, 4] = [0, 0, np.pi, 0]  # at bottom hopefully to push the thing back round
    # sd[:, 4] = [5., 7., 1., 7.] # wider sd
    #
    #
    # #auarter of the way round, only one becayse its not here for long
    # basis[:, 5] = [0, 0, np.pi/2, 0]
    # sd[:, 5] = [5., 5., 1., 5.]
    # basis[:, 6] = [0, 0, -np.pi / 2, 0]
    # sd[:, 6] = [5., 5., 1., 5.]
    #
    # ## a quarter of the way from the top
    # basis[:, 7] = [0, 0, np.pi / 4, 0]
    # sd[:, 7] = [5., 5., 0.5, 5.]
    # basis[:, 8] = [0, 0, -np.pi / 4, 0]
    # sd[:, 8] = [5., 5., 0.5, 5.]
    #
    # ## a quarter of the way from the bot
    # basis[:, 9] = [0, 0, 3*np.pi / 4, 0]
    # sd[:, 9] = [5., 5., 0.5, 5.]
    # basis[:, 10] = [0, 0, -3*np.pi / 4, 0]
    # sd[:, 10] = [5., 5., 0.5, 5.]



    nlp.basis = basis#5xm
    #
    nlp.sd = sd#5xm

    nlp.write_to_file('wholetest')
    #nlp.write_to_file('flickup')


    plot_basis_locations(basis,sd)

    return nlp


def plot_basis_locations(basis,sd):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('v')
    plt.ylabel(r'$\dot{\theta}$')
    ax.set_zlabel(r'$\theta$')
    ax.set_zlim([-np.pi,np.pi])

    m = basis.shape[1]
    for i in range(m): # for each basis location
        phi = np.linspace(0, 2 * np.pi, 256).reshape(256, 1)  # the angle of the projection in the xy-plane
        theta = np.linspace(0, np.pi, 256).reshape(-1, 256)  # the angle from the polar axis, ie the polar angle

        # Transformation formulae for a spherical coordinate system.
        x = sd[1,i] * np.sin(theta) * np.cos(phi) - basis[1,i]
        y = sd[3,i] * np.sin(theta) * np.sin(phi) - basis[3,i]
        z = sd[2,i] * np.cos(theta) - basis[2,i]

        # fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
        # ax = fig.add_subplot(111, projection='3d')
        if i!= 10 and i!= 9 and i!= 8 and i!= 6:
            ax.plot_surface(x, y, z,alpha=0.5)
    plt.xlabel(r'$\dot{x}$')
    ax.set_zlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    plt.title('First 7 basis locations and length scales')
    plt.show()