import numpy as np
from CartPole import *
import matplotlib.pyplot as plt
import random
from scipy import stats
from Task_1_Functions import *
from Linear_Modeling import *
import pickle

class non_linear_model:
    def __init__(self,m,f=False,policy=False):
        self.type=policy

        if policy==False:
            self.alpha = np.zeros((m,4))
            self.v = 4
            if f:
                self.v = 5
            self.sd = np.zeros((self.v))
        else:
            self.p = np.zeros((m))
            self.v=4
            self.sd=np.zeros((self.v,m))

        self.basis = np.zeros((self.v, m))
        self.m = m

    def transform_x(self,x):
        # if x.ndim == 1:
        #     x=x.reshape((x.shape[0],1)) ## reshapes it to a column vector

        #return kernel_matrix(x, self.m, self.sd, self.basis)[1]
        if x.ndim == 1:
            n = 1
        else: n = x.shape[1]
        Km = np.zeros((self.m,n))
        for i in range(self.m):
            for j in range(n):
                if x.ndim == 1:
                    input=x
                else:
                    input = x[:,j]
                if self.type==False:
                    kernel = gaussian_kernel(input, self.basis[:, i], self.sd)
                else:
                    kernel = gaussian_kernel(input, self.basis[:, i], self.sd[:,i])
                Km[i,j] = kernel

        if Km.ndim == 1:
            Km=Km.reshape((x.shape[0],1)) ## reshapes it to a column vector
        return Km

    def pp(self):
        print(self.m)
        print(self.basis)
        print(self.sd)
        print(self.v)
        if self.type==False:
            print(self.alpha)
        else:
            print(self.p)
    def write_to_file(self,filename):
        with open(filename, 'wb') as f:
            if self.type == False:
                pickle.dump([self.m,self.v,self.alpha,self.basis,self.sd],f)
            else:
                pickle.dump([self.m, self.v, self.p, self.basis, self.sd], f)

    def read_from_file(self,filename):
        with open(filename,'rb') as f:
            if self.type== False:
                self.m,self.v,self.alpha,self.basis,self.sd = pickle.load(f)
            else:
                self.m, self.v, self.p, self.basis, self.sd = pickle.load(f)



def gaussian_kernel(x,x_basis,sd):
    ## takes a single input x and evaluates the kernel function K(X,Xi) for one basis location

    k = (x - x_basis)
    kernel = np.e** ( - 0.5*(np.sin(k[2]/2)/sd[2])**2)
    for i in range(x.shape[0]):
        if i !=2: #not angle
            kernel *= np.e** ( -0.5*(k[i] / sd[i])**2)

    return kernel

def kernel_matrix(x,m,sd,original_basis=None):
    ## builds the kernal matrices Kmm and Kmn
    # each row is a basis location, each col is a data location
    # original basis is for transforming new unseen test data'
    n=x.shape[1]

    Kmm = np.zeros((m,m))
    Kmn = np.zeros ((m,n))
    for j in range(m):  ##  each row
        for i in range(n): ## each col
            if np.all(original_basis) == None:
                kernel = gaussian_kernel(x[:,i],x[:,j],sd)
            else:
                kernel =  gaussian_kernel(x[:,i],original_basis[:,j],sd)

            Kmn[j,i]=kernel
            if i< m:
                Kmm[j,i] = kernel
    return Kmm, Kmn

def get_basis(x,m):
    X = x.T
    #np.random.shuffle(X)## shuffle the inputs
    x=X.T
    return x[:,0:m]

def non_linear_regression(x,y,m,sd,reg,f=False):
    # x is 4xn original points
    # y is 4xn change of state
    # m is number of basis locations
    # sd is 4x1 array of standard deviations for each state  variable
    # reg is the regularisation strength between 1e-6 1e-1

    # function returns mx1 array of alpha
    v=4
    if f:
        v=5
    n = x.shape[1]
    if m > n:
        m = n
    model = non_linear_model(m,f=f)

    model.basis = get_basis(x,m)
    model.sd = sd

    Y = y.T

    Kmm,Kmn = kernel_matrix(x,m,sd,model.basis)
    ## Ca = b

    C = np.matmul(Kmn,Kmn.T)+reg*Kmm ## mxm
    b = np.matmul(Kmn,Y)  ## mx4

    model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

    transformed_x =  Kmn  # make a mxn transformed input
    #transformed_x =model.transform_x(x)

    predicted_y = np.matmul(model.alpha.T,transformed_x)  ## this is a nx4 matrix
    # plt.plot(predicted_y[1,:])
    # plt.show()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.scatter(y[i, :], predicted_y[i, :], marker='x')
        plt.ylabel('Predicted next step')
        plt.xlabel('Actual next step')
        plt.title(labels[i])

        grad, int, r, p, se = stats.linregress(y[i, :], predicted_y[i, :])
        print(labels[i], ' gradient = ', grad)
        print(labels[i], ' inter = ', int)
        print(labels[i], ' r = ', r)
        print(labels[i], ' r^2 = ', r ** 2)
    plt.show()

    return model


def test_regularisation(n, m ,p , t):
    # n training points
    # m basis centeres
    # p evaluation points for regurlasiation
    # t x,y pairs to test on extended extended

    model = non_linear_model(m)

    x, y = get_random_data_pairs(n,quasi=True)
    sd = np.std(x, axis=1)

    model.basis = get_basis(x,m)
    model.sd = sd

    lambdas = np.logspace(-6,-1,p)
    error = np.zeros((4,p))
    Y = y.T
    Kmm, Kmn = kernel_matrix(x, m, sd,model.basis)
    b = np.matmul(Kmn, Y)  ## mx4
    x_test, y_test = get_random_data_pairs(t)   ## extended the search space can cause most transformed x's to be 0 because of the exponential kernel, and only one of the four variabkes needing to be out

    transformed_x = kernel_matrix(x_test,m,sd,model.basis)[1]  #get Kmn for this data

    for i in range(p):
    ## Ca = b
        C = np.matmul(Kmn, Kmn.T) + lambdas[i] * Kmm  ## mxm

        model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

        predicted_y = np.matmul(model.alpha.T, transformed_x)
        # for j in range(4):
        #     plt.subplot(2, 2, j + 1)
        #     plt.subplots_adjust(hspace=0.3)
        #     plt.scatter(y_test[j, :], predicted_y[j, :], marker='x')
        #     plt.ylabel('Predicted next step')
        #     plt.xlabel('Actual next step')
        #     plt.title(labels[j])
        # plt.show()
        e = y_test - predicted_y
        error[:,i] = np.average(np.absolute(e) , axis=1)

    for i in range(4):
        plt.subplot(2 ,2 ,i+1)
        plt.subplots_adjust(hspace=0.3)
        plt.semilogx(lambdas,error[i,:], label = labels[i])
        plt.xlabel(r'Regularisation strength $\lambda$')
        plt.ylabel('Absolute error in' + labels[i])
    plt.title(r'Effect of $\lambda$ with n= ' + str(n) + ', m=' +str(m)+' and '+str(t)+' test points')
    plt.show()


def test_regularisations(n,p , t):
    # n training points
    # p evaluation points for regurlasiation
    # t x,y pairs to test on extended extended

    ms = [100,500,1000]
    mno=3
    x, y = get_random_data_pairs(n, quasi=True)
    sd = np.std(x, axis=1)
    lambdas = np.logspace(-6, -1, p)
    error = np.zeros((4, p,mno))
    x_test, y_test = get_random_data_pairs(t)

    for j in range(mno):
        m=ms[j]
        model = non_linear_model(m)
        model.basis = get_basis(x,m)
        model.sd = sd
        Y = y.T
        Kmm, Kmn = kernel_matrix(x, m, sd,model.basis)
        b = np.matmul(Kmn, Y)  ## mx4
        transformed_x = kernel_matrix(x_test,m,sd,model.basis)[1]  #get Kmn for this data

        for i in range(p):
        ## Ca = b
            C = np.matmul(Kmn, Kmn.T) + lambdas[i] * Kmm  ## mxm

            model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

            predicted_y = np.matmul(model.alpha.T, transformed_x)
            e = y_test - predicted_y
            error[:,i,j] = np.average(np.absolute(e) , axis=1)

    fig = plt.figure()
    labs = [None,None,None]
    for i in range(4):
        if i==3:
            labs = ms
        ax1 = fig.add_subplot(2, 2, i + 1)
        ax2 = ax1.twinx()
        plt.subplots_adjust(hspace=0.3,wspace=0.4)
        for j in range(mno):
            if j==2:
                ax2.semilogx(lambdas, error[i, :, j], label='m=' + str(labs[j]), color = 'G')
            elif j==1:
                ax2.semilogx(lambdas, error[i, :, j], label='m=' + str(labs[j]), color='B')
            else:
                ax1.semilogx(lambdas,error[i,:,j], label = 'm='+ str(labs[j]),color='R')
        ax1.set_xlabel(r'Regularisation strength $\lambda$')
        ax2.set_ylabel('Absolute error in' + labels[i])
        ax1.set_ylabel('Absolute error in' + labels[i] + 'for m=100')
        #ax1.ticklabel_format(axis='y',style='sci')
        #ax1.set_ylim(top=error[i,25,0])
        ax2.set_ylim(top=error[i, 25, 1])
        #ax2.ticklabel_format(axis='y', style='sci')
    fig.suptitle(r'Effect of $\lambda$ with n= ' + str(n) + ', m=' +str(ms) + ' and '+str(t)+' test points')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1+h2, l1+l2,loc='lower center', ncol=3)
    plt.show()

def test_length_scales(n ,p ,t ):
    # n training points

    # p evaluation points for length scale
    # t x,y pairs to test on extended extended
    ms = [100,500,1000]
    mno=3
    regs= [3e-2,4e-4,2e-4]
    x, y = get_random_data_pairs(n, quasi=True)
    sd = np.std(x, axis=1)  ## the middle sd to test
    sd_multip = np.linspace(0.01, 4, p)
    error = np.zeros((4, p,mno))
    x_test, y_test = get_random_data_pairs(t)
    for j in range(mno):
        m=ms[j]
        reg = regs[j]
        model = non_linear_model(m)
        sds= np.zeros((4,p))
        for i in range(4):
            sds[i,:] = sd_multip * sd[i]
        model.basis = get_basis(x,m)
        model.sd = sd
        Y = y.T
        for i in range(p):
        ## Ca = b
            Kmm, Kmn = kernel_matrix(x, m, sds[:,i])
            b = np.matmul(Kmn, Y)  ## mx4
            transformed_x = kernel_matrix(x_test, m, sds[:,i], model.basis)[1]  # get Kmn for this data

            C = np.matmul(Kmn, Kmn.T) + reg * Kmm  ## mxm

            model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

            predicted_y = np.matmul(model.alpha.T, transformed_x)
            e = y_test - predicted_y
            error[:,i,j] = np.average(np.absolute(e) , axis=1)

    fig = plt.figure()
    labs = [None, None, None]
    for i in range(4):
        if i == 3:
            labs = ms
        ax1 = fig.add_subplot(2, 2, i + 1)
        ax2 = ax1.twinx()
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        for j in range(mno):
            if j == 2:
                ax2.plot(sd_multip, error[i, :, j], label='m=' + str(labs[j]), color='G')
            elif j == 1:
                ax2.plot(sd_multip, error[i, :, j], label='m=' + str(labs[j]), color='B')
            else:
                ax1.plot(sd_multip, error[i, :, j], label='m=' + str(labs[j]), color='R')
        ax1.set_xlabel(r'Length Scale multiplier')
        ax2.set_ylabel('Absolute error in' + labels[i])
        ax1.set_ylabel('Absolute error in' + labels[i] + 'for m=100')
        # ax1.ticklabel_format(axis='y',style='sci')
        # ax1.set_ylim(top=error[i,25,0])
        #ax2.set_ylim(top=error[i, 25, 1])
        # ax2.ticklabel_format(axis='y', style='sci')
    fig.suptitle(r'Effect of $\sigma$ with ' + str(n) + ', m=' + str(ms) + ' and ' + str(t) + ' test points')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=3)
    plt.show()

def test_length_scale(n, m ,p ,t , reg):
    # n training points
    # m basis centeres
    # p evaluation points for length scale
    # t x,y pairs to test on extended extended
    # reg is the regularisation

    model = non_linear_model(m)

    x, y = get_random_data_pairs(n,quasi=True)
    sd = np.std(x, axis=1) ## the middle sd to test
    sd_multip = np.linspace(0.01,4,p)

    sds= np.zeros((4,p))
    for i in range(4):
        sds[i,:] = sd_multip * sd[i]

    model.basis = get_basis(x,m)
    model.sd = sd

    error = np.zeros((4,p))
    Y = y.T

    x_test, y_test = get_random_data_pairs(t)   ## extended the search space can cause most transformed x's to be 0 because of the exponential kernel, and only one of the four variabkes needing to be out



    for i in range(p):
    ## Ca = b
        Kmm, Kmn = kernel_matrix(x, m, sds[:,i])
        b = np.matmul(Kmn, Y)  ## mx4
        transformed_x = kernel_matrix(x_test, m, sds[:,i], model.basis)[1]  # get Kmn for this data

        C = np.matmul(Kmn, Kmn.T) + reg * Kmm  ## mxm

        model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

        predicted_y = np.matmul(model.alpha.T, transformed_x)
        # for j in range(4):
        #     plt.subplot(2, 2, j + 1)
        #     plt.subplots_adjust(hspace=0.3)
        #     plt.scatter(y_test[j, :], predicted_y[j, :], marker='x')
        #     plt.ylabel('Predicted next step')
        #     plt.xlabel('Actual next step')
        #     plt.title(labels[j])
        # plt.show()
        e = y_test - predicted_y
        error[:,i] = np.average(np.absolute(e) , axis=1)
    for i in range(4):
        plt.subplot(2 ,2 ,i+1)
        plt.subplots_adjust(hspace=0.3)
        plt.plot(sd_multip,error[i,:], label = labels[i])
        plt.xlabel(r'Length Scale multiplier')
        plt.ylabel('Absolute error in' + labels[i])
    plt.title(r'Effect of $\sigma$ with n= ' + str(n) + ', m=' +str(m)+' and '+str(t)+' test points')
    plt.show()


def test_mn(t ):
    # n training points

    # p evaluation points for length scale
    # t x,y pairs to test on extended extended
    p=9 #9
    n_test=4096
    m_test = 256
    reg = 4e-4
    error = np.zeros((4, p,2))
    #ns = np.linspace(100, 3000, p)
    ns=np.logspace(3,3+p,num=p,base=2,dtype='int16')
    #ms = np.linspace(10, 2000, p)
    ms = np.logspace(3, 3 + p,num=p,base=2,dtype='int16')
    x_test, y_test = get_random_data_pairs(t)
    #get n data
    x, y = get_random_data_pairs(4096, quasi=True)
    for i in range(p):
        #print(ns[i])
        x_train = x[:,:ns[i]]
        sd = np.std(x_train, axis=1)  ## the middle sd to test
        if ns[i]>=m_test:
            m=m_test
        else:
            m=ns[i]
        model = non_linear_model(m)
        model.basis = get_basis(x_train,m)
        model.sd = sd
        Y = y.T

        ## Ca = b
        Kmm, Kmn = kernel_matrix(x, m, model.sd)
        b = np.matmul(Kmn, Y)  ## mx4
        transformed_x = kernel_matrix(x_test, m, model.sd, model.basis)[1]  # get Kmn for this data

        C = np.matmul(Kmn, Kmn.T) + reg * Kmm  ## mxm

        model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

        predicted_y = np.matmul(model.alpha.T, transformed_x)
        e = y_test - predicted_y
        error[:,i,0] = np.average(np.absolute(e) , axis=1)

    #get m data
    for i in range(p):
        sd = np.std(x[:,:n_test], axis=1)  ## the middle sd to test
        m=ms[i]
        model = non_linear_model(m)
        model.basis = get_basis(x[:,:n_test],m)
        model.sd = sd
        Y = y.T

        ## Ca = b
        Kmm, Kmn = kernel_matrix(x[:,:n_test], m, model.sd)
        b = np.matmul(Kmn, Y)  ## mx4
        transformed_x = kernel_matrix(x_test, m, model.sd, model.basis)[1]  # get Kmn for this data

        C = np.matmul(Kmn, Kmn.T) + reg * Kmm  ## mxm

        model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

        predicted_y = np.matmul(model.alpha.T, transformed_x)
        e = y_test - predicted_y
        error[:,i,1] = np.average(np.absolute(e) , axis=1)

    fig = plt.figure()
    labs = [None, None]
    for i in range(4):
        if i == 3:
            labs = ['Vary n','Vary m']
        ax1 = fig.add_subplot(2, 2, i + 1)
        #ax2 = ax1.twinx()
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        for j in range(2):
            if j == 0:
                ax1.plot(np.log2(ns), error[i, :, j], label= str(labs[j]), color='G')
            elif j == 1:
                ax1.plot(np.log2(ms), error[i, :, j], label=str(labs[j]), color='B')
        ax1.set_xlabel(r'log(m) or log(n)')
        #ax2.set_ylabel('Absolute error in' + labels[i])
        ax1.set_ylabel('Absolute error in' + labels[i])
        # ax1.ticklabel_format(axis='y',style='sci')
        # ax1.set_ylim(top=error[i,25,0])
        #ax2.set_ylim(top=error[i, 25, 1])
        # ax2.ticklabel_format(axis='y', style='sci')
    fig.suptitle(r'Effect of varying m and n with n= ' + str(n_test) + ', m=' + str(m_test) + ' and $\lambda$ =' + str(reg))
    h1, l1 = ax1.get_legend_handles_labels()
    #h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1,l1,loc='lower center', ncol=2)
    #fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=3)
    plt.show()

def test_model(c,n,visual=False,loop = False, osc = False, stable_equ=False, model = 0,f=False):
    # c is 4x4 linear coefficent matrix, or contais alpha and basis location for non linear model
    #  n is number of time iterations to predict
    # stable_equ if motion is about theta = pi
    # plots real state trajectory and linear model predictions
    # model = 0 if linear model
    #modoel = 1 if non linear model
    v= 4
    if f:
        v=5
    system = CartPole(visual)
    initial_state = np.zeros(v)
    initial_state = random_init(initial_state,stable_equ,f=f)
    initial_state[0] = 0
    if loop == True:
        initial_state = np.array([0,0,np.pi,15])
    elif osc == True:
        initial_state = np.array([0, 0, np.pi, 5])


    force = 0
    if f:
        force=initial_state[4]
    system.setState(initial_state[0:4])

    state_history = np.empty((v, n))
    state_history[:, 0] = initial_state[0:v]
    for i in range(n-1):  # update dynamics n times
        system.performAction(force)  ## runs for 0.1 secs with no force
        system.remap_angle()
        state_history[0:4,i + 1] = system.getState()
        if f:
            state_history[4]=force
    #time = np.arange(0, (n ) * 0.1, 0.1)
    prediction = state_prediction(c,n,initial_state,model,f=f)

    for i in range(4):
        plt.subplot(2 ,2 ,i+1)
        plt.subplots_adjust(hspace=0.3)
        plt.plot( state_history[i,:],linestyle=':' ,label = 'Actual')
        plt.plot(prediction[i, :], label='Predicted')
        plt.xlabel('Iteration')
        plt.title(labels[i])
        plt.legend(loc='upper right')
        if i != 2:
            plt.ylim = (np.min(state_history[i,:])-0.5*np.min(state_history[i,:]), np.max(state_history[i,:])+0.5*np.max(state_history[i,:])  )

        # if i !=0 and i!=2:
        #     plt.ylim(-20, 20)

    plt.show()

def state_prediction(c,n,initial,model=0,f=False):
    # c is 4x4 linear coefficent matrix, n is number of time interations, initial is 4x1 initial state
    # returns predicted trajectory from linear model
    v=4
    if f:
        v=5
    prediction = np.zeros((v,n))
    prediction[:,0] = initial
    for i in range(n-1):
        change = np.zeros((1,v))
        if model == 0: #linear model
            change[0,0:4]=np.matmul(c,prediction[:,i])
            prediction[:,i+1] = prediction[:,i] + change
        else:
            change[0, 0:4] = np.matmul(c.alpha.T, c.transform_x(prediction[:,i])).T
            prediction[:, i + 1] = prediction[:, i] + change
        theta = remap_angle(prediction[2,i+1])
        prediction[2,i+1] = theta
    return prediction


def test_noise(p ,t ,correct=False,f=False):
    # n training points
    # m basis centeres
    # p evaluation points for length scale
    # t x,y pairs to test on extended extended
    # reg is the regularisation

    m=1000
    n=2000
    reg = 2e-4
    model = non_linear_model(m)
    noise = np.linspace(0., 5., p)
    error = np.zeros((4,p))
    fig=plt.figure()
    x_test, y_test = get_random_data_pairs(t,f=f)  #test on same points
    for i in range(p):
        print(i)
        x, y = get_random_data_pairs(n,quasi=True,noise=noise[i],f=f)
        sd = np.std(x, axis=1) ## the middle sd to test

        model.basis = get_basis(x,m)
        model.sd = sd


        Y = y.T

         ## extended the search space can cause most transformed x's to be 0 because of the exponential kernel, and only one of the four variabkes needing to be out

    ## Ca = b
        Kmm, Kmn = kernel_matrix(x, m, model.sd)
        b = np.matmul(Kmn, Y)  ## mx4
        transformed_x = kernel_matrix(x_test, m, model.sd, model.basis)[1]  # get Kmn for this data

        C = np.matmul(Kmn, Kmn.T) + reg * Kmm  ## mxm

        model.alpha = np.linalg.lstsq(C, b, rcond=None)[0]

        predicted_y = np.matmul(model.alpha.T, transformed_x)
        # for j in range(4):
        #     plt.subplot(2, 2, j + 1)
        #     plt.subplots_adjust(hspace=0.3)
        #     plt.scatter(y_test[j, :], predicted_y[j, :], marker='x')
        #     plt.ylabel('Predicted next step')
        #     plt.xlabel('Actual next step')
        #     plt.title(labels[j])
        # plt.show()
        e = y_test - predicted_y
        error[:,i] = np.average(np.absolute(e) , axis=1)
    for i in range(4):
        plt.subplot(2 ,2 ,i+1)
        plt.subplots_adjust(hspace=0.3)
        plt.plot(noise,error[i,:], label = labels[i])
        plt.xlabel(r'Mearsurment noise $\sigma$')
        plt.ylabel('Absolute error in ' + labels[i])
    fig.suptitle(r'Effect of noise with n= ' + str(n) + ', m=' +str(m)+' and '+str(t)+' test points')
    plt.show()

    with open('noisedata', 'wb') as f:
        pickle.dump([error, noise], f)