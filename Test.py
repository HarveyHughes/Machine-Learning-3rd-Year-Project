import numpy as np
from CartPole import *
import matplotlib.pyplot as plt
import random
from Task_1_Functions import *
from Linear_Modeling import *
from Non_linear_modelling import *
from Control import *
from scipy import optimize

###task 1
##full loop
# rollout([0,14],50,False)
#
# ##oscilations
# rollout([0,5],50,False)


# ##task 1.22
# for i in range(4):
#     scan_step(i,100)

# scan_all(100,1)  #plots change in state
# scan_all(100,0) #plots next state1)

#contour_plot(50,2,3,1)

##task 2
# x,y =get_random_data_pairs(1000,f=True)
# c = linear_regression(x,y,f=True)
# # scan_all(100,1,False,c)
# print(c)
# test_model(c,100,f=True)
f=True
##non linear regression
# x,y =get_random_data_pairs(10000,quasi=True,f=f)
#x1,y1 = get_random_data_pairs(1000)
# plt.scatter(x[0,:],x[1,:])
# plt.scatter(x1[0,:],x1[1,:])
# plt.show()
# plt.scatter(x[2,:],x[3,:])
# plt.scatter(x1[2,:],x1[3,:])
# plt.show()
# sd = np.std(x, axis = 1)
#model = non_linear_regression(x,y,100,sd*1.75,3e-2,f=f)
# model = non_linear_regression(x,y,500,sd*1.1,4e-4,f=f)
# model.write_to_file('m=500n=10000')
# model = non_linear_regression(x,y,1000,sd,2e-4,f=f)
# model.write_to_file('m=1000n=1000noforce')
# sd = np.std(x, axis = 1)
# model = non_linear_regression(x,y,5,sd,4e-4,f=f)
# model.write_to_file('test_new_class')
# model.pp()
#
#scan_all(100,1,False,model,nlm=True)


# model = non_linear_model(10,f=f)
# model.read_from_file('m=1000n=10000')
#scan_all(100,1,False,model,nlm=True,f=True)
# test_model(model,100,f=f,model=1)

# transformed_x = model.transform_x(x)
# print(transformed_x.shape)

# a = non_linear_regression(x,y,750,sd,1e-6)
# a = non_linear_regression(x,y,750,sd,0)
#test_regularisation(1000,500,30,500)  # for m=100 3e-2 , for m =500 4e-4 , m = 1000 2e-4             # if its a good model then lambda makes little difference
#test_regularisations(1000,30,500)
#test_length_scales(1000,30,500)
# test_length_scale(1000,100,30,500, 3e-2)   #1.75
# test_length_scale(1000,500,30,500, 1e-3)   ##1.1
#test_length_scale(1000,1000,10,500, 2e-4)   ## 1 is best

# contour_plot(50,1,2,3,model = model, error = True,f=f)
# contour_plot(50,2,3,1,model = model, error = True,f=f)
#
#
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)

#test_mn(500)
#scan_policy(0,1,points=50,top=True)
# scan_policy(2,3,points=50,top=True)
#scan_policy(0,2,points=50,top=True)
# scan_policy(0,3,points=50,top=True)
# scan_policy(1,2,points=50,top=True)
# scan_policy(1,3,points=50,top=True)
# scan_policy(2,3,points=50,top=True)

#p = grad_decent(20,model,np.array([-5. , 5., 10., -8.]))


#trough that extends forever
#[-2.9 , 2.9, -5, -7.5]





#try and force it to oscillate
#[-1.1293577   1.18508497 -4.44289383 -6.28098848]
#[-1.10047104  1.03441207 -5.19856927 -5.39793644]
# up the penalyty for being slightly off centre



#[5,0,0,-5]
#[ 4.25288791 -0.93788629 -0.53291293 -5.62347163]
#[ 4.05333611 -1.16506632 -0.66591379 -5.7730857 ]
#[ 3.69363813 -1.54454497 -0.92180985 -6.04883493]
#[ 2.7255605  -1.2124378  -1.86443561 -6.47899126]
#[ 2.09098295 -0.8883427  -2.38997734 -6.60517805]
#[ 0.46086604  0.08204547 -3.5850825  -6.78087206]
#[-0.45145868  0.70110939 -4.08845974 -6.68875464]
#[-1.1293577   1.18508497 -4.44289383 -6.28098848]



#training on random start
#[-0.48,-0.74,-6,-5.8])
#[-0.36698448 -0.50132806 -6.14096113 -5.84895659]
#[-0.22131246 -0.28809543 -5.93875405 -5.69678041]

#[ 0.05438369  0.25645456 -5.68277981 -5.43769725]


# [-0.2330697   0.53157727 -1.04678576 -3.97326816]
# [-0.1803402   0.51218414 -0.95866025 -3.91482031]
#test_policy(p,50)


#[-0.43582614  0.71224902 -3.07496319 -5.76620956]
#[-0.4805147   0.74308739 -3.11652141 -5.8005056 ]


#on actual function
#[-0.42695853  0.83341675 -1.48886871 -4.5490731 ]
#[-0.62433402  0.91229899 -1.74282501 -4.84296058]
#[-0.83964223  1.01560017 -1.99773593 -5.12171169]
#[-0.99351887  1.09727579 -2.1612305  -5.29091299]
#[-1.10211907  1.15713825 -2.28483694 -5.40929849]
#[-1.35252253  1.30237887 -2.57434577 -5.67442529]
#[-1.43196324  1.34941848 -2.73536873 -5.80675097]
#[-1.45618002  1.36338369 -2.84721507 -5.89223597]
#[-1.46048033  1.36472933 -2.93289326 -5.94044096]    no furter improvment , try new point , needs to be larger as this one never changes direction



#[ 4.39955929 25.7004174   1.12594272  7.04127632]
#[ 6.03126329 23.94430181  2.96117616 12.94970198]
#[13.24146069 25.67375723 10.51650612 30.68875856]
#[16.59400259 27.87177418 14.15819829 38.21578741]

# res= optimize.minimize(loss_function,x0=np.array([-1.,1.,-3.,-6.]) ,method='Nelder-Mead')
# print(res)
# test_policy(res.x,50)


# res= optimize.minimize(loss_function,x0=np.array([0.08818345,  -0.2735744 , -31.36871136,  -4.65684842]) ,method='Nelder-Mead')
# res= optimize.minimize(loss_function,x0=np.array([ -1.79225088,   1.36575015, -38.37208003,  -2.86461199]) ,method='Nelder-Mead')
# print(res)
# test_policy(res.x,50)
# test_policy(res.x,50,True)


#good one for mormal running, suffers from shooting off for ages, but it can get the pendulum upright from the bottom
# p = np.array([0.08818345,  -0.2735744 , -31.36871136,  -4.65684842])
# test_policy(p,50,bottom=True)
# test_policy(p,50)
# test_policy(p,50)
# test_policy(p,50)
# test_policy(p,50)
# test_policy(p,50)
# test_policy(p,50)

## from [-1.,1.,-3.,-6.]  started here because it was the point that made the oendulum very still at the start
#to [  0.08818345,  -0.2735744 , -31.36871136,  -4.65684842]


##talk about sensitivity to loss function
#all extra penalty ones generated about the good one from before [0.08818345,  -0.2735744 , -31.36871136,  -4.65684842]


# #adding in x penalty abs(x)/5

# p=np.array([ 23.4712012 , -12.56236927, -23.16326871,   2.70409019])
# test_policy(p,50,True)


# p=np.array([ 0.3033434 ,  1.30803855,  1.26010285, -6.8872815 ])
# test_policy(p,50,True)
#v penalty
# #linear  abs(v)/5
# p=np.array([  1.01139182,  -9.3456029 , -25.64859931,   1.68480808])
# test_policy(p,50,True)
#still shoots off

#squared
# p=np.array([ -1.92506945,  -7.38263384, -30.21323364,   0.95301129])
# test_policy(p,50,True)  ## itinially too strong as loss_v = (1.0-loss_angle) * x[1]**2 * loss_velocity_scale its using the pendulum swinging round to keep it stationary

#try 10* smaller :
#p=np.array([ -14.13264246,   13.19196525, -138.72905826,  -14.65152857])
#test_policy(p,50,True)
#managed to keep it oscilationg for a while then shoots off

#try weaker still : 20*
#p=np.array([ -14.97029792,   15.01466064, -156.24642981,  -15.8959439 ])
#test_policy(p,50,True)
#is better but oscilates very very fast ,

#try stronger limit on the angular velocity (three times)
# p=np.array([ -1.79225088,   1.36575015, -38.37208003,  -2.86461199])
# test_policy(p,50,False)
###whoooooooooop gets it very still and stable for points near to the start
# test_policy(p,50,True,sd=1)
# test_policy(p,50,True,bottom=True)
# test_policy(p,50,True,sd=1)
# test_policy(p,50,True,sd=1)
# ## doesnt for for sd= 1 or from the bottom
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
#0.5 is ok but still difts off slowly away from x=0 at about 30 timesteps


#try running for 30 timesteps in the loss function instead of 20
#p=np.array([  -6.39707597,   11.58085495, -127.68793243,  -12.44826855])


# try model where m =1000, start from where prev model ended,
#p=np.array([  0.30355566,  14.7607993 , -91.53447232, -12.62519575])
#test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,bottom=True)
# test_policy(p,50,True,sd=1)
# test_policy(p,50,True,sd=1)
## doesnt for for sd= 1 or from the bottom
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
#0.5ok most of the time

#try training on point further away by 10* (trained on [0.08818345,  -0.2735744 , -31.36871136,  -4.65684842])
# p=np.array([  0.14286246,  -0.16538587, -11.14879924,  -3.11638685])
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,sd=0.5)
# test_policy(p,50,True,bottom=True)
# test_policy(p,50,True,sd=1)
# test_policy(p,50,True,sd=1)
# can get it up from the bottom but shotts off,
#other ones appear to shooot  off

# train on # p=np.array([ -1.79225088,   1.36575015, -38.37208003,  -2.86461199]) instead
# p=np.array([  4.2623316 ,  10.23557599, -80.79528531, -10.03240233])#didnt finish
# test_policy(p,50,False,sd=0.75)
# test_policy(p,50,True,sd=0.75)
# test_policy(p,50,False,sd=0.75)
# test_policy(p,50,False,sd=0.75)
# test_policy(p,50,False,sd=0.75)
# test_policy(p,50,False,bottom=True)
# test_policy(p,50,False,sd=1)
# test_policy(p,50,False,sd=1)
## migh be the best yet
#cant get it up from botom, but can keep ot there for 0.5 , and maybe a bit more try 0.75 also keeps it very near to x=0

# nlm = generate_non_linear_policy(11)
# res= optimize.minimize(non_linear_loss_function,x0=np.ones((11)) ,method='Nelder-Mead')
# print(res)
# test_policy(res.x,50,nlm=nlm)

# p = np.array([-250.23134627,  -55.56979565,   38.01778897,   30.88543913,
#        -104.44498873,   42.26508438,  319.7611134 ,  -26.40337195,
#         -16.56199577,  -33.06672362]) ## just didnt work


#with 11 didnt work array([ -0.72926725,  -7.13655484,  15.40050596,  -8.49481581,
     #  -29.78226858,   3.86383929,   9.67926091,   8.67618803,
     #    0.74186436,  16.02190835,   1.88707695])


# test_policy(p,50,nlm=nlm)
# test_policy(p,50,True,nlm=nlm)
# test_force_start(10,9)
# test_force_start(20,9)

# test_force_start(10,7,False)
# test_force_start(20,7,True)



#try optimise 5 locations
# res= optimize.minimize(non_linear_loss_function,x0=[1,1,1,1,1  , 0,0,np.pi,0 ,0,0,0,0, 0,0,np.pi/2,0 , 0,0,-np.pi/2,0 , 0, 3,0,3] ,method='Nelder-Mead')
# print(res)
# [-6.59383036e+00,  6.78458229e+00, -1.51580341e+00, -6.09327591e+00,1.80406865e+01
#         ,  3.89023227e-03,  5.21471762e-02,  1.43035453e+00,1.07703253e-02
#         ,  7.33327374e-03,  4.43271917e-03,  3.99075698e-02,-2.99869444e-02
#        , -1.33294311e-02, -8.96068834e-04,  1.76919500e+01, -2.39294072e-03
#       ,  1.15807067e-02, -1.28465198e-01, -5.63118894e-01,-9.62780792e-02,
#          1.59111089e-02,  9.28812961e-01, -1.12125948e-02,1.02995201e+00]

# nlm = non_linear_model(5,policy=True)
# nlm.sd=np.ones((4,5))
# nlm.basis = np.array([[3.9e-3,7.3e-3,-1.33e-2,1.15e-2,1.59e-2],[5.21e-2,4.43e-3,-8.96e-4,-1.28e-1,9.28e-1],[1.43,4e-2,1.77e1,-5.63e-1,-1.12e-2],[1.08e-2,-3e-2,-2.39e-3,-9.63e-2,1.03]])
#
# p=np.array([-6.59383036e+00,  6.78458229e+00, -1.51580341e+00, -6.09327591e+00,1.80406865e+01])
#
# test_policy(p,50,nlm=nlm)
# test_policy(p,50,True,nlm=nlm)
#only did small oscilations about base


# nlm = generate_non_linear_policy(10)
# res= optimize.minimize(non_linear_loss_function,x0=np.ones((10)) ,method='Nelder-Mead')
# print(res)
# test_policy(res.x,50,nlm=nlm)

#about top
# p=np.array([ -1.31362979,   0.25518872, -12.80932662,  11.16310449,
#         15.87181763,  -9.17540978,   9.21673611, -12.86441258,
#          2.6984682 ,  -4.25570855])
# # res= optimize.minimize(non_linear_loss_function,x0=p ,method='Nelder-Mead')
# # print(res)
# nlm = generate_non_linear_policy(10)
# nlm.read_from_file('nonzero')
# test_policy(p,50,nlm=nlm,Top=True,visible=False)


#try with optimise top two positions
#
# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 5000,"maxfev":10000}
# res= optimize.minimize(non_linear_loss_function,x0=[-5,-5] ,method='Nelder-Mead',options=options)
# print(res)
# pres = np.array([-3.01099963, -0.52534223,  0.06382456, -0.01904216,  0.26624947,
#         1.48003872, -0.00567868,  0.89882945,  0.23362329,  0.26767945])   # when points can move
# pres=[res.x[0],-res.x[0],res.x[1],-res.x[1]]
# nlm.read_from_file('justtop')
# test_policy(pres,50,nlm=nlm,Top=True,visible=False)


#symetric points about the range , witth 4* init p=np.array([-1.15711946,  1.15711946])
#symetric points and p
# nlm.read_from_file('justtop')
# nlm.basis[:,0] = res.x[1:5].T
# nlm.basis[:,1] = -res.x[1:5].T
# p=[res.x[0],-res.x[0]]
# test_policy(p,50,nlm=nlm,Top=True,visible=False)
#array([-1.12426169e+00, -3.62897394e-04,  3.09121301e-01,  3.51876147e-01,6.59403563e-01])  ## which oscilates , try intiilising at a different p val (not -1)
#start at 2, didnt work x: array([11.70325351, -0.13364436,  0.12660699, -0.04405935, -0.82851625])
#start at -5  array([-5.47268559e+00, -1.09839130e-03,  5.24551120e-01,  2.68130009e-01,5.02517841e-01])  # seemed to try
#try -10 x: array([-9.85352053e+00, -4.96117708e-05,  5.16162925e-01,  2.59056507e-01,4.93672245e-01])  #changes direction too strongly
#try -7 array([-6.93450059e+00,  6.54290704e-05,  5.06569486e-01,  2.49215232e-01,5.11567391e-01]) # not strong enugh
#try -8  x: array([-8.26406333e+00,  6.48441315e-04,  4.17808749e-01,  3.44625895e-01,4.33674897e-01]) did 5k iterations....
#try -8.26 with thee locations

#trying various situagions for the loss func, including penalise velocity everywhere


#try 4 points as when the pendulum changes direction its not working
#didnt work
#try centering 2 points on 0v, 0theta dot
# almost works but theta dot is still getting too big, try increasing sd from 3 to 5 (p=-13.15 before)
#p=-13.13 almost works, try 7



#tried it on the real model and it looks like its just accelerating it loads
# so try 4 pints again
#on real model it actually worked x: array([-8.74489119, -2.66876906])
#x: array([-31.1264486 , -19.99398314]) on trained model  very fast oscillations tho
#try lowerr intiialisation point (-5,-5) not(-14,-14) x: array([-37.61391782, -20.38823026]) is slicghtly better
# increase size of start to 5* init x: array([-23.2721317 ,  -6.67715398])

# p=np.array([-23.2721,23.2721,-6.677,6.677]) #large init
# #p=np.array([-37.62,37.62,-20.388,20.388]) #small init
# p=np.array([-21.17,21.17,-5.17,5.17]) #large init more v penalty
# p=np.array([-27.669,27.669,-10.98,10.98])
# nlm = generate_non_linear_policy(10)
# nlm.read_from_file('justtop')
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,bottom=True,nlm=nlm)
# test_policy(p,50,False,sd=1,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=1,Top=True,nlm=nlm)
#had problems with sliding off slowly, but worked alrigght at 0.5, and some of the 1's which lauchned the pole upwards
# the one trained on the smaller init had more problems with x gtrailing off , out of the range and therefore cauing the pendulum to fall


#try and penalise a velocity more , with the larger region # use 10 not 20, (note these were trained with angular velocity penalised eerywhere, which is fine because atm i only want to be in the top section moving slowly
#x: array([-21.16835599,  -5.16637694])
# didnt work, i should really be penalising x poition not bvelocity as velocity is required to move the thing and keep steady
#return v penalty to 20, and add position penalty
#x: array([-3.9254275 , -5.57901667]) initially wayy too strong , add a dive by 10
#x: array([-27.66941614, -10.98365538])
#seems a bit better



#try optimising locations as well
# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 5000,"maxfev":10000}
# res= optimize.minimize(non_linear_loss_function,x0=[0,0,0.25,2.5, 0,0,0.25,-2.5] ,method='Nelder-Mead',options=options)
# print(res)
# pres=[res.x[0],-res.x[0],res.x[1],-res.x[1]]
# nlm.read_from_file('justtop')
# nlm.pp()
# nlm.sd[0,:]=[30,30,30,30]
# nlm.sd[1,:]=[30,30,30,30]
# # pres=[-27.669,27.669,-10.98,10.98]
# nlm.basis[:, 0] = res.x[ : 4].T
# nlm.basis[:, 1] = -nlm.basis[:,0]
# nlm.basis[:, 2] = res.x[ 4:4+ 4].T
# nlm.basis[:, 3] = -nlm.basis[:,2]
# test_policy(pres,50,nlm=nlm,Top=True,visible=False,sd=0.68)

#x: array([-0.00988998, -0.03803768,  0.57465364,  2.65458026,  0.00525632,0.09308213,  0.88202638, -1.32182546])


# p=np.array([-27.669,27.669,-10.98,10.98])
# nlm = generate_non_linear_policy(10)
# nlm.read_from_file('justtop')
# nlm.sd[0,:]=[30,30,30,30]
# nlm.sd[1,:]=[30,30,30,30]
# nlm.basis[:,0] = np.array([ -0.00988998, -0.03803768,  0.57465364,  2.65458026]).T
# nlm.basis[:,1] = -nlm.basis[:,0]
# nlm.basis[:,2] = np.array([ 0.00525632,0.09308213,  0.88202638, -1.32182546 ]).T
# nlm.basis[:,3] = -nlm.basis[:,2]
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=0.5,Top=True,nlm=nlm)
# test_policy(p,50,False,bottom=True,nlm=nlm)
# test_policy(p,50,False,sd=1,Top=True,nlm=nlm)
# test_policy(p,50,False,sd=1,Top=True,nlm=nlm)
# #didnt work



# now try starting from bottom and fliging upwards into the top four which remain constant
#try it with just righthand side basis location (posive theta) as direction shouldnt matter
#make it so that fast theta dot isnt penalised everywhere
# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 5000,"maxfev":10000}
# res= optimize.minimize(non_linear_loss_function,x0=[100,-100, -75,-15] ,method='Nelder-Mead',options=options)
# print(res)
# pres=[-27.669,27.669,-10.98,10.98,res.x[0],res.x[1],res.x[2],res.x[3]]
# pres=[-27.669,27.669,-10.98,10.98,76, -77,0, -37.025]
# nlm.read_from_file('wholetest')
# test_policy(pres,100,nlm=nlm,Top=False,visible=False)

#x: array([23.75144614, 309.11040632, -154.83442517])  didnt work (with only one point at the bottom, in a attempt to get it up in one swing

#x: array([ 9.11438853e+01, -8.47006718e+01, -7.72495253e+02,  1.28177381e-01])  gets it up in two swings, however its wayyy too fast , change the near top one to be a slow down one centered on -10 theta dot
# the last point has a value of 0.04 so is pretty useless, try itialising it at 50 # made the thing just oscilalte
#try at 10 with a smaller sd for theta  x: array([ 100.74749168, -101.83204991,  -77.14597527,    9.72082445])
##looks vaguly promising hwoever it first goes over at x=6 so extend sd on the top 4 from 3 to 7, made no difference
#try intiialising at 5 #didnt work

#try penalising angle more
#x: array([-14.83120117,  18.16611363,  42.71462915,  33.59296929])  this didnt work, probably because of the effect of loss angle on the other penaltyies


#pres=[-27.669,27.669,-10.98,10.98,100, -101,-77, -15]
#change sd of x,v to 30 and it works for a while, need to fix a constant velocity of 9 tho
#run this throuhg optimisation with the large sd
#x: array([ 1656.47761715, -1689.18218664,  1389.80530118,   121.90091981])  still shoots off

#changing from the two being 80, -80 to -75,75 makes the velocity change direction
#pres=[-27.669,27.669,-10.98,10.98,75, -75,-77, -15]
#77 is a good value and almost works, clearly it was trying to do it wayyy too fast
#try reducing the impulse at 1/2 at about , 69 it seems better

# reduce all of them as its still going very fast at the start?

#problem is the pendulum isnt at the top until after 20 iterations
#up to 40 or initialise at a mid point



#pres=[-27.669,27.669,-10.98,10.98,76, -77,0, -37.025]  and change the 2/3 point to be at 0.2 sd for angle
#gets it up in half a swing

# run it for with initialisation at about 20 iteration [-5.5,-3.36,-0.56,0.42]
# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 5000,"maxfev":10000}
# res= optimize.minimize(non_linear_loss_function,x0=[76,-77, 0,-37.025] ,method='Nelder-Mead',options=options)
# print(res)
# pres=[-27.669,27.669,-10.98,10.98,res.x[0],res.x[1],res.x[2],res.x[3]]
# pres=[-27.669,27.669,-10.98,10.98,76, -77,0, -37.025]
# nlm.read_from_file('wholetest')
# test_policy(pres,100,nlm=nlm,Top=False,visible=False)

#x: array([ 7.95716748e+01, -8.06053706e+01, -2.32638157e-04, -3.77144461e+01])  #not useful tho

# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 30,"maxfev":50}
# res= optimize.minimize(non_linear_loss_function,x0=[76,-77,-37.025] ,method='Nelder-Mead',options=options)
# print(res)
# pres=[-27.669,27.669,-10.98,10.98,res.x[0],res.x[1],0,res.x[2]]
#pres=[-27.669,27.669,-10.98,10.98,76, -77,0, -37.025]
# pres=[-27.669,27.669,-10.98,10.98,77, -77,0, -30.575]
#pres=[-27.669,27.669,-10.98,10.98,37.2 ,38.8,0, -108.1]
#pres = [-27.669,27.669,-10.98,10.98,71.71793911, -79.06023329, 0,-38.85691662]
# nlm.read_from_file('flickup')
# test_policy(pres,100,nlm=nlm,Top=False,visible=False)

# if bottom are different then x: array([ 71.71793911, -79.06023329, -38.85691662])
#else it failed

# go back to optimising the top sections, and intiialise at the [-5.5,-3.36,-0.56,0.42]
# also go back to using model with 1000 basis points that i stopped using for some reason
# nlm = generate_non_linear_policy(10)
# # options = {"disp": True, "maxiter": 5000,"maxfev":10000}
# # res= optimize.minimize(non_linear_loss_function,x0=[-27.669,-10.98] ,method='Nelder-Mead',options=options)
# # print(res)
# # pres=[res.x[0],-res.x[0],res.x[1],-res.x[1],76, -77,0, -37.025]
# pres=[-27.669,27.669,-10.98,10.98,76, -77,-20, -37.025]
# nlm.read_from_file('wholetest')
# test_policy(pres,100,nlm=nlm,Top=False,visible=False)

#x: array([-27.77622433, -11.66771078])

#add in bounce back
# nlm = generate_non_linear_policy(10)
# options = {"disp": True, "maxiter": 50,"maxfev":10}
# res= optimize.minimize(non_linear_loss_function,x0=[-20,-18.95] ,method='Nelder-Mead',options=options)
# print(res)
# pres=np.array([-27.669,27.669,-10.98,10.98,76,-77,res.x[0],-37.025,-res.x[0],res.x[1],-res.x[1]])
#pres=[-27.669,27.669,-10.98,10.98,76, -77,-20, -37.025,20 ,-18.95,18.95]#-12.74,12.74]
# pres=[-27.669,27.669,-10.98,10.98,76, -77,-19.9, -37.025,19.9 ,-18.95,18.95]#-12.74,12.74]
# nlm.read_from_file('wholetest')
# test_policy(pres,250,nlm=nlm,Top=False,visible=False,sd=0.1)
# test_policy(pres,250,nlm=nlm,Top=False,visible=False,sd=0.1)
# test_policy(pres,250,nlm=nlm,Top=False,visible=False,sd=0.1)
# test_policy(pres,250,nlm=nlm,Top=False,visible=False,sd=0)


# need to adjust the the velocity basis position of the slow down so that it doesnt act when the cart goes the opposite direction
#pres=[-27.669,27.669,-10.98,10.98,76, -77,-20, -37.025,20 ,-32.95,32.95] changed x positions to be at +-0.5 to try and stop oscilations, this setup caused a flip but a perfect stop afterwards
#pres=[-27.669,27.669,-10.98,10.98,76, -77,-20, -37.025,20 ,-18.95,18.95]#-12.74,12.74] has no flip

#doesnt seem that robust to random starts, might need to change the bounch back basis to be active over larger velocity ranges, or get the swing up to be better and not involve a massive change in x

test_noise(30,500,f=True)

# x,y = get_random_data_pairs(10000,quasi=True,f=True,noise=0.2)
# sd = np.std(x, axis = 1)
# model = non_linear_regression(x,y,1000,sd,4e-4,f=f)
# model.write_to_file('noisymodel2')
# model.pp()
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)
# test_model(model,100,model=1,f=f)