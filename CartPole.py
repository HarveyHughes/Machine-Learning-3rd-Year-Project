"""
fork from python-rl and pybrain for visualization
"""
import numpy as np
from pylab import ion, draw, Rectangle, Line2D
import pylab as plt

class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi    # angle is defined to be zero when the pole is upright, pi when hanging vertically down
        self.pole_velocity = 0.0
        self.visual = visual

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.1 # 0.005    # friction coefficient of the cart
        self.mu_p = 0.0000 # 0.000002 # friction coefficient of the pole
        self.sim_steps = 200           # number of Euler steps to perform in one go
        self.delta_time = 0.1        # time step of the Euler integrator
        self.max_force = 10.
        self.gravity = 9.8
        self.cart_mass = 0.5

        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2

        if self.visual:
            self.drawPlot()

    def setState(self, state):
        self.cart_location = state[0]
        self.cart_velocity = state[1]
        self.pole_angle = state[2]
        self.pole_velocity = state[3]
            
    def getState(self):
        return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity])

    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi
        self.pole_velocity = 0.0


    def drawPlot(self):
        ion()
        self.fig = plt.figure()
        # draw cart
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.box = Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle)], 
                           [0, np.cos(self.pole_angle)], linewidth=3, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-0.5, 2)


    def _render(self):
        self.box.set_x(self.cart_location - self.cartwidth / 2.0)
        self.pole.set_xdata([self.cart_location, self.cart_location + np.sin(self.pole_angle)])
        self.pole.set_ydata([0, np.cos(self.pole_angle)])
        draw()

        plt.pause(0.015)

  
    def performAction(self, action):
        force = self.max_force * np.tanh(action/self.max_force)

        for step in range(self.sim_steps):
            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
            cart_accel = (-2.0*self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+3.0*self.pole_mass*self.gravity*c*s+4.0*(force-self.mu_c*self.cart_velocity) )/m
            
            pole_accel = (-3.0*self.pole_length*self.pole_mass*(self.pole_velocity**2)*s*c + 6.0*(self.cart_mass+self.pole_mass)*self.gravity*s + 6.0*(force-self.mu_c*self.cart_velocity)*c)/(m*self.pole_length)

            # Update state variables
            df = (self.delta_time / float(self.sim_steps))
            self.cart_location += df * self.cart_velocity
            self.cart_velocity += df * cart_accel
            self.pole_angle += df * self.pole_velocity
            self.pole_velocity += df * pole_accel

        if self.visual:
            self._render()


    def remap_angle(self):
        # If theta  has gone past our conceptual limits of [-pi,pi]
        # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
        while self.pole_angle < -np.pi:
            self.pole_angle += 2. * np.pi
        while self.pole_angle > np.pi:
            self.pole_angle -= 2. * np.pi
       
    
    # the loss function that the policy will try to optimise (lower)
    def loss(self):
        # first of all, we want the pole to be upright (theta = 0), so we penalise theta away from that
        loss_angle_scale = np.pi/2.0
        loss_angle = 1.0-np.exp(-0.5*self.pole_angle**2/loss_angle_scale**2)
        # but also, we want to HOLD it upright, so we also penalise large angular velocities, but only near
        # the upright position
        loss_velocity_scale = 0.1
        loss_velocity = (1.0-loss_angle)*(self.pole_velocity**2)*loss_velocity_scale
        return loss_angle + loss_velocity
    
    def terminate(self):
        """Indicates whether or not the episode should terminate.

        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return np.abs(self.cart_location) > self.state_range[0, 1] or \
               (np.abs(self.pole_angle) > self.state_range[2, 1]).any()

def remap_angle(theta):
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta
    

def loss_pos(x):
    # first of all, we want the pole to be upright (theta = 0), so we penalise theta away from that
    loss_angle_scale = np.pi/2.   #make division larger for larger penly
    loss_angle = 1.0-np.exp(-0.5*x[2]**2/loss_angle_scale**2)
    # but also, we want to HOLD it upright, so we also penalise large angular velocities, but only near
    # the upright position
    loss_velocity_scale = 0.1
    loss_velocity =(1.0-loss_angle)*(x[3]**2)*loss_velocity_scale #(1.0-loss_angle)*(x[3]**2)*loss_velocity_scale
    loss_v = (1.0-loss_angle) * x[1]**2 * loss_velocity_scale/20
    loss_pos = (1.0-loss_angle) * x[0]**2 * loss_velocity_scale/10
    return loss_angle + loss_velocity +loss_v +loss_pos #abs(x[1])/5  #+ loss_v

