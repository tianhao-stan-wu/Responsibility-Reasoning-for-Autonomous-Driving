'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 11, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''
import numpy as np
import sympy as sp
from sympy import cos, sin, Max
from numpy import pi


class KinematicBicycle:
    """
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    lr: distances of the rear axles from the CoM
    lf: distances of the front axles from the CoM
    """
    def __init__(self):

        self.udim = 2
        self.lf = 1.553
        self.lr = 2.102

        x, y, theta, v = sp.symbols('x y theta v')
        self.x  = sp.Matrix([x, y, theta, v])

        # Define the symbolic expression for system dynamic and CBF
        self.f, self.g = self.simple_car_dynamics()
        self.cbf = None
        self.clf = None

        
    def simple_car_dynamics(self):
        # f, g both column vector
        f = sp.Matrix([self.x[3]*cos(self.x[2]), self.x[3]*sin(self.x[2]), 0 , 0])
        g = sp.Matrix([[0, self.x[3]*sin(self.x[2])*-1], [0, self.x[3]*cos(self.x[2])], [0, self.x[3]/self.lr], [1, 0]])
        return f, g

    def set_cbf(self, option, params):

        if option == "obstacle":

            # if not specified, set to a trivial obstacle
            if params is None:
                self.cbf = 0

            else:
                xo = params['xo']
                yo = params['yo']
                rsafe = params['rsafe']
                self.cbf = (self.x[0] - xo) ** 2 + (self.x[1] - yo) **  2 - rsafe ** 2

            

    def set_clf(self, option, params):

        if option == "speed":

            if params is None:
                self.clf = 0

            else:
                target_speed = params['target_speed']
                self.clf = 0.5 * ((self.x[3] - target_speed) ** 2)
     




