'''
=====================================
Author  :  Tianhao Wu
Email : twu4@andrew.cmu.edu
=====================================
adapted from
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

from sympy.utilities.lambdify import lambdify
import cvxpy as cp


class KinematicBicycle:
    """
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    lr: distances of the rear axles from the CoM
    lf: distances of the front axles from the CoM
    Carla model: tesla model3
    """
    def __init__(self):

        self.udim = 2
        self.lf = 1.553
        self.lr = 2.102

        x, y, theta, v = sp.symbols('x y theta v')
        self.x  = sp.Matrix([x, y, theta, v])

        # Define the symbolic expression for system dynamic and CBF
        self.f_symbolic, self.g_symbolic = self.kinematic_bicycle()

        self.f = lambdify(np.array(self.x.T), self.f_symbolic, 'numpy')
        self.g = lambdify(np.array(self.x.T), self.g_symbolic, 'numpy')

        
    def kinematic_bicycle(self):
        # f, g both column vector
        f = sp.Matrix([self.x[3]*cos(self.x[2]), self.x[3]*sin(self.x[2]), 0 , 0])
        g = sp.Matrix([[0, self.x[3]*sin(self.x[2])*-1], [0, self.x[3]*cos(self.x[2])], [0, self.x[3]/self.lr], [1, 0]])

        return f, g


