import numpy as np
import sympy as sp
from sympy import cos, sin, Max

class vehicle:
    """
    Define the symbolic dynamic:    dx = f(x) + g(x) * u
    lr: distances of the rear axles from the CoM
    lf: distances of the front axles from the CoM
    xo: x position of obstacle
    yo: y position of obstacle
    sd: safe distance of CBF
    """
    def __init__(self, params):

        self.lr = params['lr']
        self.lf = params['lf']
        self.xo = params['xo']
        self.yo = params['yo']
        self.ro = params['ro']
        # self.sd = Max(self.lr, self.lf) + self.ro
        # ignore car shape for now
        self.sd = self.ro + 0.2

        x, y, theta, v = sp.symbols('x y theta v')
        self.x  = sp.Matrix([x, y, theta, v])

        # Define the symbolic expression for system dynamic and CBF
        self.f, self.g = self.simple_car_dynamics()
        self.cbf = self.define_cbf()

        if 'udim' in params.keys():
            self.udim = params['udim']
        else:
            print(f'The dimension of input u is not given, set it to be default 1')
            self.udim = 1

    def simple_car_dynamics(self):
        # f, g both column vector
        f = sp.Matrix([self.x[3]*cos(self.x[2]), self.x[3]*sin(self.x[2]), 0 , 0])
        g = sp.Matrix([[0, self.x[3]*sin(self.x[2])*-1], [0, self.x[3]*cos(self.x[2])], [0, self.x[3]/self.lr], [1, 0]])
        return f, g

    def define_cbf(self):
        cbf = (self.x[0] - self.xo) ** 2 + (self.x[1] - self.yo) **  2 - self.sd ** 2 - self.x[3] ** 2
        return cbf

