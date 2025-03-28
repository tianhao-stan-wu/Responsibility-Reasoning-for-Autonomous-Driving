'''
=====================================
Author  :  Tianhao Wu
Email : twu4@andrew.cmu.edu
=====================================
'''


from kinematic_bicycle import KinematicBicycle
import numpy as np
import sympy as sp
from sympy import cos, sin, Max
from numpy import pi

from sympy.utilities.lambdify import lambdify
import cvxpy as cp


class normal_straight:
    """
    drive straight module
    """
    def __init__(self, start, end, v_limit, d):

        self.start = start
        self.end = end
        self.v_limit = v_limit
        self.d = d

        self.model = KinematicBicycle()
        self.x = self.model.x
        self.udim = self.model.udim

        self.f_symbolic = self.model.f_symbolic
        self.g_symbolic = self.model.g_symbolic
        self.f = self.model.f
        self.g = self.model.g

        # cbf1 ensures v < speed_limit
        self.cbf1_symbolic = self.v_limit - self.x[3]
        # cbf2 ensures lane keeping
        self.cbf2_symbolic = self.define_CBF2()

        self.cbf1 = lambdify(np.array(self.x.T), self.cbf1_symbolic, 'numpy')
        self.cbf2 = lambdify(np.array(self.x.T), self.cbf2_symbolic, 'numpy')

        self.lf_cbf1, self.lg_cbf1 = self.lie_derivatives_calculator(self.cbf1_symbolic, self.f_symbolic, self.g_symbolic)
        self.lf_cbf2, self.lg_cbf2 = self.lie_derivatives_calculator(self.cbf2_symbolic, self.f_symbolic, self.g_symbolic)

        # to be tuned
        self.gamma = 3
        self.alpha2 = 3

        self.u_max = np.array([12, 1], dtype=float)
        self.u_min = np.array([-16, -1], dtype=float)
        

    def define_CBF2(self):
        """
        Define the CBF that ensures the vehicle remains within distance d from the line.
        """
        x_s, y_s = self.start
        x_e, y_e = self.end
        x, y, _, _ = self.x  # Extract state variables

        # Compute perpendicular distance to the line
        num = (y_e - y_s) * x - (x_e - x_s) * y + x_e * y_s - y_e * x_s
        den = sp.sqrt((x_e - x_s) ** 2 + (y_e - y_s) ** 2)
        distance_sq = (num / den) ** 2

        # print("dis_sq", distance_sq)

        # Define the CBF h_d = d^2 - distance^2
        cbf = self.d**2 - distance_sq
        return cbf


    def lie_derivatives_calculator(self, cbf_clf_symbolic, f_symbolic, g_symbolic):
        """
        Compute the Lie derivatives of CBF or CLF w.r.t to x
        :return:
        """
        dx_cbf_clf_symbolic = sp.Matrix([cbf_clf_symbolic]).jacobian(self.x)  

        self.lf_cbf_clf_symbolic = dx_cbf_clf_symbolic * f_symbolic
        self.lg_cbf_clf_symbolic = dx_cbf_clf_symbolic * g_symbolic

        lf = lambdify(np.array(self.x.T), self.lf_cbf_clf_symbolic, 'numpy')
        lg = lambdify(np.array(self.x.T), self.lg_cbf_clf_symbolic, 'numpy')

        return (lf,lg)


    def solve(self, x, u_ref):
        """
        x: current state
        u_ref: reference control input

        objectives
        1. solve each cbf and get the constraints
        2. solve the qp
        """

        Lf_cbf1 = self.lf_cbf1(x)
        Lg_cbf1 = self.lg_cbf1(x)

        Lf_cbf2 = self.lf_cbf2(x)
        Lg_cbf2 = self.lg_cbf2(x)

        h1 = self.cbf1(x)
        h2 = self.cbf2(x)
        # print("cbf2 value:", h2)

        # Define control input variables (acceleration and steering)
        u = cp.Variable(self.udim)

        # Define the cost function: minimize ||u - u_ref||^2
        cost = cp.norm(u - u_ref, 2)**2

        # Define CBF constraints
        constraints = [
            Lf_cbf1 + Lg_cbf1 @ u + self.gamma * h1 >= 0,  # Velocity limit constraint
            Lf_cbf2 + Lg_cbf2 @ u + self.alpha2 * h2 >= 0,   # Distance constraint
            self.u_min <= u, u <= self.u_max,
        ]

        # Solve the QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Return the optimized control input
        return u.value



class change_speed_straight:
    """
    drive straight module
    """
    def __init__(self, start, end, v0, vd, v_limit, d):

        self.start = start
        self.end = end
        self.v_limit = v_limit
        self.v0 = v0
        self.vd = self.check_vd(vd)
        self.d = d

        self.model = KinematicBicycle()
        self.x = self.model.x
        self.udim = self.model.udim

        self.f_symbolic = self.model.f_symbolic
        self.g_symbolic = self.model.g_symbolic
        self.f = self.model.f
        self.g = self.model.g

        self.acc_min = self.compute_min_acceleration()

        self.cbf2_symbolic = self.define_CBF2()
        self.cbf2 = lambdify(np.array(self.x.T), self.cbf2_symbolic, 'numpy')
        self.lf_cbf2, self.lg_cbf2 = self.lie_derivatives_calculator(self.cbf2_symbolic, self.f_symbolic, self.g_symbolic)

        # to be tuned
        self.gamma = 3
        self.alpha2 = 3

        self.u_max = np.array([12, 1], dtype=float)
        self.u_min = np.array([-16, -1], dtype=float)

        self.update_u_max_min()
        

    def check_vd(self, vd):

        if vd > self.v_limit:
            return self.v_limit

        elif vd < 0:
            return 0

        else:
            return vd


    def compute_min_acceleration(self):
        """
        Computes the minimum constant acceleration required to reach vd before end.
        """
        # Compute the total distance to travel
        s = np.linalg.norm(self.end - self.start)

        # Avoid division by zero
        if s == 0:
            return 0
        
        # Compute acceleration using kinematic equation
        a_min = (self.vd**2 - self.v0**2) / (2 * s)

        if a_min > 12:
            a_min = 12

        elif a_min < -16:
            a_min = -16

        # print(a_min)
        
        return a_min


    def update_u_max_min(self):
        """
        Updates u_min such that u_min[0] is set to acc_min while preserving u_min[1].
        """
        if self.acc_min >= 0:
            
            if self.acc_min <= 12:
                self.u_min[0] = self.acc_min
        
            else:
                self.u_min[0] = 12


        elif self.acc_min < 0:

            if self.acc_min >= -16:
                self.u_max[0] = self.acc_min

            else:
                self.u_max[0] = -16


    def define_CBF2(self):
        """
        Define the CBF that ensures the vehicle remains within distance d from the line.
        """
        x_s, y_s = self.start
        x_e, y_e = self.end
        x, y, _, _ = self.x  # Extract state variables

        # Compute perpendicular distance to the line
        num = (y_e - y_s) * x - (x_e - x_s) * y + x_e * y_s - y_e * x_s
        den = sp.sqrt((x_e - x_s) ** 2 + (y_e - y_s) ** 2)
        distance_sq = (num / den) ** 2

        # Define the CBF h_d = d^2 - distance^2
        cbf = self.d**2 - distance_sq
        return cbf


    def lie_derivatives_calculator(self, cbf_clf_symbolic, f_symbolic, g_symbolic):
        """
        Compute the Lie derivatives of CBF or CLF w.r.t to x
        :return:
        """
        dx_cbf_clf_symbolic = sp.Matrix([cbf_clf_symbolic]).jacobian(self.x)  

        self.lf_cbf_clf_symbolic = dx_cbf_clf_symbolic * f_symbolic
        self.lg_cbf_clf_symbolic = dx_cbf_clf_symbolic * g_symbolic

        lf = lambdify(np.array(self.x.T), self.lf_cbf_clf_symbolic, 'numpy')
        lg = lambdify(np.array(self.x.T), self.lg_cbf_clf_symbolic, 'numpy')

        return (lf,lg)


    def solve(self, x, u_ref):
        """
        1. solve each cbf and get the constraints
        2. solve the qp
        """

        Lf_cbf2 = self.lf_cbf2(x)
        Lg_cbf2 = self.lg_cbf2(x)

        h2 = self.cbf2(x)

        # Define control input variables (acceleration and steering)
        u = cp.Variable(self.udim)

        # Define the cost function: minimize ||u - u_ref||^2
        cost = cp.norm(u - u_ref, 2)**2


        # Define CBF constraints
        constraints = [
            Lf_cbf2 + Lg_cbf2 @ u + self.alpha2 * h2 >= 0,   # Distance constraint
            self.u_min <= u, u <= self.u_max,
        ]

        # Solve the QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Return the optimized control input
        return u.value



        