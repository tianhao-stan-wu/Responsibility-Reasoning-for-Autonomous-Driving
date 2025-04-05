'''
=====================================
Author  :  Tianhao Wu
Email : twu4@andrew.cmu.edu
=====================================
'''


from kinematic_bicycle import KinematicBicycle
import numpy as np
import sympy as sp
import math
from sympy import cos, sin, Max
from numpy import pi

from sympy.utilities.lambdify import lambdify
import cvxpy as cp

import carla
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner



def distance(a,b):

    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)



class combined_modules:
    """
    drive straight module
    """
    def __init__(self):

        # modules that return cbf constraints
        self.speed_limit = speed_limit()

        self.u_max = np.array([12, 1], dtype=float)
        self.u_min = np.array([-16, -1], dtype=float)



    def solve(self, x, u_ref):
        """
        x: current state
        u_ref: reference control input

        objectives
        1. solve each cbf and get the constraints
        2. solve the qp
        """

        # Define control input variables (acceleration and steering)
        u = cp.Variable(self.udim)

        # Define the cost function: minimize ||u - u_ref||^2
        cost = cp.norm(u - u_ref, 2)**2

        # Define constraints
        constraints = []





        # Solve the QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status != 'infeasible':
            u = u.value

        else:
            u = None
            print('infeasible qp problem, return u None')

        return u







class speed_limit:
    '''
    This module ensures the speed never exceeds the specified speed limit
    '''

    def __init__(self):

        self.model = KinematicBicycle()
        self.x = self.model.x
        self.udim = self.model.udim

        self.f_symbolic = self.model.f_symbolic
        self.g_symbolic = self.model.g_symbolic
        self.f = self.model.f
        self.g = self.model.g

        # to be tuned
        self.gamma1 = 5


        self.v_limit = None
        self.cbf1_symbolic = None

        self.cbf1 = None
        self.lf_cbf1 = None
        self.lg_cbf1 = None

        self.activated = False


    def set_speed_limit(self, v_limit):

        self.v_limit = v_limit
        self.cbf1_symbolic = self.v_limit - self.x[3]
        self.cbf1 = lambdify(np.array(self.x.T), self.cbf1_symbolic, 'numpy')

        self.lf_cbf1, self.lg_cbf1 = self.lie_derivatives_calculator(self.cbf1_symbolic, self.f_symbolic, self.g_symbolic)
        self.activated = True


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


    def get_constraint(self, x, u_ref):


        Lf_cbf1 = self.lf_cbf1(x)
        Lg_cbf1 = self.lg_cbf1(x)

        h1 = self.cbf1(x)

        # return params that define the constraint
        # Lf_cbf1 + Lg_cbf1 @ u + self.gamma1 * h1 >= 0,  # Velocity limit constraint

        return (Lf_cbf1, Lg_cbf1, self.gamma1, h1)


class straight:
    """
    drive straight module
    """
    def __init__(self, start, end, v_limit=30, d=0.5):

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
        self.gamma1 = 5
        self.gamma2 = 0.5


        self.u_max = np.array([12, 1], dtype=float)
        self.u_min = np.array([-16, -1], dtype=float)
        

    def define_CBF2(self):
        """
        Define the CBF that ensures the vehicle remains within distance d from the line.
        """
        x_s, y_s = self.start
        x_e, y_e = self.end
        x, y, _, _ = self.x  # Extract state variables

        # avoid numeric instability by adding epsilon
        epsilon = 0.01

        # Compute perpendicular distance to the line
        num = (y_e - y_s) * x - (x_e - x_s) * y + x_e * y_s - y_e * x_s
        den = sp.sqrt((x_e - x_s) ** 2 + (y_e - y_s) ** 2) + epsilon
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


        # print(Lf_cbf1, "+", Lg_cbf1, "*u +", self.gamma1 * h1, ">=0")
        # print(Lf_cbf2, "+", Lg_cbf2, "*u +", self.gamma2 * h2, ">=0")
        # print('***************')
        # print("cbf2 value:", h2)

        # Define control input variables (acceleration and steering)
        u = cp.Variable(self.udim)

        # Define the cost function: minimize ||u - u_ref||^2
        cost = cp.norm(u - u_ref, 2)**2

        # Define CBF constraints
        constraints = [
            Lf_cbf1 + Lg_cbf1 @ u + self.gamma1 * h1 >= 0,  # Velocity limit constraint
            Lf_cbf2 + Lg_cbf2 @ u + self.gamma2 * h2 >= 0,   # Distance constraint
            self.u_min <= u, u <= self.u_max,
        ]

        # Solve the QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status != 'infeasible':
            u = u.value

        else:
            u = None
            print('infeasible qp problem, return u None')

        return u



class change_speed:
    """
    drive straight module
    """
    def __init__(self, start, end, v0, vd):

        self.start = start
        self.end = end
        self.v0 = v0
        self.vd = vd

        self.model = KinematicBicycle()
        self.x = self.model.x
        self.udim = self.model.udim

        self.f_symbolic = self.model.f_symbolic
        self.g_symbolic = self.model.g_symbolic
        self.f = self.model.f
        self.g = self.model.g

        self.acc_min = self.compute_min_acceleration()

        self.u_max = np.array([12, 1], dtype=float)
        self.u_min = np.array([-16, -1], dtype=float)

        self.update_u_max_min()


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


    def solve(self, x, u_ref):

        # Define control input variables (acceleration and steering)
        u = cp.Variable(self.udim)

        # Define the cost function: minimize ||u - u_ref||^2
        cost = cp.norm(u - u_ref, 2)**2

        constraints = [
            self.u_min <= u, u <= self.u_max,
        ]

        # Solve the QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status != 'infeasible':
            u = u.value

        else:
            u = None
            print('infeasible qp problem, return u None')

        return u



class turn:
    """
    decompose turn module into straight modules
    """
    def __init__(self, start, end, world, v_limit=15, d=0.3):


        self.waypoints_list = self.get_waypoints(world, start, end)
        self.len = len(self.waypoints_list)
        self.straight = None
        self.index = 0

        self.v_limit = v_limit
        self.d = d

        self.complete_turn = False

        self.set_module()


    def get_waypoints(self, world, start, end):

        '''
        start, end are carla locations specifying the starting and ending coordinates
        '''

        amap = world.get_map()
        sampling_resolution = 5

        grp = GlobalRoutePlanner(amap, sampling_resolution)

        waypoints = grp.trace_route(start, end)
        waypoints_list = []

        waypoints_list.append((start.x, start.y))

        for w in waypoints:
            waypoints_list.append((w[0].transform.location.x, w[0].transform.location.y))

        waypoints_list.append((end.x, end.y))

        # remove duplicate waypoints generated by carla
        waypoints_list.pop(1)
        waypoints_list.pop(1)

        # for debug
        # for i in waypoints_list:
        #     print(i)

        return waypoints_list


    def set_module(self):
        start = self.waypoints_list[self.index]
        end = self.waypoints_list[self.index+1]

        self.straight = straight(start, end, self.v_limit, self.d)

        print('set module', start, end)


    def check_update_module(self, x):

        if distance(x, self.waypoints_list[self.index+1]) <= self.d * 3 and self.index != self.len-2:
            self.index += 1
            self.set_module()

        elif distance(x, self.waypoints_list[self.index+1]) <= self.d * 3 and self.index == self.len-2:
            print('turn completed')
            self.complete_turn = True


    def solve(self, x, u_ref):

        # print(x, u_ref)

        loc = np.array([x[0], x[1]])

        self.check_update_module(loc)

        if not self.complete_turn:
        
            u = self.straight.solve(x, u_ref)

            return u

        else:
            return u_ref


class change_lane:
    """
    decompose turn module into straight modules
    """

    def __init__(self, start, end, world, v_limit=15, d=0.3):


        self.waypoints_list = self.get_waypoints(world, start, end)
        self.len = len(self.waypoints_list)
        self.straight = None
        self.index = 0

        self.v_limit = v_limit
        self.d = d

        self.complete_lane_change = False

        self.set_module()


    def get_waypoints(self, world, start, end):

        '''
        start, end are carla locations specifying the starting and ending coordinates
        '''

        amap = world.get_map()
        sampling_resolution = 1

        grp = GlobalRoutePlanner(amap, sampling_resolution)

        waypoints = grp.trace_route(start, end)
        waypoints_list = []

        waypoints_list.append((start.x, start.y))

        for w in waypoints:
            waypoints_list.append((w[0].transform.location.x, w[0].transform.location.y))

        n = len(waypoints_list)
        selected_indices = [0, n // 3, 2 * n // 3, n - 1]

        selected_waypoints = [waypoints_list[i] for i in selected_indices]

        for i in selected_waypoints:
            print(i)

        return selected_waypoints


    def set_module(self):
        start = self.waypoints_list[self.index]
        end = self.waypoints_list[self.index+1]

        self.straight = straight(start, end, self.v_limit, self.d)

        print('set module', start, end)


    def check_update_module(self, x):

        if distance(x, self.waypoints_list[self.index+1]) <= self.d * 3 and self.index != self.len-2:
            self.index += 1
            self.set_module()

        elif distance(x, self.waypoints_list[self.index+1]) <= self.d * 3 and self.index == self.len-2:
            print('turn completed')
            self.complete_lane_change = True


    def solve(self, x, u_ref):

        # print(x, u_ref)

        loc = np.array([x[0], x[1]])

        self.check_update_module(loc)

        if not self.complete_lane_change:
        
            u = self.straight.solve(x, u_ref)

            return u

        else:
            return u_ref

        