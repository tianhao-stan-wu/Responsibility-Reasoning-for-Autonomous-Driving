'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 16, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''

import numpy as np
import cvxpy as cp

class CbfQp:
    """
    This is the implementation of the vanilla CBF-CLF-QP method. The optimization problem is:

            min (u-u_ref).T * H * (u-u_ref) + p * delta**2
            s.t. L_f V(x) + L_g V(x) * u + lambda * V(x) <= delta  ---> CLF constraint
                 L_f B(x) + L_g B(x) * u + gamma * B(x) >= 0  ---> CBF constraint

    Input:
    :param  system  :   The dynamic system of interest, containing CBF, CLF, and their Lie derivatives
    :param  x       :   The current state x
    :param  u_ref   :   The reference control input
    """
    def __init__(self, system, params):
        if hasattr(system, 'udim'):
            self.udim = system.udim
        else:
            raise KeyError('udim is not given in the system dynamic!')

        self.cbf = system.cbf

        # todo check lf.lg/cbf symbolic expression and their size!
        self.lf_cbf = system.lf_cbf
        self.lg_cbf = system.lg_cbf
        
        # todo take input from the option class
        self.weight_input = params['weight_input']
        self.H = None

        self.cbf_gamma = params['cbf_gamma']
        self.u_max = params['u_max']
        self.u_min = params['u_min']
    

    def cbf_qp(self, x, u_ref=None):
        """
        Solves a Quadratic Program (QP) to find the optimal control input u
        subject to only the Control Barrier Function (CBF) constraint.
        """
        inf = np.inf
        
        # Set reference control input
        if u_ref is None:
            u_ref = np.zeros(self.udim)
        else:
            if u_ref.shape != (self.udim,):
                raise ValueError(f'u_ref should have the shape size (u_dim,), now it is {u_ref.shape}')
        
        # Define the quadratic cost matrix H
        if self.weight_input.shape == (1, 1):
            self.H = self.weight_input * np.eye(self.udim)
        elif self.weight_input.shape == (self.udim, 1):
            self.H = np.diag(self.weight_input)
        elif self.weight_input.shape == (self.udim, self.udim):
            self.H = np.copy(self.weight_input)
        else:
            self.H = np.eye(self.udim)
        
        # Compute the CBF terms
        B = self.cbf(x)       # Barrier function B(x)
        lf_B = self.lf_cbf(x) # Lie derivative LfB
        lg_B = self.lg_cbf(x) # Lie derivative LgB
        print("lg_B", lg_B)
        
        # Construct the QP problem
        u = cp.Variable((self.udim,))
        objective = cp.Minimize(0.5 * cp.quad_form(u - u_ref, self.H))
        
        # Define CBF constraint: LfB + LgB * u + gamma * B >= 0
        constraints = [lf_B + lg_B @ u + self.cbf_gamma * B >= 0]
        
        # Enforce control input bounds
        constraints += [self.u_min <= u, u <= self.u_max]
        
        # Solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Check for feasibility and return the optimal control input
        if problem.status != 'infeasible':
            u = u.value
            feas = 1
        else:
            u = None
            feas = -1
        
        return u, B, feas




