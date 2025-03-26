import numpy as np
import kinematic_bicycle as kb
import ode_solver
import math

from modules import *


def straight(start, end, v_limit, d, x0, u_ref):


	module = normal_straight(start, end, v_limit, d)

	# solve for control u at current time step
	u = module.solve(x0, u_ref)

	return u