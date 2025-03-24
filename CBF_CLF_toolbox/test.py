import numpy as np
import kinematic_bicycle as kb
import define_system
import cbf_clf_qp as ccq
import ode_solver
import math


vehicle = kb.KinematicBicycle()

x = np.array([0,0,math.pi/2,20])
f = vehicle.f(x)
print(f)