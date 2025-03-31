import numpy as np
import kinematic_bicycle as kb
import ode_solver
import math

import matplotlib.pyplot as plt

from drive_modules import *

start = np.array([0,0])
end = np.array([-5,5])
module = turn()

T = 3
dt = .02
x0 = np.array([0, 0, math.pi/2, 0])
u_ref = np.array([3, 0.5])

time_steps = int(np.ceil(T / dt))

ode_sol = ode_solver.OdeSolver(module, dt)

# initialize the input matrices
xt = np.zeros((4, time_steps))
tt = np.zeros((1, time_steps))
ut = np.zeros((2, time_steps))

Bt = np.zeros((1, time_steps))
xt[:, 0] = np.copy(x0)

for t in range(time_steps-1):
    

    # solve for control u at current time step
    u = module.solve(xt[:, t], u_ref)
 
    ut[:, t] = np.copy(u)

    # propagate the system with control u using RK4
    xt[:, t + 1] = ode_sol.time_marching(xt[:, t], u)

    # print(xt[:, t + 1])




t = np.arange(0, T, dt)

# Create a figure with 3 subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# --- Plot 1: State Trajectory ---
ax1.set_aspect('equal', adjustable='datalim')  # Ensure correct aspect ratio
ax1.grid()
ax1.plot(xt[0, :], xt[1, :], linewidth=3, color='magenta', label="Trajectory")

# Mark start and end points for clarity
ax1.scatter(start[0], start[1], color='green', s=100, label="Start", zorder=3)
ax1.scatter(end[0], end[1], color='red', s=100, label="End", zorder=3)

ax1.set_title('State Trajectory')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# --- Plot 2: State Variables Over Time ---
theta = xt[2, :]  # Assuming theta is at index 2 of xt (orientation)
v = xt[3, :]      # Assuming v is at index 3 of xt (velocity)

ax2.plot(t, theta, label='Theta (Orientation)', linewidth=2, color='blue')
ax2.plot(t, v, label='Velocity (v)', linewidth=2, color='green')

ax2.set_title('State Variables over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('State Variables')
ax2.legend()

# --- Plot 3: Control Inputs Over Time ---
a = ut[0, :]  # Acceleration is assumed to be at index 0 of ut
beta = ut[1, :]  # Steering angle is assumed to be at index 1 of ut

ax3.plot(t, a, label='Acceleration (a)', linewidth=2, color='red')
ax3.plot(t, beta, label='Steering Angle (beta)', linewidth=2, color='purple')

ax3.set_title('Control Inputs over Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Control Inputs')
ax3.legend()

# Prevent automatic figure resizing
fig.tight_layout()

plt.show()
plt.close(fig)
