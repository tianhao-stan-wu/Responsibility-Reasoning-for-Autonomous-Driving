import numpy as np
import dynamics
import define_system
import cbf_qp as cq
import ode_solver
import matplotlib.pyplot as plt


def to_radian(degree):
	return degree * np.pi / 180

dynamics_params = {
    'lr': 2.71,
    'lf': 2.71,
    'xo': 10,
    'yo': 10,
    'ro': 1,
    'udim': 2
}

ego_vehicle = dynamics.vehicle(dynamics_params)
ds = define_system.ControlAffineSystem(ego_vehicle)

beta = to_radian(70)

qp_params = {
	'weight_input': np.array([[1]]),
	'cbf_gamma': 1,
	'u_min': np.array([-10, -beta]),
	'u_max': np.array([10, beta]),
}

qp = cq.CbfQp(ds, qp_params)

T = 5
dt = 0.01
time_steps = int(np.ceil(T / dt))

# theta in degree
theta = to_radian(45)
x0 = np.array([[0, 0, theta, 7]]).T

time_steps = int(np.ceil(T / dt))
ode_sol = ode_solver.OdeSolver(ds, dt)

# initialize the input matrices
xt = np.zeros((4, time_steps))
ut = np.zeros((2, time_steps))
Bt = np.zeros((1, time_steps))
xt[:, 0] = np.copy(x0.T[0])

for t in range(time_steps-1):

    if t % 100 == 0:
        print(f't = {t}')

    # solve for control u at current time step
    # no reference control for now
    u, B, feas = qp.cbf_qp(xt[:, t])
    if feas == -1:
        # infeasible
        print("infeasible at t=", t)
        break
    else:
        pass

    ut[:, t] = np.copy(u)
    Bt[:, t] = np.copy(B)

    print(u)

    # propagate the system with control u using RK4
    xt[:, t + 1] = ode_sol.time_marching(xt[:, t], u)


def state_trajectory(xt, dt, T, show=1, save=0):
    t = np.arange(0, T, dt)
    
    # Create fixed-size figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')  # Ensures the plot is not distorted
    
    # Grid and trajectory plot
    ax.grid()
    ax.plot(xt[0, :], xt[1, :], linewidth=3, color='magenta', label="Trajectory")
    
    # Add circle at (10,10) with radius 1
    circle = plt.Circle((10, 10), dynamics_params['ro'], color='blue', fill=False, linewidth=2, linestyle='dashed')
    ax.add_patch(circle)
    
    # Labels and title
    ax.set_title('State Trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    # Prevent automatic figure resizing
    fig.tight_layout()
    
    # Show or save figure
    if show == 1:
        plt.show()
    if save == 1:
        fig.savefig('velocity.png', format='png', dpi=300)
    
    plt.close(fig)


def state_relative_distance(xt, dt, T, show=1, save=0):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t, xt[2, :], linewidth=3, color='black')
    plt.ylim(0, 100)
    plt.title('State - Relative distance')
    plt.ylabel('z')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('relative_distance.png', format='png', dpi=300)
    plt.close(fig)


def cbf(Bt, dt, T, show=1, save=0):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], Bt[0, :-1], linewidth=3, color='red')
    plt.title('cbf')
    plt.ylabel('B(x)')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('cbf.png', format='png', dpi=300)
    plt.close(fig)


    
def control(u, dt, T, show=1, save=0):
    u_max = .3 * 9.8 * 1650
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], u[0, :-1], linewidth=3, color='dodgerblue')
    plt.plot(t, u_max * np.ones(t.shape[0]), 'k--')
    plt.plot(t, -u_max * np.ones(t.shape[0]), 'k--')
    plt.title('control')
    plt.ylabel('u(t, x)')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('control.png', format='png', dpi=300)

    plt.close(fig)


state_trajectory(xt, dt, T)
# state_relative_distance(xt, dt, T)
# cbf(Bt, dt, T)
# control(ut, dt, T)





