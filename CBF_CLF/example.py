import numpy as np
import kinematic_bicycle as kb
import define_system
import cbf_clf_qp as ccq
import ode_solver
import matplotlib.pyplot as plt



def to_radian(degree):
    return degree * np.pi / 180.0


def get_control(x0, params):

    """
    x0 = np.array([[x, y, theta, v]]).T
    """

    vehicle = kb.KinematicBicycle()

    if params["obstacle"] is None:
        vehicle.set_cbf("obstacle", None)
    else:
        vehicle.set_cbf("obstacle", params["obstacle"])

    if params["speed"] is None:
        vehicle.set_clf("speed", None)
    else:
        vehicle.set_clf("speed", params["speed"])

    ds = define_system.ControlAffineSystem(vehicle)

    QPoption = ccq.CbfClfQpOptions()
    QPoption.set_option('u_max', np.array([9, to_radian(70)]))
    QPoption.set_option('u_min', np.array([-19, -1 * to_radian(70)]))
    QPoption.set_option('clf_lambda', 100.0)
    QPoption.set_option('cbf_gamma', 2.5)
    QPoption.set_option('weight_input', np.array([1.0]))
    QPoption.set_option('weight_slack', 2e-2)

    qp = ccq.CbfClfQp(ds, QPoption)

    T = 1
    dt = .05
    
    time_steps = int(np.ceil(T / dt))

    ode_sol = ode_solver.OdeSolver(ds, dt)

    # initialize the input matrices
    xt = np.zeros((4, time_steps))
    tt = np.zeros((1, time_steps))
    ut = np.zeros((2, time_steps))

    slackt = np.zeros((1, time_steps))
    Vt = np.zeros((1, time_steps))
    Bt = np.zeros((1, time_steps))
    xt[:, 0] = np.copy(x0.T[0])

    for t in range(time_steps-1):
        

        # solve for control u at current time step
        u, delta, B, V, feas = qp.cbf_clf_qp(xt[:, t])
        if feas == -1:
            # infeasible
            break
        else:
            pass

        ut[:, t] = np.copy(u)
        slackt[:, t] = np.copy(delta)
        Vt[:, t] = np.copy(V)
        Bt[:, t] = np.copy(B)

        # propagate the system with control u using RK4
        xt[:, t + 1] = ode_sol.time_marching(xt[:, t], u)


    return (xt, ut, dt, T)


def show_three_plots(params, xt, ut, dt, T, show=1, save=0):
    t = np.arange(0, T, dt)
    
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- Plot 1: State Trajectory ---
    ax1.set_aspect('equal')  # Ensures the plot is not distorted
    ax1.grid()
    ax1.plot(xt[0, :], xt[1, :], linewidth=3, color='magenta', label="Trajectory")
    
    # Add obstacle and goal
    obs = plt.Circle((params['obstacle']['xo'], params['obstacle']['yo']), params['obstacle']['rsafe'], color='red', fill=False, linewidth=2, linestyle='dashed')
    # goal = plt.Circle((dynamics_params['xg'], dynamics_params['yg']), 1, color='blue', fill=False, linewidth=2, linestyle='dashed')
    ax1.add_patch(obs)
    # ax1.add_patch(goal)
    
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
    
    # Show or save figure
    if show == 1:
        plt.show()
    if save == 1:
        fig.savefig('combined_plots_with_control.png', format='png', dpi=300)
    
    plt.close(fig)



params = {
    'obstacle': {
        'xo':2,
        'yo':2,
        'rsafe':2.1
    },
    'speed':{
        'target_speed': 3
    }
}

xt, ut, dt, T = get_control(np.array([[0, 0, 0, 2]]).T, params)
# Call the function to display all three plots together
show_three_plots(params, xt, ut, dt, T)




