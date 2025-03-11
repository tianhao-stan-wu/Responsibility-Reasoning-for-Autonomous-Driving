import numpy as np
import kinematic_bicycle as kb
import define_system
import cbf_clf_qp as ccq
import ode_solver


def get_control(x0, params, u_ref):

    """
    x0 = np.array([[x, y, theta, v]]).T
    """

    # todo: add reference control u to solver

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

    def to_radian(degree):
        return degree * np.pi / 180.0

    QPoption = ccq.CbfClfQpOptions()
    QPoption.set_option('u_max', np.array([9, to_radian(70)]))
    QPoption.set_option('u_min', np.array([-19, -1 * to_radian(70)]))
    QPoption.set_option('clf_lambda', 100.0)
    QPoption.set_option('cbf_gamma', 2.5)
    QPoption.set_option('weight_input', np.array([1.0]))
    QPoption.set_option('weight_slack', 2e-2)

    qp = ccq.CbfClfQp(ds, QPoption)

    T = 0.5
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

    return ut, time_steps



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

ut, time_steps = get_control(np.array([[0, 0, 0, 2]]).T, params)
for t in range(time_steps):
    print(ut[:,t])

