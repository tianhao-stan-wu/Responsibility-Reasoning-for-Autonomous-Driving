import math


def steer_to_beta(steer):

    # steer is between [-1, 1] to [-70, 70] degrees
    steer = math.radians(steer*70)
    # kinematic bicycle model
    beta = math.atan((2.102/(2.102+1.553)) * math.tan(steer))

    return beta


def beta_to_steer(beta):

    steer = math.atan(math.tan(beta)*((2.102+1.553)/2.102))

    steer = math.degrees(steer) / 70

    return steer


def get_throttle_brake(acc):

    # throttle brake guaranteed to be within [0,1] by control bounds in qp
    if acc >= 0:
        throttle = acc/12
        brake = 0
    else:
        throttle = 0
        brake = acc/(-1 * 16)

    return throttle, brake


def get_acc(throttle, brake):

    if throttle > 0:
        return throttle * 12
    elif brake > 0:
        return brake * -16
    else:
        return 0