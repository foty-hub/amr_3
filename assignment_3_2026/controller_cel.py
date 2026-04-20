import numpy as np

Kp = np.array([1.8, 1.8, 2.4])
Kd = np.array([0.6, 0.6, 0.9])
Kp_yaw = 1.6

_prev_error = np.zeros(3)
_prev_target = None


def controller(state, target_pos, dt, wind_enabled=False):

    global _prev_error, _prev_target
    pos_x, pos_y, pos_z, roll, pitch, yaw = state
    tar_x, tar_y, tar_z, tar_yaw = target_pos
    pos = np.array([pos_x, pos_y, pos_z])
    target = np.array([tar_x, tar_y, tar_z])
    if _prev_target is None or np.linalg.norm(target - _prev_target) > 1e-4:
        _prev_error = np.zeros(3)
    _prev_target = target.copy()
    error = target - pos
    dist = np.linalg.norm(error)

    if dist < 0.08:
        return 0.0, 0.0, 0.0, 0.0

    derivative = (error - _prev_error) / dt
    _prev_error = error

    if dist < 0.25:
        derivative *= 0.6 + 0.4 * (dist / 0.25)

    vel = Kp * error + Kd * derivative

    if dist < 0.5:
        vel *= dist / 0.5

    vel = np.clip(vel, -1.2, 1.2)

    if wind_enabled:
        vel *= 0.95

    vx_w, vy_w, vz = vel
    vx =  np.cos(yaw) * vx_w + np.sin(yaw) * vy_w
    vy = -np.sin(yaw) * vx_w + np.cos(yaw) * vy_w
    yaw_error = tar_yaw - yaw
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

    if abs(yaw_error) < 0.04:
        yaw_rate = 0.0
    else:
        yaw_rate = np.clip(Kp_yaw * yaw_error, -1.0, 1.0)

    return float(vx), float(vy), float(vz), float(yaw_rate)