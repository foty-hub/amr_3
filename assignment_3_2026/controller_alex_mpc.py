# wind_flag = False
# Implement a controller
import numpy as np


class MPCController:
    def __init__(self):
        # define the model

        return None

    def __call__(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: np.ndarray,
    ) -> np.ndarray:
        return state


def get_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


mpc_controller = MPCController()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    state = np.array(state)
    target = np.array(target_pos)

    state = np.delete(state, [3, 4])  # ignore roll/pitch
    control = mpc_controller(np.array(state), np.array(target), dt)

    # transform
    theta = state[3]
    control[0:2] = get_rot_matrix(theta) @ control[0:2]

    return output
