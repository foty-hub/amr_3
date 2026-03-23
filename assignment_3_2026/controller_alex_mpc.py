# wind_flag = False
# Implement a controller
# %%
from functools import cache

import numpy as np
import qpsolvers

dt = 0.02


class MPCController:
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        # define the model

        I = np.eye(4)
        C = np.array([1, 0])  # only emit
        self.C = np.kron(C, I)
        return

    @staticmethod
    @cache  # cache so we're not recomputing unnecessarily
    def _get_A(dt: float):
        """Encodes p_kt = p_k + dt * v_k. This is a function so we can handle changing dt"""
        A = np.array(
            [
                [1, dt],
                [0, 1],
            ]
        )
        return np.kron(A, np.eye(4))

    @staticmethod
    @cache
    def _get_B(dt: float):
        B = np.array(
            [
                [0],
                [dt],
            ]
        )
        return np.kron(B, np.eye(4))

    def kalman(self, state: np.ndarray) -> np.ndarray:
        return state

    def mpc(self, x: np.ndarray, t: np.ndarray, dt: np.ndarray) -> np.ndarray: ...

    def __call__(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: np.ndarray,
    ) -> np.ndarray:
        state_estimate = self.kalman(state)
        control = self.mpc(state_estimate, target, dt)
        return control


def get_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


mpc_controller = MPCController()
# %%


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

    output = None
    return output
