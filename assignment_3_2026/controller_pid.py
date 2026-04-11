# wind_flag = False
# Implement a controller
from collections import deque

import numpy as np


class PIDController:
    def __init__(self):
        self.integral = np.zeros(4)
        self.prev_err = np.zeros(4)
        self.history = deque(maxlen=100)
        self.last_target = None

        self.Kp = np.array([0.25, 0.25, 0.25, 0.2])
        self.Ki = np.array([0.05, 0.05, 0.00, 0.0])
        self.Kd = np.array([0.01, 0.01, 0.00, 0.0])

    def __call__(self, state, target_pos, dt):
        if self._target_changed(target_pos):
            self.reset()
        self.last_target = np.array(target_pos, copy=True)

        err = target_pos - state
        # wrap the angular error into [-pi, +pi]
        err[3] = (err[3] + np.pi) % (2 * np.pi) - np.pi

        self.history.append(err * dt)
        derivative = (err - self.prev_err) / dt
        self.prev_err = err

        return self.Kp * err + self.Ki * sum(self.history) + self.Kd * derivative

    def reset(self):
        self.integral = np.zeros(4)
        self.prev_err = np.zeros(4)
        self.history.clear()
        self.last_target = None

    def _target_changed(self, target_pos):
        if self.last_target is None:
            return False

        same_position = np.allclose(target_pos[:3], self.last_target[:3], atol=1e-6)
        same_yaw = (
            abs((target_pos[3] - self.last_target[3] + np.pi) % (2 * np.pi) - np.pi)
            <= 1e-6
        )
        return not (same_position and same_yaw)


def get_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


pid_controller = PIDController()


def reset():
    pid_controller.reset()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    state = np.delete(state, [3, 4])  # ignore roll/pitch
    control = pid_controller(np.array(state), np.array(target_pos), dt)

    # Convert from world frame to robot frame - rotate x,y
    theta = state[3]
    control[0:2] = get_rot_matrix(theta) @ control[0:2]

    output = (control[0], control[1], control[2], control[3])
    return output
