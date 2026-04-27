# wind_flag = False
# Implement a controller
from collections import deque
import csv
import time
from pathlib import Path

import numpy as np


def save_data(
    dt: float,
    state: np.ndarray,
    target: np.ndarray,
    control_output: np.ndarray,
    wind_enabled: bool,
    output_file: str,
):
    row = dict(
        recording_time_ns=time.time_ns(),
        dt=dt,
        wind_enabled=wind_enabled,
        pos_x=state[0],
        pos_y=state[1],
        pos_z=state[2],
        pos_roll=state[3],
        pos_pitch=state[4],
        pos_yaw=state[5],
        targetpos_x=target[0],
        targetpos_y=target[1],
        targetpos_z=target[2],
        targetpos_yaw=target[3],
        control_vel_x=control_output[0],
        control_vel_y=control_output[1],
        control_vel_z=control_output[2],
        control_vel_yaw=control_output[3],
    )

    file_exists = Path(output_file).is_file()

    with open(output_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class PIDController:
    def __init__(self):
        self.integral = np.zeros(4)
        self.prev_err = np.zeros(4)
        self.history = deque(maxlen=100)
        self.last_target = None
        
        self.KGain = 0.6
        self.IGain = 0.0001
        self.DGain = 0.3
        self.Kp = np.array([self.KGain,self.KGain,self.KGain, 0.6])
        self.Ki = np.array([self.IGain,self.IGain,0,0])
        self.Kd = np.array([self.DGain,self.DGain, 0.00, 0.0])
        self.prev_dt = time.time()

    def __call__(self, state, target_pos, dt):
        if self._target_changed(target_pos):
            self.reset()
        self.last_target = np.array(target_pos, copy=True)

        if dt > 1e5:
            temp = dt / pow(10,3)
            dt = (dt - self.prev_dt) /  pow(10,3)
            self.prev_dt = temp
            
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
    state_saved = np.copy(state)
    state = np.delete(state, [3, 4])  # ignore roll/pitch
    control = pid_controller(np.array(state), np.array(target_pos), dt)

    # Convert from world frame to robot frame - rotate x,y
    theta = state[3]
    control[0:2] = get_rot_matrix(theta) @ control[0:2]

    #output = (control[0], control[1], control[2], control[3])
    output = (
                    np.clip(control[0], -0.3, 0.3),
                    np.clip(control[1], -0.3, 0.3),
                    np.clip(control[2], -0.3, 0.3),
                    np.clip(control[3], -1.74533, 1.74533),
    )
    
    save_data(dt, state_saved, target_pos, output, wind_enabled, 'data.csv')

    return output
