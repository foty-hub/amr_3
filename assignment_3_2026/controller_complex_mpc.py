# wind_flag = False
# Implement a controller

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from qpsolvers import solve_qp

YAW_RATE_LIMIT = 1.74533


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_body_to_world(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_world_to_body(yaw):
    return rotation_body_to_world(yaw).T


def load_vehicle_params():
    urdf_path = Path(__file__).resolve().parent / "resources" / "tello.urdf"
    default_mass = 0.088
    default_izz = 0.01313

    try:
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue

            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if mass is None or inertia is None:
                continue

            mass_value = float(mass.attrib["value"])
            izz_value = float(inertia.attrib["izz"])
            if mass_value > 0.0 and izz_value > 0.0:
                return mass_value, izz_value
    except (ET.ParseError, OSError, KeyError, ValueError):
        pass

    return default_mass, default_izz


class ComplexMPCController:
    def __init__(self, horizon: int = 16):
        self.horizon = horizon
        self.state_dim = 8
        self.control_dim = 4
        self.M = horizon

        self.mass, self.izz = load_vehicle_params()
        self.k_trans = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.motor_time_constant = 0.0163
        self.vel_kp = 7.0
        self.vel_kd = 0.2
        self.rate_kp = 0.05

        self.position_error_slice = slice(0, 3)
        self.velocity_slice = slice(3, 6)
        self.yaw_error_index = 6
        self.yaw_rate_index = 7

        self.velocity_decay, self.velocity_static_gain = self._derive_velocity_model()
        self.yaw_decay, self.yaw_static_gain = self._derive_yaw_model()

        self.stage_weights = np.diag([18.0, 18.0, 24.0, 10.0, 10.0, 12.0, 16.0, 5.0])
        self.terminal_weights = np.diag(
            [32.0, 32.0, 40.0, 16.0, 16.0, 20.0, 24.0, 8.0]
        )
        self.Q_bar = self._build_state_cost()

        self.delta_weights = np.array([14.76, 14.76, 18.04, 3.28])
        self.absolute_control_weights = np.array([6.24, 6.24, 7.02, 1.404])
        self.bias_tracking_weights = np.array([16.1, 16.1, 18.4, 4.025])
        self.W_delta = np.kron(np.eye(self.M), np.diag(self.delta_weights))
        self.W_control = np.kron(
            np.eye(self.M), np.diag(self.absolute_control_weights)
        )
        self.W_bias = np.kron(
            np.eye(self.M), np.diag(self.bias_tracking_weights)
        )
        self.D = self._build_delta_matrix()

        yaw_limit = min(0.90, YAW_RATE_LIMIT)
        self.control_lb = np.array([-0.29, -0.29, -0.32, -yaw_limit])
        self.control_ub = np.array([0.29, 0.29, 0.32, yaw_limit])
        self.lb = np.tile(self.control_lb, self.M)
        self.ub = np.tile(self.control_ub, self.M)

        self.bias_gains = np.array([0.3796, 0.3796, 0.4672, 0.8030])
        self.damping_gains = np.array([0.5376, 0.5376, 0.6144, 0.2816])
        self.measurement_blend = 0.52

        self.control = np.zeros(self.control_dim)
        self.prev_measurement_position = None
        self.prev_measurement_yaw = None
        self.filtered_body_velocity = np.zeros(3)
        self.filtered_yaw_rate = 0.0
        self.last_target = None

    def reset(self):
        self.control = np.zeros(self.control_dim)
        self.prev_measurement_position = None
        self.prev_measurement_yaw = None
        self.filtered_body_velocity = np.zeros(3)
        self.filtered_yaw_rate = 0.0
        self.last_target = None

    def _derive_velocity_model(self):
        drag_accel = self.k_trans / self.mass
        tau_raw = (1.0 + self.vel_kd) / (self.vel_kp + drag_accel)
        tau_total = tau_raw + self.motor_time_constant
        decay = 1.0 / tau_total
        static_gain = self.vel_kp / (self.vel_kp + drag_accel)
        return decay, static_gain

    def _derive_yaw_model(self):
        tau_total = self.izz / self.rate_kp + self.motor_time_constant
        decay = 1.0 / tau_total
        return decay, 1.0

    def _build_state_cost(self):
        blocks = [self.stage_weights.copy() for _ in range(self.M)]
        blocks[-1] = self.terminal_weights.copy()
        return np.block(
            [
                [blocks[i] if i == j else np.zeros_like(blocks[0]) for j in range(self.M)]
                for i in range(self.M)
            ]
        )

    def _build_delta_matrix(self):
        D = np.zeros((self.control_dim * self.M, self.control_dim * self.M))
        eye = np.eye(self.control_dim)
        for step in range(self.M):
            row = slice(step * self.control_dim, (step + 1) * self.control_dim)
            col = slice(step * self.control_dim, (step + 1) * self.control_dim)
            D[row, col] = eye
            if step > 0:
                prev = slice(
                    (step - 1) * self.control_dim, step * self.control_dim
                )
                D[row, prev] = -eye
        return D

    def _discretize_first_order(self, decay, static_gain, dt):
        decay = np.asarray(decay, dtype=float)
        static_gain = np.asarray(static_gain, dtype=float)
        a = np.exp(-decay * dt)
        b = (1.0 - a) * static_gain
        return a, b

    def _build_model_matrices(self, yaw, dt):
        dt = max(float(dt), 1e-3)
        rot = rotation_body_to_world(yaw)
        vel_a, vel_b = self._discretize_first_order(
            self.velocity_decay, self.velocity_static_gain, dt
        )
        yaw_a, yaw_b = self._discretize_first_order(
            self.yaw_decay, self.yaw_static_gain, dt
        )

        A = np.eye(self.state_dim)
        A[self.position_error_slice, self.velocity_slice] = -dt * rot
        A[self.velocity_slice, self.velocity_slice] = np.diag(vel_a)
        A[self.yaw_error_index, self.yaw_rate_index] = -dt
        A[self.yaw_rate_index, self.yaw_rate_index] = float(yaw_a)

        B = np.zeros((self.state_dim, self.control_dim))
        B[self.velocity_slice, 0:3] = np.diag(vel_b)
        B[self.yaw_rate_index, 3] = float(yaw_b)
        return A, B

    def _build_measurement(self, reduced_state, target, dt):
        position = reduced_state[:3]
        yaw = float(reduced_state[3])
        safe_dt = max(float(dt), 1e-3)

        if self.prev_measurement_position is None or self.prev_measurement_yaw is None:
            raw_body_velocity = np.zeros(3)
            raw_yaw_rate = 0.0
        else:
            world_velocity = (position - self.prev_measurement_position) / safe_dt
            raw_body_velocity = rotation_world_to_body(yaw) @ world_velocity
            raw_yaw_rate = wrap_angle(yaw - self.prev_measurement_yaw) / safe_dt

        self.prev_measurement_position = position.copy()
        self.prev_measurement_yaw = yaw

        blend = self.measurement_blend
        self.filtered_body_velocity = (
            blend * raw_body_velocity + (1.0 - blend) * self.filtered_body_velocity
        )
        self.filtered_yaw_rate = float(
            blend * raw_yaw_rate + (1.0 - blend) * self.filtered_yaw_rate
        )

        body_velocity = np.clip(self.filtered_body_velocity, -1.5, 1.5)
        yaw_rate = float(np.clip(self.filtered_yaw_rate, -2.5, 2.5))
        position_error_world = target[:3] - position
        yaw_error = wrap_angle(target[3] - yaw)

        state = np.zeros(self.state_dim)
        state[self.position_error_slice] = position_error_world
        state[self.velocity_slice] = body_velocity
        state[self.yaw_error_index] = yaw_error
        state[self.yaw_rate_index] = yaw_rate
        return state

    def _build_prediction_matrices(self, A, B):
        Lambda = np.zeros((self.state_dim * self.M, self.state_dim))
        Phi = np.zeros((self.state_dim * self.M, self.control_dim * self.M))

        powers = [np.eye(self.state_dim)]
        for _ in range(self.M):
            powers.append(powers[-1] @ A)

        for row_step in range(self.M):
            row = slice(row_step * self.state_dim, (row_step + 1) * self.state_dim)
            Lambda[row] = powers[row_step + 1]
            for col_step in range(row_step + 1):
                col = slice(
                    col_step * self.control_dim, (col_step + 1) * self.control_dim
                )
                Phi[row, col] = powers[row_step - col_step] @ B

        return Lambda, Phi

    def _delta_offset(self):
        offset = np.zeros(self.control_dim * self.M)
        offset[: self.control_dim] = self.control
        return offset

    def _bias_control(self, x_hat, yaw):
        position_error_body = rotation_world_to_body(yaw) @ x_hat[self.position_error_slice]
        control = self.bias_gains * np.array(
            [
                position_error_body[0],
                position_error_body[1],
                position_error_body[2],
                x_hat[self.yaw_error_index],
            ]
        ) - self.damping_gains * np.array(
            [
                x_hat[self.velocity_slice.start + 0],
                x_hat[self.velocity_slice.start + 1],
                x_hat[self.velocity_slice.start + 2],
                x_hat[self.yaw_rate_index],
            ]
        )
        return np.clip(control, self.control_lb, self.control_ub)

    def _build_qp(self, x_hat, yaw, dt):
        A, B = self._build_model_matrices(yaw, dt)
        Lambda, Phi = self._build_prediction_matrices(A, B)
        tracking_error = Lambda @ x_hat
        delta_offset = self._delta_offset()
        bias_control = self._bias_control(x_hat, yaw)
        bias_stack = np.tile(bias_control, self.M)

        hessian = (
            Phi.T @ self.Q_bar @ Phi
            + self.D.T @ self.W_delta @ self.D
            + self.W_control
            + self.W_bias
        )
        gradient = (
            Phi.T @ self.Q_bar @ tracking_error
            - self.D.T @ self.W_delta @ delta_offset
            - self.W_bias @ bias_stack
        )

        P_qp = 2.0 * hessian + 1e-8 * np.eye(self.control_dim * self.M)
        P_qp = 0.5 * (P_qp + P_qp.T)
        q_qp = 2.0 * gradient
        return P_qp, q_qp

    def _fallback_control(self, x_hat, yaw):
        return self._bias_control(x_hat, yaw)

    def mpc(self, x_hat, yaw, dt):
        P_qp, q_qp = self._build_qp(x_hat, yaw, dt)
        u_opt = solve_qp(P_qp, q_qp, lb=self.lb, ub=self.ub, solver="cvxopt")
        if u_opt is None:
            return self._fallback_control(x_hat, yaw)
        return np.asarray(u_opt[: self.control_dim], dtype=float)

    def __call__(self, reduced_state, target, dt):
        reduced_state = np.asarray(reduced_state, dtype=float)
        target = np.asarray(target, dtype=float)

        yaw = float(reduced_state[3])
        measurement = self._build_measurement(reduced_state, target, dt)
        self.last_target = target.copy()
        self.control = self.mpc(measurement, yaw, dt)
        return self.control.copy()


complex_mpc_controller = ComplexMPCController()


def reset():
    complex_mpc_controller.reset()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    del wind_enabled

    state = np.asarray(state, dtype=float)
    target = np.asarray(target_pos, dtype=float)

    reduced_state = np.delete(state, [3, 4])
    control_body = complex_mpc_controller(reduced_state, target, dt)
    return tuple(control_body.tolist())
