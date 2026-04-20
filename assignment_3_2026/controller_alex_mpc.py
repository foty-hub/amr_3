# wind_flag = False
# Implement a controller

import numpy as np
from qpsolvers import solve_qp
from save_data import save_data

YAW_RATE_LIMIT = 1.74533
CONTROL_LIMIT = 1.0
HORIZON = 15
REGULARISATION_STRENGTH = 3.0
OUTPUT_FILE = "data1.csv"


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MPCController:
    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.state_dim = 4
        self.control_dim = 4
        self.A = np.eye(self.state_dim)
        self.C = np.eye(self.state_dim)
        self.M = horizon

        self.state_weights = np.array([2.0, 2.0, 2.5, 2.5])
        self.delta_weights = np.array([16.0, 16.0, 16.0, 3.0])
        self.control_regularization_weights = (
            np.array([0.40, 0.40, 0.40, 0.07]) * REGULARISATION_STRENGTH
        )

        yaw_limit = min(0.35, YAW_RATE_LIMIT)
        self.control_lb = np.array(
            [-CONTROL_LIMIT, -CONTROL_LIMIT, -CONTROL_LIMIT, -yaw_limit]
        )
        self.control_ub = np.array(
            [CONTROL_LIMIT, CONTROL_LIMIT, CONTROL_LIMIT, yaw_limit]
        )

        self.Lambda = self._build_lambda()
        self.Q_bar = np.kron(np.eye(self.M), np.diag(self.state_weights))
        self.W_delta = np.kron(np.eye(self.M), np.diag(self.delta_weights))
        self.W_control = np.kron(
            np.eye(self.M), np.diag(self.control_regularization_weights)
        )
        self.D = self._build_delta_matrix()
        self.lb = np.tile(self.control_lb, self.M)
        self.ub = np.tile(self.control_ub, self.M)

        self.P = np.eye(self.state_dim)
        self.Q_kf = np.diag([8e-4, 8e-4, 8e-4, 1.6e-3])
        self.R_kf = np.diag([1e-4, 1e-4, 1e-4, 2e-4])

        self.state_estimate = np.zeros(self.state_dim)
        self.control = np.zeros(self.control_dim)
        self.last_target = None

    def reset(self):
        self.P = np.eye(self.state_dim)
        self.state_estimate = np.zeros(self.state_dim)
        self.control = np.zeros(self.control_dim)
        self.last_target = None

    def _target_changed(self, target: np.ndarray) -> bool:
        if self.last_target is None:
            return False

        same_position = np.allclose(target[:3], self.last_target[:3], atol=1e-6)
        same_yaw = abs(wrap_angle(target[3] - self.last_target[3])) <= 1e-6
        return not (same_position and same_yaw)

    def _build_lambda(self) -> np.ndarray:
        return np.vstack(
            [
                self.C @ np.linalg.matrix_power(self.A, step + 1)
                for step in range(self.M)
            ]
        )

    def _build_delta_matrix(self) -> np.ndarray:
        D = np.zeros((self.control_dim * self.M, self.control_dim * self.M))
        eye = np.eye(self.control_dim)
        for step in range(self.M):
            row = slice(step * self.control_dim, (step + 1) * self.control_dim)
            col = slice(step * self.control_dim, (step + 1) * self.control_dim)
            D[row, col] = eye
            if step > 0:
                prev = slice((step - 1) * self.control_dim, step * self.control_dim)
                D[row, prev] = -eye
        return D

    def _get_B(self, dt: float) -> np.ndarray:
        return dt * np.eye(self.control_dim)

    def _get_Phi(self, B: np.ndarray) -> np.ndarray:
        Phi = np.zeros((self.state_dim * self.M, self.control_dim * self.M))
        for row_step in range(self.M):
            row = slice(row_step * self.state_dim, (row_step + 1) * self.state_dim)
            for col_step in range(row_step + 1):
                col = slice(
                    col_step * self.control_dim, (col_step + 1) * self.control_dim
                )
                power = row_step - col_step
                Phi[row, col] = self.C @ np.linalg.matrix_power(self.A, power) @ B
        return Phi

    def _stack_reference(self, x_hat: np.ndarray, target: np.ndarray) -> np.ndarray:
        reference = np.asarray(target, dtype=float).copy()
        reference[3] = x_hat[3] + wrap_angle(reference[3] - x_hat[3])
        return np.tile(reference, self.M)

    def _delta_offset(self) -> np.ndarray:
        offset = np.zeros(self.control_dim * self.M)
        offset[: self.control_dim] = self.control
        return offset

    def kalman_filter(
        self,
        x_hat: np.ndarray,
        B: np.ndarray,
        u_prev: np.ndarray,
        measurement: np.ndarray,
    ) -> np.ndarray:
        x_prior = self.A @ x_hat + B @ u_prev
        P_prior = self.A @ self.P @ self.A.T + self.Q_kf
        y_pred = self.C @ x_prior

        residual = np.asarray(measurement, dtype=float) - y_pred
        residual[3] = wrap_angle(residual[3])

        innovation_cov = self.C @ P_prior @ self.C.T + self.R_kf
        kalman_gain = P_prior @ self.C.T @ np.linalg.inv(innovation_cov)

        x_post = x_prior + kalman_gain @ residual
        x_post[3] = wrap_angle(x_post[3])
        self.P = (np.eye(self.state_dim) - kalman_gain @ self.C) @ P_prior
        self.P = 0.5 * (self.P + self.P.T)
        return x_post

    def _build_qp(
        self, x_hat: np.ndarray, target: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        B = self._get_B(dt)
        Phi = self._get_Phi(B)
        reference = self._stack_reference(x_hat, target)
        tracking_error = self.Lambda @ x_hat - reference
        delta_offset = self._delta_offset()

        hessian = (
            Phi.T @ self.Q_bar @ Phi + self.D.T @ self.W_delta @ self.D + self.W_control
        )
        gradient = (
            Phi.T @ self.Q_bar @ tracking_error - self.D.T @ self.W_delta @ delta_offset
        )

        P_qp = 2.0 * hessian + 1e-8 * np.eye(self.control_dim * self.M)
        P_qp = 0.5 * (P_qp + P_qp.T)
        q_qp = 2.0 * gradient
        return P_qp, q_qp

    def mpc(self, x_hat: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        P_qp, q_qp = self._build_qp(x_hat, target, dt)
        u_opt = solve_qp(P_qp, q_qp, lb=self.lb, ub=self.ub, solver="cvxopt")
        if u_opt is None:
            return self.control.copy()

        return np.asarray(u_opt[: self.control_dim], dtype=float)

    def __call__(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        measurement = np.asarray(state, dtype=float)
        target = np.asarray(target, dtype=float)

        if self._target_changed(target):
            self.reset()
        self.last_target = target.copy()

        B = self._get_B(dt)
        self.state_estimate = self.kalman_filter(
            self.state_estimate,
            B,
            self.control,
            measurement,
        )

        self.control = self.mpc(self.state_estimate, target, dt)
        return self.control.copy()


def get_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


mpc_controller = MPCController()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    state = np.asarray(state, dtype=float)
    target = np.asarray(target_pos, dtype=float)

    state_trimmed = np.delete(state, [3, 4])
    control = mpc_controller(state_trimmed, target, dt)

    theta = state_trimmed[3]
    control[0:2] = get_rot_matrix(theta) @ control[0:2]

    save_data(dt, state, target_pos, control, wind_enabled, OUTPUT_FILE)

    return tuple(control.tolist())
