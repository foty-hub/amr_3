# wind_flag = False
# This controller implements MPC - a form of control which uses a predictive model of the form:
#  x_{t+1} = A x_t + B u_t
#  y_t     = C x_t + D u_t
# In this controller we're using a very simplistic model, operating only on the state but not on the velocities
# x = [x, y, z, yaw]  (note does not include the velocity terms x_dot, y_dot etc...)
# A = I_{4x4}         (4x4 identity)
# B = dt * I_{4x4}    (assumes that the velocity setpoint is instantly achieved)
# C = I_{4x4}
# D = 0_{4x4}
#
# In addition to MPC, this controller implements a Kalman filter to smooth observations.
# This has no effect on the sim, but may be useful for the real hardware loop with noisy measurements.

import numpy as np
from qpsolvers import solve_qp
from save_data import save_data

# These are the limits placed on the sim - but note that
# the real-hardware limits are clipped to [-0.3, 0.3]
POS_CONTROL_LIMIT = 1.0
YAW_CONTROL_LIMIT = 1.74533
HORIZON = 15  # How many steps forwards to consider in the MPC optimisation
REGULARISATION_STRENGTH = 3.0  # multiplier to tweak regularisation strentch quickly
OUTPUT_FILE = "data.csv"


def wrap_angle(angle: float) -> float:
    """Converts an angle to the range [-pi, +pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MPCController:
    def __init__(self, horizon: int = HORIZON):
        """Sets up the MPC controller and precomputes all possible matrices. Note that it is typical
        to precompute the Hessian, but as the hardware system sees variable-length timesteps we instead
        compute it per-call.

        In addition to MPC, this controller also applies a Kalman filter to observations, operating on
        smoothed state estimates."""
        self.horizon = horizon
        self.state_dim = 4
        self.control_dim = 4

        # Set up the base matrices for the system. We assume a simplified state representation without
        # velocity or inertia considerations. That limits
        self.A = np.eye(self.state_dim)
        self.C = np.eye(self.state_dim)
        self.M = horizon

        # These weights define the constrained optimisation MPC problem
        #  - The state weights penalise states which are far from the target position
        #  - The delta weights penalise rapid changes in the control signal
        #  - The control regularisation penalises large control signals. This is set lower than
        #     the other weights so that, all other things equal, the controller prefers
        #     a trajectory with lower rotor speeds
        self.state_weights = np.array([2.0, 2.0, 2.0, 2.5])
        self.delta_weights = np.array([16.0, 16.0, 16.0, 3.0])
        self.control_regularization_weights = (
            np.array([0.40, 0.40, 0.40, 0.07]) * REGULARISATION_STRENGTH
        )

        # Because we're doing linear MPC, which just uses a quadratic solver, we can
        # bound the valid control signals - these variables set those bounds
        self.control_ub = np.array(
            [POS_CONTROL_LIMIT, POS_CONTROL_LIMIT, POS_CONTROL_LIMIT, YAW_CONTROL_LIMIT]
        )
        self.control_lb = -self.control_ub

        # Build the matrices used for MPC. We use np.kron to construct block matrices
        # Because MPC unrolls state prediction over a horizon, all the matrices are expanded to be
        # of shape (control_dim * horizon) - hence the block structure
        self.Lambda = self._build_lambda()
        self.Q_bar = np.kron(np.eye(self.M), np.diag(self.state_weights))
        self.W_delta = np.kron(np.eye(self.M), np.diag(self.delta_weights))
        self.W_control = np.kron(
            np.eye(self.M), np.diag(self.control_regularization_weights)
        )
        self.D = self._build_delta_matrix()
        self.lb = np.tile(self.control_lb, self.M)
        self.ub = np.tile(self.control_ub, self.M)

        # In addition to MPC, we implement a Kalman filter to denoise real state observations on
        # the hardware - these are the process matrices for the Kalman filter.
        self.P = np.eye(self.state_dim)
        self.Q_kf = np.diag([8e-4, 8e-4, 8e-4, 1.6e-3])
        self.R_kf = np.diag([1e-4, 1e-4, 1e-4, 2e-4])
        self.state_estimate = np.zeros(self.state_dim)

        self.control = np.zeros(self.control_dim)

    def _build_lambda(self) -> np.ndarray:
        """Builds the lambda vector, which encodes how the state would unroll with no control input"""
        return np.vstack(
            [
                self.C @ np.linalg.matrix_power(self.A, step + 1)
                for step in range(self.M)
            ]
        )

    def _build_delta_matrix(self) -> np.ndarray:
        """Constructs the block-differencing matrix D used to penalise the rate of change of control inputs."""
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
        """Computes the discrete-time input matrix B, assuming linear single-integrator kinematics over dt."""
        return dt * np.eye(self.control_dim)

    def _get_Phi(self, B: np.ndarray) -> np.ndarray:
        """Constructs the forced response convolution matrix Phi, which maps future control sequences to future states."""
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
        """Stacks the reference over the length of the horizon"""
        reference = target.copy()
        # cheeky trick so that reference will be wrapped around to the positive/negative angle
        # that's closest to x_hat -> so it won't overrotate to achieve the target yaw
        reference[3] = x_hat[3] + wrap_angle(reference[3] - x_hat[3])
        # Repeat the reference over the length of the horizon
        return np.tile(reference, self.M)

    def _delta_offset(self) -> np.ndarray:
        # Compute the delta_u values across the horizon
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
        """Implements a basic Kalman filter to smooth state measurements."""
        x_prior = self.A @ x_hat + B @ u_prev
        P_prior = self.A @ self.P @ self.A.T + self.Q_kf
        y_pred = self.C @ x_prior

        residual = measurement - y_pred
        residual[3] = wrap_angle(residual[3])  # ensures yaw is in [-pi, pi]

        cov = self.C @ P_prior @ self.C.T + self.R_kf
        kalman_gain = P_prior @ self.C.T @ np.linalg.inv(cov)

        x_post = x_prior + kalman_gain @ residual
        x_post[3] = wrap_angle(x_post[3])
        self.P = (np.eye(self.state_dim) - kalman_gain @ self.C) @ P_prior
        # For numerical stability, ensure P is symmetric
        self.P = 0.5 * (self.P + self.P.T)
        return x_post

    def _build_qp(
        self, x_hat: np.ndarray, target: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        # Get B and Phi. Because we want to accept varying timesteps
        # we need to recompute these at each time step
        B = self._get_B(dt)
        Phi = self._get_Phi(B)

        # We assume a constant reference over the course of the trajectory. This is a
        # reasonable since the target only undergoes step changes
        reference = self._stack_reference(x_hat, target)
        tracking_error = self.Lambda @ x_hat - reference
        delta_offset = self._delta_offset()

        # Compute the Hessian and gradient for the linear MPC quadratic problem
        hessian = (
            Phi.T @ self.Q_bar @ Phi + self.D.T @ self.W_delta @ self.D + self.W_control
        )
        gradient = (
            Phi.T @ self.Q_bar @ tracking_error - self.D.T @ self.W_delta @ delta_offset
        )

        # Extra tricks for numerical stability: Add an epsilon along the diagonal (so eigenvalues are nonzero)
        # and symmetrise so P_qp is guaranteed positive semidefinite for the solver.
        P_qp = 2.0 * hessian + 1e-8 * np.eye(self.control_dim * self.M)
        P_qp = 0.5 * (P_qp + P_qp.T)
        q_qp = 2.0 * gradient
        return P_qp, q_qp

    def mpc(self, x_hat: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        P_qp, q_qp = self._build_qp(x_hat, target, dt)
        # Solve the constrained optimisation using qpsolvers to find the optimal trajectory u_opt.
        # u_opt should never be None given the conditioning in _build_qp, but just in case
        # we add a guard to prevent crashes
        u_opt = solve_qp(P_qp, q_qp, lb=self.lb, ub=self.ub, solver="cvxopt")
        if u_opt is None:
            return self.control.copy()

        # Return only the first step of the optimal trajectory - this is the classic MPC approach.
        return np.asarray(u_opt[: self.control_dim], dtype=float)

    def __call__(
        self,
        measurement: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        B = self._get_B(dt)
        # Pass the state measurement through a Kalman filter to smooth out noise
        self.state_estimate = self.kalman_filter(
            self.state_estimate,
            B,
            self.control,
            measurement,
        )

        # Perform MPC to get the control at the next step
        self.control = self.mpc(self.state_estimate, target, dt)
        # make a copy in case the control is modified elsewhere
        return self.control.copy()


def get_rot_matrix(theta):
    "Returns a 2x2 matrix representing the 2D rotation matrix for a rotation around the z-axis."
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


# Instantiate the controller outside the function so it persists state across calls.
mpc_controller = MPCController()


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))

    # Convert everything to numpy arrays for convenience/faster computation
    state = np.asarray(state, dtype=float)
    target = np.asarray(target_pos, dtype=float)

    state_trimmed = np.delete(state, [3, 4])
    control = mpc_controller(state_trimmed, target, dt)

    # We get the state and target_pos in world coordinates, but the control is in the drone's
    # local coordinate system. We're told we can ignore pitch and yaw, so we can simply
    # rotate the control signal by the 2D rotation matrix to convert to the right frame.
    yaw = state_trimmed[3]
    control[0:2] = get_rot_matrix(yaw) @ control[0:2]

    # Save data to the output file. This appends so we don't need to change the file for each run.
    save_data(dt, state, target_pos, control, wind_enabled, OUTPUT_FILE)

    return tuple(control.tolist())
