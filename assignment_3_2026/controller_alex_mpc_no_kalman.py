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
# This variant runs the same MPC logic directly on the latest observed state,
# without applying the Kalman-filter smoothing used by controller_alex_mpc.py.

import csv
import time
from pathlib import Path

import numpy as np
from qpsolvers import solve_qp

# These are the limits placed on the sim - but note that
# the real-hardware limits are clipped to [-0.3, 0.3]
POS_CONTROL_LIMIT = 1.0
YAW_CONTROL_LIMIT = 1.0
HORIZON = 10  # How many steps forwards to consider in the MPC optimisation
DELTA_REGULARISATION_STRENGTH = 0.1
CONTROL_REGULARISATION_STRENGTH = 3.0
OUTPUT_FILE = "data.csv"


def write_data(
    dt: float,
    state: np.ndarray,
    target: np.ndarray,
    control_output: np.ndarray,
    wind_enabled: bool,
    output_file: str,
):
    """
    Write data to a persistent CSV file. This function appends so
    we do not need a new file for each run
    """
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


def wrap_angle(angle: float) -> float:
    """Converts an angle to the range [-pi, +pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MPCController:
    def __init__(
        self,
        horizon: int = HORIZON,
        delta_regularisation_strength: float = DELTA_REGULARISATION_STRENGTH,
        control_regularisation_strength: float = CONTROL_REGULARISATION_STRENGTH,
    ):
        """Sets up the MPC controller and precomputes all possible matrices. Note that it is typical
        to precompute the Hessian, but as the hardware system sees variable-length timesteps we instead
        compute it per-call.

        This variant does not filter observations before passing them into the MPC solve.
        """
        self.horizon = horizon
        self.state_dim = 4
        self.control_dim = 4

        # Set up the base matrices for the system. We assume a simplified state representation without
        # velocity or inertia considerations. That limits
        self.A = np.eye(self.state_dim)
        self.C = np.eye(self.state_dim)
        self.M = horizon

        # We pass regularisation strengths as arguments to enable tuning in a loop
        self.delta_regularisation_strength = delta_regularisation_strength
        self.control_regularisation_strength = control_regularisation_strength

        # These weights define the constrained optimisation MPC problem
        #  - The state weights penalise states which are far from the target position
        #  - The delta weights penalise rapid changes in the control signal, encouraging
        #     smooth changes in the control signal
        #  - The control regularisation penalises large control signals. This is set low
        #     so that, all other things equal, the controller prefers
        #     a trajectory with lower rotor speeds
        self.state_weights = np.array([5.0, 5.0, 5.0, 1.5])
        self.delta_weights = (
            np.array([16.0, 16.0, 16.0, 3.0]) * delta_regularisation_strength
        )
        self.control_regularization_weights = (
            np.array([0.40, 0.40, 0.40, 0.07]) * control_regularisation_strength
        )

        # This is linear MPC, which uses a quadratic solver capable of constrained
        # optimisation - these variables set the bounds for the control signal
        self.control_ub = np.array(
            [POS_CONTROL_LIMIT, POS_CONTROL_LIMIT, POS_CONTROL_LIMIT, YAW_CONTROL_LIMIT]
        )
        self.control_lb = -self.control_ub

        # Build the matrices used for MPC. We use np.kron to construct block matrices
        # Because MPC unrolls prediction over a horizon, matrices are expanded to
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

        # Initialise the
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
        # wrap the target yaw to the angle closest to the current position
        # so it won't over-rotate to achieve the target yaw
        reference[3] = x_hat[3] + wrap_angle(reference[3] - x_hat[3])
        # Repeat the reference over the length of the horizon
        return np.tile(reference, self.M)

    def _delta_offset(self) -> np.ndarray:
        # Compute the delta_u values across the horizon
        offset = np.zeros(self.control_dim * self.M)
        offset[: self.control_dim] = self.control
        return offset

    def _build_qp(
        self, x_hat: np.ndarray, target: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        # Get B and Phi. Because we want to accept varying timesteps
        # we need to recompute these at each time step
        B = self._get_B(dt)
        Phi = self._get_Phi(B)

        # We assume a constant reference over the horizon. This makes sense,
        # we're regulating to fixed points, not tracking a changing trajectory
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

        # Conditioning tricks for numerical stability:
        # - Add an epsilon along the diagonal so eigenvalues are guaranteed nonzero
        # - Symmetrise so P is guaranteed positive semidefinite
        P_qp = 2.0 * hessian + 1e-8 * np.eye(self.control_dim * self.M)
        P_qp = 0.5 * (P_qp + P_qp.T)
        q_qp = 2.0 * gradient
        return P_qp, q_qp

    def __call__(
        self,
        measurement: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        P_qp, q_qp = self._build_qp(measurement, target, dt)
        # Solve the constrained optimisation using qpsolvers, giving the optimal trajectory u_opt.
        # u_opt should never be None given the conditioning in _build_qp, but
        # we add a guard to satisfy the type checker.
        u_opt = solve_qp(P_qp, q_qp, lb=self.lb, ub=self.ub, solver="cvxopt")
        if u_opt is None:
            return self.control.copy()

        # Return only the first step of the optimal trajectory - this is classic MPC.
        return np.asarray(u_opt[: self.control_dim], dtype=float)


def get_rot_matrix(theta):
    """Returns a 2x2 matrix representing the 2D rotation
    matrix for a rotation around the z-axis."""
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


# Instantiate the controller outside the function so it persists state across calls.
mpc_controller = MPCController()


def configure_controller(
    horizon: int = HORIZON,
    delta_regularisation_strength: float = DELTA_REGULARISATION_STRENGTH,
    control_regularisation_strength: float = CONTROL_REGULARISATION_STRENGTH,
) -> MPCController:
    """Modifies the global controller. Should only be invoked for tuning"""
    global mpc_controller
    mpc_controller = MPCController(
        horizon=horizon,
        delta_regularisation_strength=delta_regularisation_strength,
        control_regularisation_strength=control_regularisation_strength,
    )
    return mpc_controller


def controller(state, target_pos, dt, wind_enabled=False, save_data=True):
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

    # We get state and target_pos in the world frame, but the control is in
    # the drone's local frame. We can ignore pitch and yaw, so rotate the xy
    # velocities by a 2D rotation matrix to convert to the right frame.
    yaw = state_trimmed[3]
    control[0:2] = get_rot_matrix(yaw) @ control[0:2]

    # Save data to the output file. This appends so we don't need to change
    # the file for each run.
    if save_data:
        write_data(dt, state, target_pos, control, wind_enabled, OUTPUT_FILE)

    return tuple(control.tolist())
