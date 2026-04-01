# wind_flag = False
# Implement a controller
# %%
import numpy as np
from qpsolvers import solve_qp

dt = 0.02

# Let's try a basic MPC controller using the following setup
# x = [x, y, z, yaw]
# A = I_4          // 4x4 identity matrix
# B = dt * I_4
# C = I_4
# D = 0_4          // 4x4 matrix of 0s - ie. no effect


class MPCController:
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        # Define the MPC model. Note the state and control dimension are the same
        self.state_dim = 4
        self.A = np.eye(self.state_dim)
        self.C = np.eye(self.state_dim)

        # MPC Parameters
        self.q = 1  # cost weighting
        self.s = 1  # controller weighting
        self.M = horizon  # MPC horizon
        self.lb = -1  # Upper/lower bounds for control output
        self.ub = 1
        # TODO: how do you do different bounds for the
        # positional and rotational controls?
        # MPC Matrices. Since we compute B dynamically to accommodate fluctuations in sampling
        # time, we can't precompute most of the MPC matrices as you would typically do
        self.Lambda = np.block(
            [[self.C @ np.linalg.matrix_power(self.A, k)] for k in range(self.M)]
        )
        self.Q = np.diag(np.kron(np.ones(self.M), self.q))
        # Construct S - it has structure like
        # 2S -S  0  0  0
        # -S 2S -S  0  0
        #  0 -S 2S -S  0
        #  0  0 -S 2S -S
        #  0  0  0 -S  S
        S_diag = np.ones(self.M)
        S_diag[: self.M - 1] *= 2
        S_weights = np.diag(S_diag)
        S_weights += np.diag(-1 * np.ones(self.M - 1), k=1)
        S_weights += np.diag(-1 * np.ones(self.M - 1), k=-1)
        self.S = np.kron(S_weights, self.s)

        # Kalman Filter matrices
        self.P = np.eye(self.state_dim)
        self.Q_kf = np.eye(self.state_dim) * 1e-4  # Process noise
        self.R_kf = np.eye(self.state_dim) * 1e-4  # Measurement noise

        # Set initial state estimate and prev control values to 0
        self.state_estimate = np.zeros(self.state_dim)
        self.control = np.zeros(self.state_dim)

    def _get_B(self, dt: float):
        # B is dynamically computed in case the sample rate fluctuates.
        # Less relevant in sim but may matter for real deployment
        return dt * np.eye(self.state_dim)

    def _get_Phi(self, B):
        Phi = np.zeros((self.M, self.M))
        for t, j in np.ndindex(self.M, self.M):
            if t > j:
                Phi[t, j] = (self.C @ np.linalg.matrix_power(self.A, t - j - 1) @ B)[
                    0, 0
                ]
        return Phi

    def kalman_filter(
        self,
        x_hat: np.ndarray,
        B: np.ndarray,
        u_prev: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        # Kalman filter observer
        x_prior = self.A @ x_hat + B @ u_prev
        P_prior = self.A @ self.P @ self.A.T + self.Q_kf
        y_pred = self.C @ x_prior
        residual = y - y_pred
        Skf = self.C @ P_prior @ self.C.T + self.R_kf
        K = P_prior @ self.C.T @ np.linalg.inv(Skf)

        x_hat = x_prior + K @ residual
        self.P = (np.eye(self.state_dim) - K @ self.C) @ P_prior
        return x_hat

    def compute_f(self, x: np.ndarray, R: np.ndarray, u: np.ndarray, Phi: np.ndarray):
        xR = np.block([[x.reshape(-1, 1)], [R]])
        Gamma = np.block([Phi.T @ self.Q @ self.Lambda, -Phi.T @ self.Q])
        Sut = np.block(
            [
                [self.s * u],  # note this is small s - not S the matrix
                [np.zeros((self.M - 1, 1))],
            ]
        )
        return Gamma @ xR - Sut

    def compute_H(self): ...

    def mpc(self, x: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        # MPC
        B = self._get_B(dt)
        Phi = self._get_Phi(B)
        H = Phi.T @ self.Q @ Phi + self.S
        R = (target * np.ones(self.M)).reshape(-1, 1)
        f = self.compute_f(self.state_estimate, R, self.control, Phi)
        u_opt: np.ndarray = solve_qp(H, f, lb=self.lb, ub=self.ub, solver="cvxopt")  # type: ignore[assignment]
        return u_opt[0]

    def __call__(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        B = self._get_B(dt)
        self.state_estimate = self.kalman_filter(
            self.state_estimate,
            B,
            self.control,
            state,
        )

        self.control = self.mpc(self.state_estimate, target, dt)
        return self.control


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
