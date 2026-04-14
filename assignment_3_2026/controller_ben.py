# wind_flag = False
# Implement a controller
import numpy as np
from collections import deque

class KalmanFilter:
    def __init__(self):
        '''
    ### Inputs:
    - F: State transition matrix
    - C: Observation matrix
    - V: Process noise covariance matrix
    - W: Observation noise covariance matrix
    - x0: Initial value of state
    - P0: Initial value of error covariance matrix
    - y: Noisy observations, a 2D array with MxN dimensions, M represents the modality of the signals, N representes the number of time steps.  
    ### Output:
    - x_hat - Esimated state variable over time
    '''
        self.history = deque(maxlen=100)
        self.x_hat = np.array[[0],[0],[0]]
        self.P = np.eye(3)

    # Allocate memory for x_hat based on the number of columns in y
        self.x_hat = np.zeros(3,1)
      
    # Initialize Kalman filter
    
    def __call__(self, y):
        
        self.history.append(y)
            
        
        # Kalman filter loop
        # Prediction
        x_hat_Prior = F @ x_hat
        P = F @ P @ F.T + V

        # Correction
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + W)
        x_hat[:, i+1] = x_hat_Prior + K @ (y[:, i] - C @ x_hat_Prior)
        P = (np.eye(F.shape[0]) - K @ C) @ P
    
        return x_hat

KalmanFilterItem = KalmanFilter()

def controller(state, target_pos, dt, wind_enabled=False):
    
    # System parameters
    m = 4       # mass of aircraft
    J = 0.0475  # inertia around pitch axis
    r = 0.25    # distance to center of force
    g = 9.8     # gravitational constant
    c = 0.05    # damping factor (estimated)

    #   xdot = Ax + B u  =>  xdot = (A-BK)x + K xd
    #      u = -K(x - xd)       y = Cx + Du

    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    # pos_x, pos_y, pos_z, roll, pitch, yaw = state
    # tar_x, tar_y, tar_z, tar_yaw = target_pos

    # pos, quat = p.getBasePositionAndOrientation(sim.drone_id)
    # lin_vel_world, ang_vel_world = p.getBaseVelocity(sim.drone_id)

    # roll, pitch, yaw = p.getEulerFromQuaternion(quat)
    # yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
    # inverted_pos, inverted_quat = p.invertTransform([0, 0, 0], quat)
    # inverted_pos_yaw, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
    
    # lin_vel = p.rotateVector(inverted_quat_yaw, lin_vel_world)
    # ang_vel = p.rotateVector(inverted_quat, ang_vel_world)
    # lin_vel = np.array(lin_vel)
    # ang_vel = np.array(ang_vel)

    # state = np.concatenate((pos, p.getEulerFromQuaternion(quat)))
    # rpm = sim.tello_controller.compute_control( desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, timestep)
    # desired_vel = np.array(controller_output[:3])
    # yaw_rate_setpoint = controller_output[3]
    # rpm = sim.motor_model(rpm, prev_rpm, timestep)
    # prev_rpm = rpm
    # force, torque = sim.compute_dynamics(rpm, lin_vel_world, quat)

    # Compute_dynamics
    # rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    # omega = rpm_values * (2 * np.pi / 60)
    # omega_squared = omega**2
    # motor_forces = omega_squared * self.KF
    # thrust = np.array([0, 0, np.sum(motor_forces)])
    # vel_body = np.dot(rotation.T, lin_vel_world)
    # drag_body = -self.K_TRANS * vel_body
    # force = drag_body + thrust
    # z_torques = omega_squared * self.KM
    # z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
    # x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * self.L
    # y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * self.L
    # torques = np.array([x_torque, y_torque, z_torque])
    # return force, torques

    # States for LQR:
    # x = [x,y,z,x_dot,theta,y_dot,z_dot,theta_dot]
    # x_dot = [x_dot,y_dot,z_dot,theta_dot,x_dot_dot,y_dot,z_dot_dot,theta_dot_dot]
    # u = [x_dot_setpoint,y_dot_setpoint,z_dot_setpoint]
    # y = [x,y,z]

    # A is x by x_dot (in our case 8 by 8)
    A = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]]
     )
    #     
    # B is x by u (8 by 4)
    B = np.array(
     [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
     )
    # 
    # C is y by x (4 by 8)
    C = np.array([
     [1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0]]
     )
    # 
    # D is y by u (4 x 4)
    D = np.zeros(4)

    # Internal States:

    dx = pos_x - tar_x
    dy = pos_y - tar_y
    dz = pos_z - tar_z

    output = (dx, dy, dz, 0)
    return output
