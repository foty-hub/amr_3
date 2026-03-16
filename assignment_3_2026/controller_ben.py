# wind_flag = False
# Implement a controller


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    pos_x, pos_y, pos_z, roll, pitch, yaw = state
    tar_x, tar_y, tar_z, tar_yaw = target_pos

    dx = pos_x - tar_x
    dy = pos_y - tar_y
    dz = pos_z - tar_z

    output = (dx, dy, dz, 0)
    return output
