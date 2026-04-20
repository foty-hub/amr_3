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
