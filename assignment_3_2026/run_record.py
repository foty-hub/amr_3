import argparse
import csv
import importlib
import os
import random
import time
from pathlib import Path

import controller_alex_mpc_no_kalman as controller
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from src.tello_controller import TelloController
from src.wind import Wind


THIS_DIR = Path(__file__).resolve().parent


class Simulator:
    def __init__(
        self, connection_mode=p.DIRECT, show_wind_plot=False, wind_enabled=False
    ):
        self.connection_mode = connection_mode
        self.show_wind_plot = show_wind_plot

        self.client_id = p.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet")

        if connection_mode == p.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.start_pos = [0, 0, 1]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.loadURDF(
            str(THIS_DIR / "resources" / "tello.urdf"),
            self.start_pos,
            self.start_orientation,
        )
        self.wind_enabled = wind_enabled
        self.wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1)

        self.M = 0.088
        self.L = 0.06
        self.IR = 4.95e-5
        self.KF = 0.566e-5
        self.KM = 0.762e-7
        self.K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.K_ROT = np.array([4.609e-3, 4.609e-3, 4.609e-3])
        self.TM = 0.0163
        self.tello_controller = TelloController(
            9.81, self.M, self.L, 0.35, self.KF, self.KM
        )

        self.targets = self.load_targets()
        self.current_target = 0

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]
        )
        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.targets[self.current_target][0:3],
        )
        print(f"INFO: Target set to: {self.targets[self.current_target]}")

        self.fig = None
        self.ax = None
        self.quiver = None
        if self.show_wind_plot:
            self.init_plot()

    def init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.grid(False)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Wind Speed and Direction")

        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 0, length=0, color="b")

    def update_plot(self, wind_vector):
        if not self.show_wind_plot:
            return

        if self.quiver:
            self.quiver.remove()

        scale = 30.0
        u, v, w = wind_vector * scale
        magnitude = np.linalg.norm([u, v, w])

        self.quiver = self.ax.quiver(
            0, 0, 0, u, v, w, length=magnitude, color="c", normalize=False
        )

        limit = max(abs(u), abs(v), abs(w)) + 0.2

        limit = max(limit, 0.5)

        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])

        cam_data = p.getDebugVisualizerCamera()
        if cam_data:
            pb_yaw = cam_data[8]
            pb_pitch = cam_data[9]
            new_elev = -pb_pitch
            new_azim = pb_yaw - 90
            self.ax.view_init(elev=new_elev, azim=new_azim)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def load_targets(self):
        targets = []
        try:
            with (THIS_DIR / "targets.csv").open("r") as file:
                csvreader = csv.reader(file)
                header = next(csvreader)
                for row in csvreader:
                    if len(row) != 4:
                        continue
                    if float(row[2]) < 0:
                        continue
                    targets.append(
                        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    )
        except FileNotFoundError:
            pass
        if not targets:
            targets.append((0.0, 0.0, 0.0, 0.0))
        return targets

    def compute_dynamics(self, rpm_values, lin_vel_world, quat):
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm_values * (2 * np.pi / 60)
        omega_squared = omega**2
        motor_forces = omega_squared * self.KF
        thrust = np.array([0, 0, np.sum(motor_forces)])
        vel_body = np.dot(rotation.T, lin_vel_world)
        drag_body = -self.K_TRANS * vel_body
        force = drag_body + thrust
        z_torques = omega_squared * self.KM
        z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque = (
            -motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]
        ) * self.L
        y_torque = (
            -motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]
        ) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        return force, torques

    def display_target(self):
        p.resetBasePositionAndOrientation(
            self.marker_id,
            self.targets[self.current_target][0:3],
            self.start_orientation,
        )
        print(f"INFO: Target set to: {self.targets[self.current_target]}")

    def check_action(self, unchecked_action):
        if isinstance(unchecked_action, (tuple, list)):
            if len(unchecked_action) not in [4, 5]:
                checked_action = (0, 0, 0, 0)
                p.disconnect()
            else:
                checked_action = [
                    np.clip(unchecked_action[0], -1, 1),
                    np.clip(unchecked_action[1], -1, 1),
                    np.clip(unchecked_action[2], -1, 1),
                    np.clip(unchecked_action[3], -1.74533, 1.74533),
                ]
                if len(unchecked_action) == 5:
                    checked_action.append(unchecked_action[4])
        else:
            checked_action = (0, 0, 0, 0)
            p.disconnect()
        return tuple(checked_action)

    def spin_motors(self, rpm, timestep):
        for joint_index in range(4):
            rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
            current_angle = p.getJointState(self.drone_id, joint_index)[0]
            new_angle = current_angle + rad_s * timestep
            p.resetJointState(self.drone_id, joint_index, new_angle)

    def motor_model(self, desired_rpm, current_rpm, dt):
        rpm_derivative = (desired_rpm - current_rpm) / self.TM
        real_rpm = current_rpm + rpm_derivative * dt
        return real_rpm

    def reload_controller(self):
        try:
            importlib.reload(controller)
        except Exception:
            print("ERROR: Failed to reload controller module")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Tello PyBullet simulation and record a fixed-camera MP4."
    )
    parser.add_argument("--output", default="videos/recording.mp4")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--connection", choices=["gui", "direct"], default="gui")
    parser.add_argument("--renderer", choices=["opengl", "tiny"], default="opengl")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-plot", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-index", type=int, default=0)

    parser.add_argument("--camera-distance", type=float)
    parser.add_argument("--camera-yaw", type=float, default=45.0)
    parser.add_argument("--camera-pitch", type=float, default=-25.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument(
        "--camera-target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--camera-margin", type=float, default=1.8)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    return parser.parse_args()


def validate_args(args):
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than zero")
    if args.camera_distance is not None and args.camera_distance <= 0:
        raise ValueError("--camera-distance must be greater than zero")
    if args.camera_margin <= 0:
        raise ValueError("--camera-margin must be greater than zero")
    if len(args.codec) != 4:
        raise ValueError("--codec must be a four-character OpenCV codec, e.g. mp4v")


def make_video_writer(args):
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer for {output_path}. "
            "Try --codec avc1 or --codec XVID if your OpenCV build lacks mp4v."
        )
    return writer, output_path


def make_camera_settings(args, sim):
    start_position = np.array(sim.start_pos, dtype=float)
    target_position = np.array(sim.targets[sim.current_target][:3], dtype=float)
    span = float(np.linalg.norm(target_position - start_position))

    camera_target = args.camera_target
    if camera_target is None:
        camera_target = ((start_position + target_position) / 2.0).tolist()

    camera_distance = args.camera_distance
    if camera_distance is None:
        camera_distance = max(4.0, span * args.camera_margin)

    return camera_target, camera_distance


def make_camera_matrices(args, camera_target, camera_distance):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=args.camera_yaw,
        pitch=args.camera_pitch,
        roll=args.camera_roll,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=args.fov,
        aspect=args.width / args.height,
        nearVal=args.near,
        farVal=args.far,
    )
    return view_matrix, projection_matrix


def render_frame(writer, args, view_matrix, projection_matrix, renderer):
    _, _, rgba, _, _ = p.getCameraImage(
        args.width,
        args.height,
        view_matrix,
        projection_matrix,
        renderer=renderer,
    )
    rgb_frame = np.asarray(rgba, dtype=np.uint8).reshape(
        args.height, args.width, 4
    )[:, :, :3]
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    writer.write(np.ascontiguousarray(bgr_frame))


def main():
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.chdir(THIS_DIR)

    connection_mode = p.GUI if args.connection == "gui" else p.DIRECT
    renderer = (
        p.ER_BULLET_HARDWARE_OPENGL
        if args.renderer == "opengl"
        else p.ER_TINY_RENDERER
    )

    sim = None
    writer = None
    output_path = None

    try:
        sim = Simulator(
            connection_mode=connection_mode,
            show_wind_plot=args.wind_plot,
            wind_enabled=args.wind,
        )
        if not 0 <= args.target_index < len(sim.targets):
            raise ValueError(
                f"--target-index must be between 0 and {len(sim.targets) - 1}"
            )
        if args.target_index != sim.current_target:
            sim.current_target = args.target_index
            sim.display_target()

        camera_target, camera_distance = make_camera_settings(args, sim)

        if connection_mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=args.camera_yaw,
                cameraPitch=args.camera_pitch,
                cameraTargetPosition=camera_target,
            )

        writer, output_path = make_video_writer(args)
        view_matrix, projection_matrix = make_camera_matrices(
            args, camera_target, camera_distance
        )

        print(
            f"INFO: Recording {args.duration:.2f}s at {args.fps} fps "
            f"({args.width}x{args.height}) to {output_path}"
        )
        print(
            "INFO: Camera target "
            f"{[round(value, 3) for value in camera_target]}, "
            f"distance {camera_distance:.3f}, yaw {args.camera_yaw:.1f}, "
            f"pitch {args.camera_pitch:.1f}"
        )

        timestep = 1.0 / 1000  # 1000 Hz
        pos_control_timestep = 1.0 / 50  # 20 Hz
        steps_between_pos_control = int(round(pos_control_timestep / timestep))
        loop_counter = 0

        prev_rpm = np.array([0, 0, 0, 0])
        desired_vel = np.array([0, 0, 0])
        yaw_rate_setpoint = 0

        current_wind_display = np.array([0.0, 0.0, 0.0])

        total_steps = int(round(args.duration / timestep))
        total_frames = max(1, int(round(args.duration * args.fps)))
        capture_steps = [
            int(round(frame_index / args.fps / timestep))
            for frame_index in range(total_frames)
        ]
        frames_written = 0

        for sim_step in range(total_steps + 1):
            while (
                frames_written < total_frames
                and sim_step >= capture_steps[frames_written]
            ):
                render_frame(writer, args, view_matrix, projection_matrix, renderer)
                frames_written += 1

            if sim_step >= total_steps:
                break

            loop_start = time.time() if args.realtime else None
            loop_counter += 1

            pos, quat = p.getBasePositionAndOrientation(sim.drone_id)
            lin_vel_world, ang_vel_world = p.getBaseVelocity(sim.drone_id)

            _, _, yaw = p.getEulerFromQuaternion(quat)
            yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
            _, inverted_quat = p.invertTransform([0, 0, 0], quat)
            _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)

            lin_vel = p.rotateVector(inverted_quat_yaw, lin_vel_world)
            ang_vel = p.rotateVector(inverted_quat, ang_vel_world)
            lin_vel = np.array(lin_vel)
            ang_vel = np.array(ang_vel)

            if loop_counter >= steps_between_pos_control:
                loop_counter = 0

                state = np.concatenate((pos, p.getEulerFromQuaternion(quat)))
                controller_output = sim.check_action(
                    controller.controller(
                        state,
                        sim.targets[sim.current_target],
                        pos_control_timestep,
                        sim.wind_enabled,
                    )
                )
                desired_vel = np.array(controller_output[:3])
                yaw_rate_setpoint = controller_output[3]

                sim.update_plot(current_wind_display)

            rpm = sim.tello_controller.compute_control(
                desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, timestep
            )
            rpm = sim.motor_model(rpm, prev_rpm, timestep)
            prev_rpm = rpm
            force, torque = sim.compute_dynamics(rpm, lin_vel_world, quat)

            p.applyExternalForce(sim.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
            p.applyExternalTorque(sim.drone_id, -1, torque, p.LINK_FRAME)

            current_wind_display = np.array([0.0, 0.0, 0.0])
            if sim.wind_enabled:
                current_wind_display = sim.wind_sim.get_wind(timestep)
                p.applyExternalForce(
                    sim.drone_id, -1, current_wind_display, pos, p.WORLD_FRAME
                )

            sim.spin_motors(rpm, timestep)

            if connection_mode == p.GUI:
                keys = p.getKeyboardEvents()
                if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                    print("INFO: Recording stopped by q key.")
                    break

            p.stepSimulation()

            if args.realtime and loop_start is not None:
                loop_time = time.time() - loop_start
                if loop_time < timestep:
                    time.sleep(timestep - loop_time)

        print(f"INFO: Wrote {frames_written} frame(s) to {output_path}")
    finally:
        if writer is not None:
            writer.release()
        if p.isConnected():
            p.disconnect()
        plt.close("all")


if __name__ == "__main__":
    main()
