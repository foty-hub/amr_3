from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import random

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from src.tello_controller import TelloController
from src.wind import Wind

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

CONTROLLER_PATHS = {
    "complex_mpc": "controller_complex_mpc",
    "mpc": "controller_alex_mpc",
    "mpc_no_kalman": "controller_alex_mpc_no_kalman",
    "pid": "controller_pid",
    "lqr": "controller_ben",
}
DEFAULT_CONTROLLER = "pid"

SIM_TIMESTEP = 1.0 / 1000.0
POS_CONTROL_TIMESTEP = 1.0 / 50.0

RANDOM_SEED = 0
TARGET_NO = 25
TARGET_MAX = 4
TARGET_MIN = -4
TARGET_Z = 1.0

SETTLING_WINDOW_SECONDS = 10.0
MEASUREMENT_WINDOW_SECONDS = 10.0
SCENARIO_TIMEOUT_SECONDS = SETTLING_WINDOW_SECONDS + MEASUREMENT_WINDOW_SECONDS

ENABLE_WIND = False
WIND_CONFIG = {
    "max_steady_state": 0.02,
    "max_gust": 0.02,
    "k_gusts": 0.1,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    target: tuple[float, float, float, float]
    timeout_s: float = SCENARIO_TIMEOUT_SECONDS


@dataclass
class Observation:
    time_s: float
    pos: np.ndarray
    euler: np.ndarray
    quat: tuple[float, float, float, float]
    lin_vel_world: np.ndarray
    lin_vel_body: np.ndarray
    ang_vel_body: np.ndarray


@dataclass
class ScenarioResult:
    name: str
    target: tuple[float, float, float, float]
    start_time_s: float
    end_time_s: float
    duration_s: float
    final_position_error_m: float
    final_yaw_error_rad: float
    mean_position_error_m: float
    mean_yaw_error_rad: float
    position_error_variance_m2: float
    yaw_error_variance_rad2: float
    peak_position_overshoot_m: float
    peak_yaw_overshoot_rad: float
    mean_abs_x_error_m: float
    mean_abs_y_error_m: float
    mean_abs_z_error_m: float
    mean_abs_yaw_error_rad: float
    x_error_variance_m2: float
    y_error_variance_m2: float
    z_error_variance_m2: float
    sample_count: int
    measurement_sample_count: int
    time_trace_s: np.ndarray
    position_trace: np.ndarray
    x_error_trace: np.ndarray
    y_error_trace: np.ndarray
    z_error_trace: np.ndarray
    position_error_trace: np.ndarray
    yaw_signed_error_trace: np.ndarray
    yaw_abs_error_trace: np.ndarray
    measurement_x_error_trace: np.ndarray
    measurement_y_error_trace: np.ndarray
    measurement_z_error_trace: np.ndarray
    measurement_yaw_signed_error_trace: np.ndarray
    measurement_position_error_trace: np.ndarray
    measurement_yaw_abs_error_trace: np.ndarray


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_error(target_yaw: float, current_yaw: float) -> float:
    return wrap_angle(target_yaw - current_yaw)


def normalize_controller_name(raw_value: str) -> str:
    controller_name = raw_value.strip()
    if controller_name.startswith("controller="):
        controller_name = controller_name.split("=", maxsplit=1)[1]

    controller_name = controller_name.strip().lower()
    if controller_name not in CONTROLLER_PATHS:
        available = ", ".join(sorted(CONTROLLER_PATHS))
        raise argparse.ArgumentTypeError(
            f"Unknown controller '{controller_name}'. Choose one of: {available}"
        )
    return controller_name


def resolve_controller_path(controller_name: str) -> Path:
    return THIS_DIR / f"{CONTROLLER_PATHS[controller_name]}.py"


def load_controller_module(controller_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"smoke_test_{controller_path.stem}", controller_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load controller module from {controller_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "controller"):
        raise AttributeError(
            f"Controller module {controller_path.name} does not define controller(...)"
        )
    return module


class HeadlessSmokeSimulator:
    def __init__(
        self,
        controller_module,
        wind_enabled: bool,
        controller_kwargs: dict[str, object] | None = None,
    ):
        self.controller_module = controller_module
        self.wind_enabled = wind_enabled
        self.controller_kwargs = controller_kwargs or {}
        self.client_id = p.connect(p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet in DIRECT mode")

        p.setGravity(0, 0, -9.81)

        self.start_pos = np.array([0.0, 0.0, 1.0], dtype=float)
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
        self.plane_id = p.loadURDF(str(plane_path))
        self.drone_id = p.loadURDF(
            str(THIS_DIR / "resources" / "tello.urdf"),
            self.start_pos.tolist(),
            self.start_orientation,
        )

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
        self.steps_between_pos_control = int(round(POS_CONTROL_TIMESTEP / SIM_TIMESTEP))

        self.prev_rpm = np.zeros(4)
        self.desired_vel = np.zeros(3)
        self.yaw_rate_setpoint = 0.0
        self.loop_counter = 0
        self.sim_time = 0.0
        self.wind_sim = Wind(**WIND_CONFIG) if self.wind_enabled else None

    def disconnect(self):
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1

    def reset_vehicle(self):
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.start_pos.tolist(),
            self.start_orientation,
        )
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        self.prev_rpm = np.zeros(4)
        self.desired_vel = np.zeros(3)
        self.yaw_rate_setpoint = 0.0
        self.loop_counter = self.steps_between_pos_control
        self.sim_time = 0.0
        self.tello_controller.reset()
        self.wind_sim = Wind(**WIND_CONFIG) if self.wind_enabled else None

    def begin_target(self):
        self.tello_controller.reset()
        self.loop_counter = self.steps_between_pos_control

    def observe(self) -> Observation:
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(self.drone_id)
        euler = np.array(p.getEulerFromQuaternion(quat))

        yaw_quat = p.getQuaternionFromEuler([0, 0, euler[2]])
        _, inverted_quat = p.invertTransform([0, 0, 0], quat)
        _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)

        lin_vel_body = np.array(p.rotateVector(inverted_quat_yaw, lin_vel_world))
        ang_vel_body = np.array(p.rotateVector(inverted_quat, ang_vel_world))

        return Observation(
            time_s=self.sim_time,
            pos=np.array(pos),
            euler=euler,
            quat=quat,
            lin_vel_world=np.array(lin_vel_world),
            lin_vel_body=lin_vel_body,
            ang_vel_body=ang_vel_body,
        )

    def step(self, target: tuple[float, float, float, float]) -> Observation:
        self.loop_counter += 1
        obs = self.observe()

        if self.loop_counter >= self.steps_between_pos_control:
            self.loop_counter = 0
            state = np.concatenate((obs.pos, obs.euler))
            controller_output = self.check_action(
                self.controller_module.controller(
                    state,
                    target,
                    POS_CONTROL_TIMESTEP,
                    self.wind_enabled,
                    **self.controller_kwargs,
                )
            )
            self.desired_vel = np.array(controller_output[:3], dtype=float)
            self.yaw_rate_setpoint = controller_output[3]

        rpm = self.tello_controller.compute_control(
            self.desired_vel,
            obs.lin_vel_body,
            obs.quat,
            obs.ang_vel_body,
            self.yaw_rate_setpoint,
            SIM_TIMESTEP,
        )
        rpm = self.motor_model(rpm, self.prev_rpm, SIM_TIMESTEP)
        self.prev_rpm = rpm

        force, torque = self.compute_dynamics(rpm, obs.lin_vel_world, obs.quat)
        p.applyExternalForce(
            self.drone_id,
            -1,
            force.tolist(),
            [0, 0, 0],
            p.LINK_FRAME,
        )
        p.applyExternalTorque(self.drone_id, -1, torque.tolist(), p.LINK_FRAME)

        if self.wind_enabled and self.wind_sim is not None:
            wind_force = np.array(self.wind_sim.get_wind(SIM_TIMESTEP))
            p.applyExternalForce(
                self.drone_id,
                -1,
                wind_force.tolist(),
                obs.pos.tolist(),
                p.WORLD_FRAME,
            )

        self.spin_motors(rpm, SIM_TIMESTEP)
        p.stepSimulation()
        self.sim_time += SIM_TIMESTEP
        return self.observe()

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

    def spin_motors(self, rpm, timestep: float):
        for joint_index in range(4):
            rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
            current_angle = p.getJointState(self.drone_id, joint_index)[0]
            new_angle = current_angle + rad_s * timestep
            p.resetJointState(self.drone_id, joint_index, new_angle)

    def motor_model(self, desired_rpm, current_rpm, dt: float):
        rpm_derivative = (desired_rpm - current_rpm) / self.TM
        return current_rpm + rpm_derivative * dt

    @staticmethod
    def check_action(unchecked_action):
        try:
            values = list(unchecked_action)
        except TypeError as exc:
            raise TypeError(
                "Controller output must be an iterable of length 4 or 5"
            ) from exc

        if len(values) not in (4, 5):
            raise ValueError("Controller output must have length 4 or 5")

        checked_action = (
            float(np.clip(values[0], -1, 1)),
            float(np.clip(values[1], -1, 1)),
            float(np.clip(values[2], -1, 1)),
            float(np.clip(values[3], -1.74533, 1.74533)),
        )
        return checked_action


def compute_position_overshoot(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    current_pos: np.ndarray,
) -> float:
    delta = target_pos - start_pos
    active_axes = np.abs(delta) > 1e-6
    if not np.any(active_axes):
        return 0.0

    overshoot = np.zeros(3)
    overshoot[active_axes] = np.maximum(
        (current_pos[active_axes] - target_pos[active_axes])
        * np.sign(delta[active_axes]),
        0.0,
    )
    return float(np.linalg.norm(overshoot))


def compute_yaw_overshoot(
    start_yaw: float, target_yaw: float, current_yaw: float
) -> float:
    delta = yaw_error(target_yaw, start_yaw)
    if abs(delta) <= 1e-6:
        return 0.0

    overshoot = wrap_angle(current_yaw - target_yaw) * math.copysign(1.0, delta)
    return float(max(overshoot, 0.0))


def build_scenarios(
    seed: int = RANDOM_SEED,
    target_count: int = TARGET_NO,
) -> list[Scenario]:
    rng = random.Random(seed)
    return [
        Scenario(
            name=f"Random Target Test: {index + 1}",
            target=(
                rng.uniform(TARGET_MIN, TARGET_MAX),
                rng.uniform(TARGET_MIN, TARGET_MAX),
                TARGET_Z,
                rng.uniform(-math.pi, math.pi),
            ),
        )
        for index in range(target_count)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a headless smoke test against one of the configured controllers."
    )
    parser.add_argument(
        "--controller",
        dest="controller_name",
        type=normalize_controller_name,
        help="Controller key from CONTROLLER_PATHS, for example '--controller pid'.",
    )
    parser.add_argument(
        "controller_selector",
        nargs="?",
        type=normalize_controller_name,
        help="Optional shorthand controller selector, for example 'controller=pid' or 'pid'.",
    )
    wind_group = parser.add_mutually_exclusive_group()
    wind_group.add_argument(
        "--wind",
        dest="wind_enabled",
        action="store_true",
        help="Enable the configured wind preset for this run.",
    )
    wind_group.add_argument(
        "--no-wind",
        dest="wind_enabled",
        action="store_false",
        help="Disable wind for this run.",
    )
    parser.set_defaults(wind_enabled=ENABLE_WIND)
    args = parser.parse_args()

    if (
        args.controller_name is not None
        and args.controller_selector is not None
        and args.controller_name != args.controller_selector
    ):
        parser.error(
            "Conflicting controller selections provided via '--controller' and the positional selector."
        )

    args.controller_name = (
        args.controller_name or args.controller_selector or DEFAULT_CONTROLLER
    )
    return args


def run_scenario(
    sim: HeadlessSmokeSimulator,
    scenario: Scenario,
) -> ScenarioResult:
    sim.begin_target()

    start_obs = sim.observe()
    start_pos = start_obs.pos.copy()
    start_yaw = float(start_obs.euler[2])
    target_pos = np.array(scenario.target[:3], dtype=float)
    target_yaw = float(scenario.target[3])

    timestamps = []
    positions = []
    x_errors = []
    y_errors = []
    z_errors = []
    position_errors = []
    yaw_signed_errors = []
    yaw_abs_errors = []

    peak_position_overshoot = 0.0
    peak_yaw_overshoot = 0.0
    scenario_start_s = sim.sim_time

    while sim.sim_time - scenario_start_s < scenario.timeout_s:
        obs = sim.step(scenario.target)

        position_delta = target_pos - obs.pos
        signed_yaw_error = yaw_error(target_yaw, obs.euler[2])
        position_error = float(np.linalg.norm(position_delta))
        current_yaw_error = abs(signed_yaw_error)

        timestamps.append(sim.sim_time)
        positions.append(obs.pos.copy())
        x_errors.append(position_delta[0])
        y_errors.append(position_delta[1])
        z_errors.append(position_delta[2])
        position_errors.append(position_error)
        yaw_signed_errors.append(signed_yaw_error)
        yaw_abs_errors.append(current_yaw_error)

        peak_position_overshoot = max(
            peak_position_overshoot,
            compute_position_overshoot(start_pos, target_pos, obs.pos),
        )
        peak_yaw_overshoot = max(
            peak_yaw_overshoot,
            compute_yaw_overshoot(start_yaw, target_yaw, obs.euler[2]),
        )

    position_error_trace = np.array(position_errors)
    yaw_abs_error_trace = np.array(yaw_abs_errors)

    if position_error_trace.size == 0:
        raise RuntimeError(f"Scenario {scenario.name} produced no simulation samples")

    duration_s = float(timestamps[-1] - scenario_start_s)
    time_trace_s = np.array(timestamps)
    elapsed_trace_s = time_trace_s - scenario_start_s
    measurement_mask = elapsed_trace_s >= SETTLING_WINDOW_SECONDS

    x_error_trace = np.array(x_errors)
    y_error_trace = np.array(y_errors)
    z_error_trace = np.array(z_errors)
    yaw_signed_error_trace = np.array(yaw_signed_errors)
    measurement_position_errors = position_error_trace[measurement_mask]
    measurement_x_errors = x_error_trace[measurement_mask]
    measurement_y_errors = y_error_trace[measurement_mask]
    measurement_z_errors = z_error_trace[measurement_mask]
    measurement_yaw_signed_errors = yaw_signed_error_trace[measurement_mask]
    measurement_yaw_errors = yaw_abs_error_trace[measurement_mask]
    if measurement_position_errors.size == 0:
        raise RuntimeError(
            f"Scenario {scenario.name} produced no measurement-window samples"
        )

    return ScenarioResult(
        name=scenario.name,
        target=scenario.target,
        start_time_s=scenario_start_s,
        end_time_s=float(timestamps[-1]),
        duration_s=duration_s,
        final_position_error_m=float(position_error_trace[-1]),
        final_yaw_error_rad=float(yaw_abs_error_trace[-1]),
        mean_position_error_m=float(np.mean(measurement_position_errors)),
        mean_yaw_error_rad=float(np.mean(measurement_yaw_errors)),
        position_error_variance_m2=float(np.var(measurement_position_errors)),
        yaw_error_variance_rad2=float(np.var(measurement_yaw_signed_errors)),
        peak_position_overshoot_m=peak_position_overshoot,
        peak_yaw_overshoot_rad=peak_yaw_overshoot,
        mean_abs_x_error_m=float(np.mean(np.abs(measurement_x_errors))),
        mean_abs_y_error_m=float(np.mean(np.abs(measurement_y_errors))),
        mean_abs_z_error_m=float(np.mean(np.abs(measurement_z_errors))),
        mean_abs_yaw_error_rad=float(np.mean(measurement_yaw_errors)),
        x_error_variance_m2=float(np.var(measurement_x_errors)),
        y_error_variance_m2=float(np.var(measurement_y_errors)),
        z_error_variance_m2=float(np.var(measurement_z_errors)),
        sample_count=int(position_error_trace.size),
        measurement_sample_count=int(measurement_position_errors.size),
        time_trace_s=time_trace_s,
        position_trace=np.vstack(positions),
        x_error_trace=x_error_trace,
        y_error_trace=y_error_trace,
        z_error_trace=z_error_trace,
        position_error_trace=position_error_trace,
        yaw_signed_error_trace=yaw_signed_error_trace,
        yaw_abs_error_trace=yaw_abs_error_trace,
        measurement_x_error_trace=measurement_x_errors,
        measurement_y_error_trace=measurement_y_errors,
        measurement_z_error_trace=measurement_z_errors,
        measurement_yaw_signed_error_trace=measurement_yaw_signed_errors,
        measurement_position_error_trace=measurement_position_errors,
        measurement_yaw_abs_error_trace=measurement_yaw_errors,
    )


def aggregate_results(results: list[ScenarioResult]) -> dict[str, float]:
    per_pose_metrics = {
        "avg_final_position_error_m": float(
            np.mean([result.final_position_error_m for result in results])
        ),
        "avg_final_yaw_error_rad": float(
            np.mean([result.final_yaw_error_rad for result in results])
        ),
        "avg_mean_position_error_m": float(
            np.mean([result.mean_position_error_m for result in results])
        ),
        "avg_mean_yaw_error_rad": float(
            np.mean([result.mean_yaw_error_rad for result in results])
        ),
        "avg_position_error_variance_m2": float(
            np.mean([result.position_error_variance_m2 for result in results])
        ),
        "avg_yaw_error_variance_rad2": float(
            np.mean([result.yaw_error_variance_rad2 for result in results])
        ),
        "avg_peak_position_overshoot_m": float(
            np.mean([result.peak_position_overshoot_m for result in results])
        ),
        "avg_peak_yaw_overshoot_rad": float(
            np.mean([result.peak_yaw_overshoot_rad for result in results])
        ),
        "avg_mean_abs_x_error_m": float(
            np.mean([result.mean_abs_x_error_m for result in results])
        ),
        "avg_mean_abs_y_error_m": float(
            np.mean([result.mean_abs_y_error_m for result in results])
        ),
        "avg_mean_abs_z_error_m": float(
            np.mean([result.mean_abs_z_error_m for result in results])
        ),
        "avg_mean_abs_yaw_error_rad": float(
            np.mean([result.mean_abs_yaw_error_rad for result in results])
        ),
        "avg_x_error_variance_m2": float(
            np.mean([result.x_error_variance_m2 for result in results])
        ),
        "avg_y_error_variance_m2": float(
            np.mean([result.y_error_variance_m2 for result in results])
        ),
        "avg_z_error_variance_m2": float(
            np.mean([result.z_error_variance_m2 for result in results])
        ),
    }

    all_position_errors = np.concatenate(
        [result.measurement_position_error_trace for result in results]
    )
    all_yaw_errors = np.concatenate(
        [result.measurement_yaw_abs_error_trace for result in results]
    )
    all_x_errors = np.concatenate([result.measurement_x_error_trace for result in results])
    all_y_errors = np.concatenate([result.measurement_y_error_trace for result in results])
    all_z_errors = np.concatenate([result.measurement_z_error_trace for result in results])
    all_yaw_signed_errors = np.concatenate(
        [result.measurement_yaw_signed_error_trace for result in results]
    )

    per_pose_metrics["measurement_sample_mean_position_error_m"] = float(
        np.mean(all_position_errors)
    )
    per_pose_metrics["measurement_sample_mean_yaw_error_rad"] = float(
        np.mean(all_yaw_errors)
    )
    per_pose_metrics["measurement_sample_position_error_variance_m2"] = float(
        np.var(all_position_errors)
    )
    per_pose_metrics["measurement_sample_yaw_error_variance_rad2"] = float(
        np.var(all_yaw_signed_errors)
    )
    per_pose_metrics["measurement_sample_mean_abs_x_error_m"] = float(
        np.mean(np.abs(all_x_errors))
    )
    per_pose_metrics["measurement_sample_mean_abs_y_error_m"] = float(
        np.mean(np.abs(all_y_errors))
    )
    per_pose_metrics["measurement_sample_mean_abs_z_error_m"] = float(
        np.mean(np.abs(all_z_errors))
    )
    per_pose_metrics["measurement_sample_mean_abs_yaw_error_rad"] = float(
        np.mean(all_yaw_errors)
    )
    per_pose_metrics["measurement_sample_x_error_variance_m2"] = float(
        np.var(all_x_errors)
    )
    per_pose_metrics["measurement_sample_y_error_variance_m2"] = float(
        np.var(all_y_errors)
    )
    per_pose_metrics["measurement_sample_z_error_variance_m2"] = float(
        np.var(all_z_errors)
    )
    return per_pose_metrics


def settings_dict() -> dict[str, float | int | bool]:
    return {
        "random_seed": RANDOM_SEED,
        "target_count": TARGET_NO,
        "target_min": TARGET_MIN,
        "target_max": TARGET_MAX,
        "target_z": TARGET_Z,
        "sim_timestep_s": SIM_TIMESTEP,
        "position_control_timestep_s": POS_CONTROL_TIMESTEP,
        "settling_window_s": SETTLING_WINDOW_SECONDS,
        "measurement_window_s": MEASUREMENT_WINDOW_SECONDS,
        "scenario_timeout_s": SCENARIO_TIMEOUT_SECONDS,
    }


def scenario_result_to_dict(result: ScenarioResult) -> dict[str, object]:
    return {
        "name": result.name,
        "target": [float(value) for value in result.target],
        "start_time_s": result.start_time_s,
        "end_time_s": result.end_time_s,
        "duration_s": result.duration_s,
        "final_position_error_m": result.final_position_error_m,
        "final_yaw_error_rad": result.final_yaw_error_rad,
        "mean_position_error_m": result.mean_position_error_m,
        "mean_yaw_error_rad": result.mean_yaw_error_rad,
        "position_error_variance_m2": result.position_error_variance_m2,
        "yaw_error_variance_rad2": result.yaw_error_variance_rad2,
        "peak_position_overshoot_m": result.peak_position_overshoot_m,
        "peak_yaw_overshoot_rad": result.peak_yaw_overshoot_rad,
        "mean_abs_x_error_m": result.mean_abs_x_error_m,
        "mean_abs_y_error_m": result.mean_abs_y_error_m,
        "mean_abs_z_error_m": result.mean_abs_z_error_m,
        "mean_abs_yaw_error_rad": result.mean_abs_yaw_error_rad,
        "x_error_variance_m2": result.x_error_variance_m2,
        "y_error_variance_m2": result.y_error_variance_m2,
        "z_error_variance_m2": result.z_error_variance_m2,
        "sample_count": result.sample_count,
        "measurement_sample_count": result.measurement_sample_count,
    }


def results_to_dict(
    controller_name: str,
    controller_path: Path,
    wind_enabled: bool,
    results: list[ScenarioResult],
) -> dict[str, object]:
    return {
        "controller": controller_name,
        "controller_file": controller_path.name,
        "wind_enabled": wind_enabled,
        "settings": settings_dict(),
        "aggregate": aggregate_results(results),
        "targets": [scenario_result_to_dict(result) for result in results],
    }


def run_smoke_test_results(
    controller_name: str = DEFAULT_CONTROLLER,
    wind_enabled: bool = ENABLE_WIND,
    controller_module=None,
    controller_kwargs: dict[str, object] | None = None,
) -> tuple[str, Path, list[ScenarioResult]]:
    controller_name = normalize_controller_name(controller_name)
    controller_path = resolve_controller_path(controller_name)
    if not controller_path.exists():
        raise FileNotFoundError(
            f"Controller '{controller_name}' resolves to missing file: {controller_path}"
        )

    if controller_module is None:
        controller_module = load_controller_module(controller_path)
    scenarios = build_scenarios()
    simulator = HeadlessSmokeSimulator(
        controller_module,
        wind_enabled=wind_enabled,
        controller_kwargs=controller_kwargs,
    )

    try:
        simulator.reset_vehicle()
        results = [run_scenario(simulator, scenario) for scenario in scenarios]
    finally:
        simulator.disconnect()

    return controller_name, controller_path, results


def run_smoke_test(
    controller_name: str = DEFAULT_CONTROLLER,
    wind_enabled: bool = ENABLE_WIND,
    controller_module=None,
    controller_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    controller_name, controller_path, results = run_smoke_test_results(
        controller_name=controller_name,
        wind_enabled=wind_enabled,
        controller_module=controller_module,
        controller_kwargs=controller_kwargs,
    )
    return results_to_dict(
        controller_name=controller_name,
        controller_path=controller_path,
        wind_enabled=wind_enabled,
        results=results,
    )


def add_time_annotations(ax, results: list[ScenarioResult], show_labels: bool = False):
    for result in results[1:]:
        ax.axvline(
            result.start_time_s,
            color="0.55",
            linewidth=1.0,
            alpha=0.7,
            zorder=0,
        )

    for result in results:
        ax.axvline(
            result.start_time_s + SETTLING_WINDOW_SECONDS,
            color="0.35",
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
            zorder=0,
        )

    if show_labels:
        for result in results:
            midpoint = (
                result.start_time_s + (result.end_time_s - result.start_time_s) / 2.0
            )
            ax.text(
                midpoint,
                1.02,
                result.name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.25",
            )


def plot_results(
    results: list[ScenarioResult],
    wind_enabled: bool,
    controller_path: Path,
):
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    ax_x = fig.add_subplot(grid[0, 0])
    ax_y = fig.add_subplot(grid[0, 1], sharex=ax_x)
    ax_z = fig.add_subplot(grid[0, 2], sharex=ax_x)
    ax_yaw = fig.add_subplot(grid[1, 0], sharex=ax_x)
    ax_pos_norm = fig.add_subplot(grid[1, 1], sharex=ax_x)
    ax_xy = fig.add_subplot(grid[1, 2])

    series_axes = (
        (ax_x, "X Error", "x_error_trace", "m"),
        (ax_y, "Y Error", "y_error_trace", "m"),
        (ax_z, "Z Error", "z_error_trace", "m"),
        (ax_yaw, "Yaw Error", "yaw_signed_error_trace", "rad"),
        (ax_pos_norm, "Position Error Norm", "position_error_trace", "m"),
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    for axis, title, _, unit in series_axes:
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)

    for axis in (ax_x, ax_y, ax_z, ax_yaw):
        axis.axhline(0.0, color="0.75", linewidth=0.9, zorder=0)

    for color, result in zip(colors, results):
        for axis, _, attribute_name, _ in series_axes:
            axis.plot(
                result.time_trace_s,
                getattr(result, attribute_name),
                color=color,
                linewidth=1.4,
            )

        ax_xy.plot(
            result.position_trace[:, 0],
            result.position_trace[:, 1],
            color=color,
            linewidth=1.6,
        )
        ax_xy.scatter(
            result.target[0],
            result.target[1],
            color=color,
            marker="x",
            s=60,
        )
        ax_xy.annotate(
            result.name,
            (result.target[0], result.target[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    initial_pos = results[0].position_trace[0]
    ax_xy.scatter(
        initial_pos[0],
        initial_pos[1],
        color="0.1",
        marker="o",
        s=35,
        label="start",
    )
    ax_xy.set_title("XY Trajectory")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.grid(alpha=0.25)
    ax_xy.axis("equal")

    for axis in (ax_x, ax_y, ax_z, ax_yaw, ax_pos_norm):
        add_time_annotations(axis, results, show_labels=axis is ax_pos_norm)
        axis.set_xlim(results[0].start_time_s, results[-1].end_time_s)

    ax_yaw.set_xlabel("time (s)")
    ax_pos_norm.set_xlabel("time (s)")

    fig.suptitle(
        f"Controller Smoke Test: {controller_path.name} | wind={'on' if wind_enabled else 'off'}",
        fontsize=14,
    )

    if "agg" in plt.get_backend().lower():
        fig.canvas.draw()
        return

    plt.show()


def print_report(
    results: list[ScenarioResult],
    wind_enabled: bool,
    controller_path: Path,
):
    aggregate = aggregate_results(results)
    settings = settings_dict()

    print(f"Controller: {controller_path.name}")
    print(f"Wind enabled: {wind_enabled}")
    print(
        "Settings: "
        f"sim_dt={SIM_TIMESTEP:.4f}s "
        f"control_dt={POS_CONTROL_TIMESTEP:.3f}s "
        f"seed={settings['random_seed']} "
        f"targets={settings['target_count']} "
        f"settle_window={SETTLING_WINDOW_SECONDS:.1f}s "
        f"measure_window={MEASUREMENT_WINDOW_SECONDS:.1f}s "
        f"timeout={SCENARIO_TIMEOUT_SECONDS:.1f}s"
    )
    print()
    print("Per-pose results")
    for index, result in enumerate(results, start=1):
        target_x, target_y, target_z, target_yaw = result.target
        print(
            f"{index}. {result.name:<22} "
            f"target=({target_x:+.2f}, {target_y:+.2f}, {target_z:+.2f}, {target_yaw:+.2f}) "
            f"duration={result.duration_s:.2f}s"
        )
        print(
            "   "
            f"final_pos={result.final_position_error_m:.3f}m "
            f"final_yaw={result.final_yaw_error_rad:.3f}rad "
            f"mean_pos={result.mean_position_error_m:.3f}m "
            f"mean_yaw={result.mean_yaw_error_rad:.3f}rad "
            f"var_pos={result.position_error_variance_m2:.5f} "
            f"var_yaw={result.yaw_error_variance_rad2:.5f}"
        )
        print(
            "   "
            f"peak_pos_overshoot={result.peak_position_overshoot_m:.3f}m "
            f"peak_yaw_overshoot={result.peak_yaw_overshoot_rad:.3f}rad "
            f"samples={result.sample_count} "
            f"measurement_samples={result.measurement_sample_count}"
        )
    print()
    print("Average per pose")
    print(
        f"final_pos={aggregate['avg_final_position_error_m']:.3f}m "
        f"final_yaw={aggregate['avg_final_yaw_error_rad']:.3f}rad "
        f"mean_pos={aggregate['avg_mean_position_error_m']:.5f}m "
        f"mean_yaw={aggregate['avg_mean_yaw_error_rad']:.3f}rad"
    )
    print(
        f"var_pos={aggregate['avg_position_error_variance_m2']:.5f} "
        f"var_yaw={aggregate['avg_yaw_error_variance_rad2']:.5f} "
        f"peak_pos_overshoot={aggregate['avg_peak_position_overshoot_m']:.3f}m "
        f"peak_yaw_overshoot={aggregate['avg_peak_yaw_overshoot_rad']:.3f}rad"
    )
    print()
    print("Measurement samples across all poses")
    print(
        f"mean_pos={aggregate['measurement_sample_mean_position_error_m']:.3f}m "
        f"mean_yaw={aggregate['measurement_sample_mean_yaw_error_rad']:.3f}rad "
        f"var_pos={aggregate['measurement_sample_position_error_variance_m2']:.5f} "
        f"var_yaw={aggregate['measurement_sample_yaw_error_variance_rad2']:.5f}"
    )


def main():
    args = parse_args()
    _, controller_path, results = run_smoke_test_results(
        controller_name=args.controller_name,
        wind_enabled=args.wind_enabled,
    )

    print_report(
        results,
        wind_enabled=args.wind_enabled,
        controller_path=controller_path,
    )
    plot_results(
        results,
        wind_enabled=args.wind_enabled,
        controller_path=controller_path,
    )


if __name__ == "__main__":
    main()
