from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.tello_controller import TelloController
from src.wind import Wind


CONTROLLER_PATH = THIS_DIR / "controller_alex.py"

SIM_TIMESTEP = 1.0 / 1000.0
POS_CONTROL_TIMESTEP = 1.0 / 50.0
SCENARIO_TIMEOUT_SECONDS = 4.0

SETTLE_POSITION_THRESHOLD_M = 0.08
SETTLE_YAW_THRESHOLD_RAD = 0.12
SETTLE_HOLD_SECONDS = 0.50
METRIC_WINDOW_SECONDS = 0.50

ENABLE_WIND = False
WIND_CONFIG = {
    "max_steady_state": 0.02,
    "max_gust": 0.02,
    "k_gusts": 0.1,
}

SCENARIOS = (
    ("x_step", (0.30, 0.00, 1.00, 0.00)),
    ("xy_climb", (0.30, 0.30, 1.20, math.pi / 6.0)),
    ("y_translate", (0.00, 0.30, 1.20, -math.pi / 6.0)),
    ("diag_drop", (-0.20, -0.20, 0.90, math.pi / 2.0)),
    ("return_home", (0.00, 0.00, 1.00, 0.00)),
)


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
    duration_s: float
    settled: bool
    settling_time_s: float | None
    final_position_error_m: float
    final_yaw_error_rad: float
    mean_position_error_m: float
    mean_yaw_error_rad: float
    position_error_variance_m2: float
    yaw_error_variance_rad2: float
    peak_position_overshoot_m: float
    peak_yaw_overshoot_rad: float
    sample_count: int
    position_error_trace: np.ndarray
    yaw_error_trace: np.ndarray


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_error(target_yaw: float, current_yaw: float) -> float:
    return wrap_angle(target_yaw - current_yaw)


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
    def __init__(self, controller_module, wind_enabled: bool):
        self.controller_module = controller_module
        self.wind_enabled = wind_enabled
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
            raise TypeError("Controller output must be an iterable of length 4 or 5") from exc

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
        (current_pos[active_axes] - target_pos[active_axes]) * np.sign(delta[active_axes]),
        0.0,
    )
    return float(np.linalg.norm(overshoot))


def compute_yaw_overshoot(start_yaw: float, target_yaw: float, current_yaw: float) -> float:
    delta = yaw_error(target_yaw, start_yaw)
    if abs(delta) <= 1e-6:
        return 0.0

    overshoot = wrap_angle(current_yaw - target_yaw) * math.copysign(1.0, delta)
    return float(max(overshoot, 0.0))


def build_scenarios() -> list[Scenario]:
    return [Scenario(name=name, target=target) for name, target in SCENARIOS]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a headless smoke test against controller_alex.py."
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
    return parser.parse_args()


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
    position_errors = []
    yaw_errors = []

    peak_position_overshoot = 0.0
    peak_yaw_overshoot = 0.0
    settling_window_start = None
    settling_time_s = None
    scenario_start_s = sim.sim_time

    while sim.sim_time - scenario_start_s < scenario.timeout_s:
        obs = sim.step(scenario.target)
        elapsed_s = sim.sim_time - scenario_start_s

        position_error = float(np.linalg.norm(target_pos - obs.pos))
        current_yaw_error = abs(yaw_error(target_yaw, obs.euler[2]))

        timestamps.append(elapsed_s)
        position_errors.append(position_error)
        yaw_errors.append(current_yaw_error)

        peak_position_overshoot = max(
            peak_position_overshoot,
            compute_position_overshoot(start_pos, target_pos, obs.pos),
        )
        peak_yaw_overshoot = max(
            peak_yaw_overshoot,
            compute_yaw_overshoot(start_yaw, target_yaw, obs.euler[2]),
        )

        within_position = position_error <= SETTLE_POSITION_THRESHOLD_M
        within_yaw = current_yaw_error <= SETTLE_YAW_THRESHOLD_RAD

        if within_position and within_yaw:
            if settling_window_start is None:
                settling_window_start = elapsed_s
            elif elapsed_s - settling_window_start >= SETTLE_HOLD_SECONDS:
                settling_time_s = settling_window_start
                break
        else:
            settling_window_start = None

    position_error_trace = np.array(position_errors)
    yaw_error_trace = np.array(yaw_errors)

    if position_error_trace.size == 0:
        raise RuntimeError(f"Scenario {scenario.name} produced no simulation samples")

    duration_s = float(timestamps[-1])
    window_start_s = max(0.0, duration_s - METRIC_WINDOW_SECONDS)
    window_mask = np.array(timestamps) >= window_start_s

    window_position_errors = position_error_trace[window_mask]
    window_yaw_errors = yaw_error_trace[window_mask]

    return ScenarioResult(
        name=scenario.name,
        target=scenario.target,
        duration_s=duration_s,
        settled=settling_time_s is not None,
        settling_time_s=settling_time_s,
        final_position_error_m=float(position_error_trace[-1]),
        final_yaw_error_rad=float(yaw_error_trace[-1]),
        mean_position_error_m=float(np.mean(window_position_errors)),
        mean_yaw_error_rad=float(np.mean(window_yaw_errors)),
        position_error_variance_m2=float(np.var(window_position_errors)),
        yaw_error_variance_rad2=float(np.var(window_yaw_errors)),
        peak_position_overshoot_m=peak_position_overshoot,
        peak_yaw_overshoot_rad=peak_yaw_overshoot,
        sample_count=int(position_error_trace.size),
        position_error_trace=position_error_trace,
        yaw_error_trace=yaw_error_trace,
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
        "settled_fraction": float(
            np.mean([1.0 if result.settled else 0.0 for result in results])
        ),
    }

    settled_times = [result.settling_time_s for result in results if result.settled]
    per_pose_metrics["avg_settling_time_s"] = (
        float(np.mean(settled_times)) if settled_times else float("nan")
    )

    all_position_errors = np.concatenate([result.position_error_trace for result in results])
    all_yaw_errors = np.concatenate([result.yaw_error_trace for result in results])

    per_pose_metrics["all_sample_mean_position_error_m"] = float(
        np.mean(all_position_errors)
    )
    per_pose_metrics["all_sample_mean_yaw_error_rad"] = float(np.mean(all_yaw_errors))
    per_pose_metrics["all_sample_position_error_variance_m2"] = float(
        np.var(all_position_errors)
    )
    per_pose_metrics["all_sample_yaw_error_variance_rad2"] = float(np.var(all_yaw_errors))
    return per_pose_metrics


def format_optional(value: float | None, unit: str = "") -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}{unit}"


def print_report(results: list[ScenarioResult], wind_enabled: bool):
    aggregate = aggregate_results(results)

    print(f"Controller: {CONTROLLER_PATH.name}")
    print(f"Wind enabled: {wind_enabled}")
    print(
        "Settings: "
        f"sim_dt={SIM_TIMESTEP:.4f}s "
        f"control_dt={POS_CONTROL_TIMESTEP:.3f}s "
        f"timeout={SCENARIO_TIMEOUT_SECONDS:.1f}s "
        f"settle_pos<={SETTLE_POSITION_THRESHOLD_M:.3f}m "
        f"settle_yaw<={SETTLE_YAW_THRESHOLD_RAD:.3f}rad "
        f"hold={SETTLE_HOLD_SECONDS:.2f}s"
    )
    print()
    print("Per-pose results")
    for index, result in enumerate(results, start=1):
        target_x, target_y, target_z, target_yaw = result.target
        settled_text = "yes" if result.settled else "no"
        print(
            f"{index}. {result.name:<12} "
            f"target=({target_x:+.2f}, {target_y:+.2f}, {target_z:+.2f}, {target_yaw:+.2f}) "
            f"duration={result.duration_s:.2f}s settled={settled_text} "
            f"settling_time={format_optional(result.settling_time_s, 's')}"
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
            f"samples={result.sample_count}"
        )
    print()
    print("Average per pose")
    print(
        f"final_pos={aggregate['avg_final_position_error_m']:.3f}m "
        f"final_yaw={aggregate['avg_final_yaw_error_rad']:.3f}rad "
        f"mean_pos={aggregate['avg_mean_position_error_m']:.3f}m "
        f"mean_yaw={aggregate['avg_mean_yaw_error_rad']:.3f}rad"
    )
    print(
        f"var_pos={aggregate['avg_position_error_variance_m2']:.5f} "
        f"var_yaw={aggregate['avg_yaw_error_variance_rad2']:.5f} "
        f"peak_pos_overshoot={aggregate['avg_peak_position_overshoot_m']:.3f}m "
        f"peak_yaw_overshoot={aggregate['avg_peak_yaw_overshoot_rad']:.3f}rad"
    )
    print(
        f"settled_fraction={aggregate['settled_fraction']:.2f} "
        f"avg_settling_time={format_optional(aggregate['avg_settling_time_s'], 's')}"
    )
    print()
    print("All samples across all poses")
    print(
        f"mean_pos={aggregate['all_sample_mean_position_error_m']:.3f}m "
        f"mean_yaw={aggregate['all_sample_mean_yaw_error_rad']:.3f}rad "
        f"var_pos={aggregate['all_sample_position_error_variance_m2']:.5f} "
        f"var_yaw={aggregate['all_sample_yaw_error_variance_rad2']:.5f}"
    )


def main():
    if not CONTROLLER_PATH.exists():
        raise FileNotFoundError(f"Controller file not found: {CONTROLLER_PATH}")

    args = parse_args()
    controller_module = load_controller_module(CONTROLLER_PATH)
    scenarios = build_scenarios()
    simulator = HeadlessSmokeSimulator(
        controller_module,
        wind_enabled=args.wind_enabled,
    )

    try:
        simulator.reset_vehicle()
        results = [run_scenario(simulator, scenario) for scenario in scenarios]
    finally:
        simulator.disconnect()

    print_report(results, wind_enabled=args.wind_enabled)


if __name__ == "__main__":
    main()
