from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import controller_alex_mpc_no_kalman as controller_alex_mpc

HORIZON_VALUES = (1, 5, 10, 15, 20)
DELTA_REGULARISATION_VALUES = (0.1, 0.5, 1.0, 2.0, 3.0)
CONTROL_REGULARISATION_VALUES = (3.0,)
STD_WEIGHT = 0.1
DEFAULT_WIND_ENABLED = False
DEFAULT_CSV_OUTPUT = THIS_DIR / "tune_controller_alex_mpc_no_kalman_results.csv"
POSITIONAL_MEAN_ERROR_GOAL_M = 0.01
POSITIONAL_ERROR_STD_GOAL_M = 0.01
YAW_MEAN_ERROR_GOAL_RAD = 0.01
YAW_ERROR_STD_GOAL_RAD = 0.001
ERROR_DIAGNOSTIC_FIELDS = (
    (
        "pos",
        "position",
        "measurement_sample_mean_position_error_m",
        "measurement_sample_position_error_variance_m2",
        "m",
    ),
    (
        "yaw",
        "yaw",
        "measurement_sample_mean_yaw_error_rad",
        "measurement_sample_yaw_error_variance_rad2",
        "rad",
    ),
)
CSV_AGGREGATE_KEYS = (
    "avg_final_position_error_m",
    "avg_final_yaw_error_rad",
    "avg_mean_position_error_m",
    "avg_mean_yaw_error_rad",
    "avg_position_error_variance_m2",
    "avg_yaw_error_variance_rad2",
    "avg_peak_position_overshoot_m",
    "avg_peak_yaw_overshoot_rad",
    "measurement_sample_mean_position_error_m",
    "measurement_sample_mean_yaw_error_rad",
    "measurement_sample_position_error_variance_m2",
    "measurement_sample_yaw_error_variance_rad2",
)


@dataclass(frozen=True)
class TuningCandidate:
    horizon: int
    delta_regularisation_strength: float
    control_regularisation_strength: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "horizon": self.horizon,
            "delta_regularisation_strength": self.delta_regularisation_strength,
            "control_regularisation_strength": self.control_regularisation_strength,
        }


def build_parameter_grid() -> list[TuningCandidate]:
    return [
        TuningCandidate(
            horizon=horizon,
            delta_regularisation_strength=delta_regularisation_strength,
            control_regularisation_strength=control_regularisation_strength,
        )
        for horizon, delta_regularisation_strength, control_regularisation_strength in itertools.product(
            HORIZON_VALUES,
            DELTA_REGULARISATION_VALUES,
            CONTROL_REGULARISATION_VALUES,
        )
    ]


def score_aggregate(
    aggregate: dict[str, float],
    std_weight: float = STD_WEIGHT,
) -> float:
    mean_error = (
        aggregate["measurement_sample_mean_position_error_m"]
        + aggregate["measurement_sample_mean_yaw_error_rad"]
    )
    std_error = error_std(
        aggregate, "measurement_sample_position_error_variance_m2"
    ) + error_std(aggregate, "measurement_sample_yaw_error_variance_rad2")
    return float(mean_error + std_weight * std_error)


def error_std(aggregate: dict[str, float], variance_key: str) -> float:
    return math.sqrt(max(aggregate[variance_key], 0.0))


def format_error_diagnostics(aggregate: dict[str, float]) -> str:
    return " ".join(
        f"{display_label}={aggregate[mean_key]:.4f}+/-"
        f"{error_std(aggregate, variance_key):.4f}{unit}"
        for display_label, _, mean_key, variance_key, unit in ERROR_DIAGNOSTIC_FIELDS
    )


def error_standard_deviations(aggregate: dict[str, float]) -> dict[str, float]:
    return {
        f"measurement_sample_{csv_label}_error_std_{unit}": error_std(
            aggregate, variance_key
        )
        for _, csv_label, _, variance_key, unit in ERROR_DIAGNOSTIC_FIELDS
    }


def meets_error_criteria(aggregate: dict[str, float]) -> bool:
    return (
        aggregate["measurement_sample_mean_position_error_m"]
        < POSITIONAL_MEAN_ERROR_GOAL_M
        and error_std(aggregate, "measurement_sample_position_error_variance_m2")
        < POSITIONAL_ERROR_STD_GOAL_M
        and aggregate["measurement_sample_mean_yaw_error_rad"] < YAW_MEAN_ERROR_GOAL_RAD
        and error_std(aggregate, "measurement_sample_yaw_error_variance_rad2")
        < YAW_ERROR_STD_GOAL_RAD
    )


def criteria_marker(aggregate: dict[str, float]) -> str:
    return "*" if meets_error_criteria(aggregate) else " "


def load_smoke_test_module():
    import smoke_test_controller_random_targets as smoke_test

    return smoke_test


def run_candidate(
    candidate: TuningCandidate,
    wind_enabled: bool = DEFAULT_WIND_ENABLED,
    std_weight: float = STD_WEIGHT,
) -> dict[str, object]:
    smoke_test = load_smoke_test_module()
    controller_alex_mpc.configure_controller(
        horizon=candidate.horizon,
        delta_regularisation_strength=candidate.delta_regularisation_strength,
        control_regularisation_strength=candidate.control_regularisation_strength,
    )
    smoke_result = smoke_test.run_smoke_test(
        controller_name="mpc",
        wind_enabled=wind_enabled,
        controller_module=controller_alex_mpc,
        controller_kwargs={"save_data": False},
    )
    score = score_aggregate(smoke_result["aggregate"], std_weight)
    return {
        "parameters": candidate.to_dict(),
        "score": score,
        "aggregate": smoke_result["aggregate"],
        "targets": smoke_result["targets"],
    }


def run_tuning(
    wind_enabled: bool = DEFAULT_WIND_ENABLED,
    std_weight: float = STD_WEIGHT,
    verbose: bool = False,
) -> dict[str, object]:
    smoke_test = load_smoke_test_module()
    grid = build_parameter_grid()
    candidate_results = []

    try:
        for index, candidate in enumerate(grid, start=1):
            if verbose:
                print(
                    f"[{index:02d}/{len(grid)}] "
                    f"horizon={candidate.horizon} "
                    f"delta_reg={candidate.delta_regularisation_strength:.3g} "
                    f"control_reg={candidate.control_regularisation_strength:.3g}"
                )
            result = run_candidate(
                candidate,
                wind_enabled=wind_enabled,
                std_weight=std_weight,
            )
            candidate_results.append(result)
            if verbose:
                aggregate = result["aggregate"]
                print(
                    f"    [{criteria_marker(aggregate)}] score={result['score']:.6f} "
                    f"mean_error+/-std: {format_error_diagnostics(aggregate)}"
                )
    finally:
        controller_alex_mpc.configure_controller()

    ranking = sorted(candidate_results, key=lambda item: item["score"])
    return {
        "settings": {
            "horizons": list(HORIZON_VALUES),
            "delta_regularisation_strengths": list(DELTA_REGULARISATION_VALUES),
            "control_regularisation_strengths": list(CONTROL_REGULARISATION_VALUES),
            "std_weight": std_weight,
            "candidate_count": len(grid),
            "wind_enabled": wind_enabled,
            "criteria": {
                "position_mean_error_goal_m": POSITIONAL_MEAN_ERROR_GOAL_M,
                "position_error_std_goal_m": POSITIONAL_ERROR_STD_GOAL_M,
                "yaw_mean_error_goal_rad": YAW_MEAN_ERROR_GOAL_RAD,
                "yaw_error_std_goal_rad": YAW_ERROR_STD_GOAL_RAD,
            },
            "smoke_test": smoke_test.settings_dict(),
        },
        "best_candidate": ranking[0],
        "ranking": [
            {
                "rank": rank,
                "score": candidate_result["score"],
                "parameters": candidate_result["parameters"],
                "aggregate": candidate_result["aggregate"],
            }
            for rank, candidate_result in enumerate(ranking, start=1)
        ],
        "candidates": candidate_results,
    }


def print_summary(tuning_result: dict[str, object], top_n: int):
    settings = tuning_result["settings"]
    ranking = tuning_result["ranking"]

    print(
        "MPC tuning grid: "
        f"{settings['candidate_count']} candidates, "
        f"std_weight={settings['std_weight']}"
    )
    print(
        "Smoke test: "
        f"targets={settings['smoke_test']['target_count']} "
        f"settle={settings['smoke_test']['settling_window_s']:.1f}s "
        f"measure={settings['smoke_test']['measurement_window_s']:.1f}s "
        f"wind={settings['wind_enabled']}"
    )
    print()
    print(
        "* = meets criteria: "
        f"pos_mean<{POSITIONAL_MEAN_ERROR_GOAL_M:g}m, "
        f"pos_std<{POSITIONAL_ERROR_STD_GOAL_M:g}m, "
        f"yaw_mean<{YAW_MEAN_ERROR_GOAL_RAD:g}rad, "
        f"yaw_std<{YAW_ERROR_STD_GOAL_RAD:g}rad"
    )
    print(f"Top {min(top_n, len(ranking))} candidates")
    for row in ranking[:top_n]:
        params = row["parameters"]
        aggregate = row["aggregate"]
        print(
            f"[{criteria_marker(aggregate)}] {row['rank']:>2}. "
            f"score={row['score']:.6f} "
            f"horizon={params['horizon']} "
            f"delta_reg={params['delta_regularisation_strength']:.3g} "
            f"control_reg={params['control_regularisation_strength']:.3g} "
            f"mean_error+/-std: {format_error_diagnostics(aggregate)}"
        )


def build_csv_rows(tuning_result: dict[str, object]) -> list[dict[str, object]]:
    settings = tuning_result["settings"]
    rows = []

    for row in tuning_result["ranking"]:
        params = row["parameters"]
        aggregate = row["aggregate"]
        rows.append(
            {
                "rank": row["rank"],
                "meets_criteria": meets_error_criteria(aggregate),
                "criteria_marker": criteria_marker(aggregate).strip(),
                "score": row["score"],
                "horizon": params["horizon"],
                "delta_regularisation_strength": params[
                    "delta_regularisation_strength"
                ],
                "control_regularisation_strength": params[
                    "control_regularisation_strength"
                ],
                "std_weight": settings["std_weight"],
                "wind_enabled": settings["wind_enabled"],
                **error_standard_deviations(aggregate),
                **{key: aggregate[key] for key in CSV_AGGREGATE_KEYS},
            }
        )

    return rows


def write_csv(tuning_result: dict[str, object], output_path: Path) -> Path:
    rows = build_csv_rows(tuning_result)
    if not rows:
        raise ValueError("No tuning results available to write")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune controller_alex_mpc_no_kalman.py with the random-target smoke test."
    )
    parser.add_argument(
        "--wind",
        action="store_true",
        help="Enable wind during tuning runs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of ranked candidates to print.",
    )
    parser.add_argument(
        "--std-weight",
        "--variance-weight",
        dest="std_weight",
        type=float,
        default=STD_WEIGHT,
        help=(
            "Weight applied to the standard deviation part of the score. "
            "--variance-weight is kept as a deprecated alias."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Path to write ranked tuning results as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tuning_result = run_tuning(
        wind_enabled=args.wind,
        std_weight=args.std_weight,
        verbose=True,
    )
    print()
    print_summary(tuning_result, top_n=args.top)
    csv_path = write_csv(tuning_result, args.csv)
    print()
    print(f"Wrote CSV results to {csv_path}")


if __name__ == "__main__":
    main()
