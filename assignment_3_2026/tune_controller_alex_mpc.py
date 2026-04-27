from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import controller_alex_mpc

HORIZON_VALUES = (10, 15, 20)
DELTA_REGULARISATION_VALUES = (0.5, 1.0, 2.0)
CONTROL_REGULARISATION_VALUES = (1.5, 3.0, 6.0)
VARIANCE_WEIGHT = 0.1
DEFAULT_WIND_ENABLED = False


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
    variance_weight: float = VARIANCE_WEIGHT,
) -> float:
    mean_error = (
        aggregate["measurement_sample_mean_abs_x_error_m"]
        + aggregate["measurement_sample_mean_abs_y_error_m"]
        + aggregate["measurement_sample_mean_abs_z_error_m"]
        + aggregate["measurement_sample_mean_abs_yaw_error_rad"]
    )
    variance_error = (
        aggregate["measurement_sample_x_error_variance_m2"]
        + aggregate["measurement_sample_y_error_variance_m2"]
        + aggregate["measurement_sample_z_error_variance_m2"]
        + aggregate["measurement_sample_yaw_error_variance_rad2"]
    )
    return float(mean_error + variance_weight * variance_error)


def load_smoke_test_module():
    import smoke_test_controller_random_targets as smoke_test

    return smoke_test


def run_candidate(
    candidate: TuningCandidate,
    wind_enabled: bool = DEFAULT_WIND_ENABLED,
    variance_weight: float = VARIANCE_WEIGHT,
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
    score = score_aggregate(smoke_result["aggregate"], variance_weight)
    return {
        "parameters": candidate.to_dict(),
        "score": score,
        "aggregate": smoke_result["aggregate"],
        "targets": smoke_result["targets"],
    }


def run_tuning(
    wind_enabled: bool = DEFAULT_WIND_ENABLED,
    variance_weight: float = VARIANCE_WEIGHT,
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
                variance_weight=variance_weight,
            )
            candidate_results.append(result)
            if verbose:
                print(f"    score={result['score']:.6f}")
    finally:
        controller_alex_mpc.configure_controller()

    ranking = sorted(candidate_results, key=lambda item: item["score"])
    return {
        "settings": {
            "horizons": list(HORIZON_VALUES),
            "delta_regularisation_strengths": list(DELTA_REGULARISATION_VALUES),
            "control_regularisation_strengths": list(CONTROL_REGULARISATION_VALUES),
            "variance_weight": variance_weight,
            "candidate_count": len(grid),
            "wind_enabled": wind_enabled,
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
        f"variance_weight={settings['variance_weight']}"
    )
    print(
        "Smoke test: "
        f"targets={settings['smoke_test']['target_count']} "
        f"settle={settings['smoke_test']['settling_window_s']:.1f}s "
        f"measure={settings['smoke_test']['measurement_window_s']:.1f}s "
        f"wind={settings['wind_enabled']}"
    )
    print()
    print(f"Top {min(top_n, len(ranking))} candidates")
    for row in ranking[:top_n]:
        params = row["parameters"]
        aggregate = row["aggregate"]
        print(
            f"{row['rank']:>2}. score={row['score']:.6f} "
            f"horizon={params['horizon']} "
            f"delta_reg={params['delta_regularisation_strength']:.3g} "
            f"control_reg={params['control_regularisation_strength']:.3g} "
            f"mean_xyz_yaw=("
            f"{aggregate['measurement_sample_mean_abs_x_error_m']:.3f}, "
            f"{aggregate['measurement_sample_mean_abs_y_error_m']:.3f}, "
            f"{aggregate['measurement_sample_mean_abs_z_error_m']:.3f}, "
            f"{aggregate['measurement_sample_mean_abs_yaw_error_rad']:.3f})"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune controller_alex_mpc.py with the random-target smoke test."
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
        "--variance-weight",
        type=float,
        default=VARIANCE_WEIGHT,
        help="Weight applied to the variance part of the score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tuning_result = run_tuning(
        wind_enabled=args.wind,
        variance_weight=args.variance_weight,
        verbose=True,
    )
    print()
    print_summary(tuning_result, top_n=args.top)


if __name__ == "__main__":
    main()
