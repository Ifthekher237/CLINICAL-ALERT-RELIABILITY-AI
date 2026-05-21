"""Synthetic patient-monitoring data simulator.

This module creates realistic-enough time-series data for engineering
experiments in the clinical alert reliability prototype. It does not use real
patient data and must not be interpreted as clinically validated simulation.
The goal is to produce reproducible data with trends, deterioration patterns,
sensor artifacts, missingness, and simple outcome labels for later roadmap
steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VITAL_COLUMNS = [
    "heart_rate",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    *VITAL_COLUMNS,
    "patient_condition_label",
    "deterioration_event",
    "sensor_noise_flag",
    "missing_data_flag",
    "patient_outcome_after_alert",
    "outcome_timestamp",
    "outcome_severity_change",
]

DEFAULT_OUTPUT_PATH = Path("data/simulated/patient_monitoring.csv")
DEFAULT_START_TIMESTAMP = pd.Timestamp("2026-01-01 08:00:00")

BEHAVIOR_TYPES = ("normal", "gradual_deterioration", "sudden_deterioration")
CONDITION_LABELS = ("normal", "deteriorating", "critical")
OUTCOME_LABELS = ("improved", "unchanged", "worsened", "unknown")


@dataclass(frozen=True)
class PatientProfile:
    """Configuration for one synthetic patient timeline."""

    patient_id: str
    behavior_type: str
    baseline: dict[str, float]
    deterioration_start_index: int | None
    total_deterioration_delta: dict[str, float]
    sudden_course: str


class PatientDataSimulator:
    """Small object-oriented wrapper around ``simulate_patient_data``."""

    def __init__(
        self,
        num_patients: int,
        duration_hours: int,
        frequency_minutes: int,
        seed: int = 42,
    ) -> None:
        self.num_patients = num_patients
        self.duration_hours = duration_hours
        self.frequency_minutes = frequency_minutes
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        """Generate synthetic monitoring records using the configured settings."""
        return simulate_patient_data(
            num_patients=self.num_patients,
            duration_hours=self.duration_hours,
            frequency_minutes=self.frequency_minutes,
            seed=self.seed,
        )


def simulate_patient_data(
    num_patients: int,
    duration_hours: int,
    frequency_minutes: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate multi-patient monitoring data and save it to CSV.

    Args:
        num_patients: Number of independent synthetic patient timelines.
        duration_hours: Duration of each timeline in hours. A zero-hour
            duration still returns one timestamp per patient for edge-case
            testing.
        frequency_minutes: Sampling frequency in minutes.
        seed: Random seed for reproducible simulation.

    Returns:
        A pandas DataFrame containing the mandatory Step 2 schema.
    """
    _validate_simulation_inputs(num_patients, duration_hours, frequency_minutes)

    rng = np.random.default_rng(seed)
    timestamps = _build_timestamps(duration_hours, frequency_minutes)
    behavior_types = _assign_behavior_types(num_patients, rng)

    patient_frames = []
    for patient_number, behavior_type in enumerate(behavior_types, start=1):
        profile = _create_patient_profile(
            patient_id=f"P{patient_number:04d}",
            behavior_type=behavior_type,
            num_steps=len(timestamps),
            rng=rng,
        )
        patient_frames.append(
            _simulate_single_patient(
                profile=profile,
                timestamps=timestamps,
                frequency_minutes=frequency_minutes,
                rng=rng,
            )
        )

    df = pd.concat(patient_frames, ignore_index=True)
    df = _ensure_required_schema(df)
    df = _ensure_observable_artifacts(df, rng)
    df = df.sort_values(["timestamp", "patient_id"]).reset_index(drop=True)

    saved_path = save_simulated_data(df)
    print_simulation_summary(df, saved_path)
    return df


def save_simulated_data(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save simulated monitoring data to CSV.

    Relative paths are resolved from the project root so callers get consistent
    behavior whether the function is invoked from notebooks, scripts, or tests.
    """
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = _project_root() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_patient_monitoring_data(
    num_patients: int = 5,
    duration_hours: int = 24,
    frequency_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Backward-compatible convenience function for the simulator."""
    return simulate_patient_data(
        num_patients=num_patients,
        duration_hours=duration_hours,
        frequency_minutes=frequency_minutes,
        seed=seed,
    )


def print_simulation_summary(df: pd.DataFrame, saved_path: Path | None = None) -> None:
    """Print a compact summary of the generated dataset."""
    missing_values = int(df[VITAL_COLUMNS].isna().sum().sum())
    print("Simulation summary")
    print(f"- number of patients: {df['patient_id'].nunique()}")
    print(f"- total records: {len(df)}")
    print(f"- number of deterioration events: {int(df['deterioration_event'].sum())}")
    print(f"- number of noisy points: {int(df['sensor_noise_flag'].sum())}")
    print(f"- number of missing values: {missing_values}")
    if saved_path is not None:
        print(f"- saved CSV: {saved_path}")


def _validate_simulation_inputs(
    num_patients: int,
    duration_hours: int,
    frequency_minutes: int,
) -> None:
    """Validate public simulation inputs with clear errors."""
    if num_patients < 1:
        raise ValueError("num_patients must be at least 1.")
    if duration_hours < 0:
        raise ValueError("duration_hours must be zero or greater.")
    if frequency_minutes < 1:
        raise ValueError("frequency_minutes must be at least 1.")


def _build_timestamps(duration_hours: int, frequency_minutes: int) -> pd.DatetimeIndex:
    """Create a deterministic timeline for all synthetic patients."""
    total_minutes = duration_hours * 60
    num_steps = max(1, int(total_minutes // frequency_minutes) + 1)
    return pd.date_range(
        start=DEFAULT_START_TIMESTAMP,
        periods=num_steps,
        freq=f"{frequency_minutes}min",
    )


def _assign_behavior_types(
    num_patients: int,
    rng: np.random.Generator,
) -> list[str]:
    """Assign a trajectory type to each patient.

    When there are at least three patients, one of each trajectory is included
    before filling the rest randomly. This preserves random assignment while
    making typical demo datasets cover all required behavior types.
    """
    if num_patients >= len(BEHAVIOR_TYPES):
        assigned = list(BEHAVIOR_TYPES)
        remaining = num_patients - len(assigned)
        if remaining:
            assigned.extend(
                rng.choice(
                    BEHAVIOR_TYPES,
                    size=remaining,
                    p=[0.55, 0.30, 0.15],
                ).tolist()
            )
        rng.shuffle(assigned)
        return assigned

    return rng.choice(
        BEHAVIOR_TYPES,
        size=num_patients,
        p=[0.55, 0.30, 0.15],
    ).tolist()


def _create_patient_profile(
    patient_id: str,
    behavior_type: str,
    num_steps: int,
    rng: np.random.Generator,
) -> PatientProfile:
    """Create baseline vitals and trajectory settings for one patient."""
    baseline = _sample_baseline_vitals(rng)
    deterioration_start_index = _choose_deterioration_start(
        behavior_type=behavior_type,
        num_steps=num_steps,
        rng=rng,
    )

    intensity = float(rng.uniform(0.85, 1.25))
    total_deterioration_delta = {
        "heart_rate": float(rng.uniform(22.0, 42.0) * intensity),
        "oxygen_saturation": float(-rng.uniform(5.0, 12.0) * intensity),
        "systolic_bp": float(-rng.uniform(8.0, 24.0) * intensity),
        "diastolic_bp": float(-rng.uniform(4.0, 14.0) * intensity),
        "respiratory_rate": float(rng.uniform(5.0, 13.0) * intensity),
        "temperature": float(rng.uniform(0.3, 1.2) * intensity),
    }

    sudden_course = str(
        rng.choice(
            ["recovery", "persistent", "worsening"],
            p=[0.45, 0.30, 0.25],
        )
    )

    return PatientProfile(
        patient_id=patient_id,
        behavior_type=behavior_type,
        baseline=baseline,
        deterioration_start_index=deterioration_start_index,
        total_deterioration_delta=total_deterioration_delta,
        sudden_course=sudden_course,
    )


def _sample_baseline_vitals(rng: np.random.Generator) -> dict[str, float]:
    """Sample stable starting vitals for a synthetic patient."""
    return {
        "heart_rate": float(np.clip(rng.normal(74.0, 7.0), 55.0, 92.0)),
        "oxygen_saturation": float(np.clip(rng.normal(97.0, 1.0), 94.0, 99.5)),
        "systolic_bp": float(np.clip(rng.normal(120.0, 10.0), 100.0, 145.0)),
        "diastolic_bp": float(np.clip(rng.normal(76.0, 6.0), 60.0, 92.0)),
        "respiratory_rate": float(np.clip(rng.normal(15.5, 1.7), 11.0, 20.0)),
        "temperature": float(np.clip(rng.normal(36.8, 0.2), 36.1, 37.3)),
    }


def _choose_deterioration_start(
    behavior_type: str,
    num_steps: int,
    rng: np.random.Generator,
) -> int | None:
    """Choose when deterioration begins for a patient trajectory."""
    if behavior_type == "normal" or num_steps < 2:
        return None

    if behavior_type == "gradual_deterioration":
        lower = max(1, int(num_steps * 0.20))
        upper = max(lower + 1, int(num_steps * 0.55))
    else:
        lower = max(1, int(num_steps * 0.30))
        upper = max(lower + 1, int(num_steps * 0.75))

    return int(rng.integers(lower, min(upper, num_steps)))


def _simulate_single_patient(
    profile: PatientProfile,
    timestamps: pd.DatetimeIndex,
    frequency_minutes: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one patient's dependent time-series records."""
    current_vitals = dict(profile.baseline)
    missing_plan = _build_missing_plan(num_steps=len(timestamps), rng=rng)
    rows: list[dict[str, Any]] = []

    for step_index, timestamp in enumerate(timestamps):
        if step_index > 0:
            current_vitals = _advance_true_vitals(
                current_vitals=current_vitals,
                profile=profile,
                step_index=step_index,
                num_steps=len(timestamps),
                rng=rng,
            )

        true_vitals = _clip_true_vitals(current_vitals)
        severity_score = _calculate_severity_score(true_vitals)
        condition_label = _condition_label_from_score(true_vitals, severity_score)

        recorded_vitals, sensor_noise_flag = _apply_sensor_noise(true_vitals, rng)
        recorded_vitals, missing_data_flag = _apply_missing_data(
            recorded_vitals=recorded_vitals,
            step_index=step_index,
            missing_plan=missing_plan,
            rng=rng,
        )

        rows.append(
            {
                "patient_id": profile.patient_id,
                "timestamp": timestamp,
                **_round_vitals(recorded_vitals),
                "patient_condition_label": condition_label,
                "deterioration_event": False,
                "sensor_noise_flag": bool(sensor_noise_flag),
                "missing_data_flag": bool(missing_data_flag),
                "patient_outcome_after_alert": "unknown",
                "outcome_timestamp": None,
                "outcome_severity_change": None,
                "_severity_score": severity_score,
            }
        )

    _mark_deterioration_events(rows)
    _assign_outcomes(rows, frequency_minutes)

    for row in rows:
        row.pop("_severity_score", None)

    return pd.DataFrame(rows)


def _advance_true_vitals(
    current_vitals: dict[str, float],
    profile: PatientProfile,
    step_index: int,
    num_steps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Advance latent vitals using previous values, trends, and noise."""
    if profile.behavior_type == "normal" or profile.deterioration_start_index is None:
        updated = _move_toward_baseline(current_vitals, profile.baseline, rng)
    elif profile.behavior_type == "gradual_deterioration":
        updated = _advance_gradual_deterioration(
            current_vitals=current_vitals,
            profile=profile,
            step_index=step_index,
            num_steps=num_steps,
            rng=rng,
        )
    else:
        updated = _advance_sudden_deterioration(
            current_vitals=current_vitals,
            profile=profile,
            step_index=step_index,
            rng=rng,
        )

    return _clip_true_vitals(updated)


def _move_toward_baseline(
    current_vitals: dict[str, float],
    baseline: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Stable patients fluctuate but tend to return toward baseline."""
    noise_scale = {
        "heart_rate": 1.1,
        "oxygen_saturation": 0.18,
        "systolic_bp": 1.8,
        "diastolic_bp": 1.1,
        "respiratory_rate": 0.35,
        "temperature": 0.035,
    }
    return {
        vital: float(
            current_vitals[vital]
            + 0.08 * (baseline[vital] - current_vitals[vital])
            + rng.normal(0.0, noise_scale[vital])
        )
        for vital in VITAL_COLUMNS
    }


def _advance_gradual_deterioration(
    current_vitals: dict[str, float],
    profile: PatientProfile,
    step_index: int,
    num_steps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Apply a slow, explainable worsening trend after deterioration starts."""
    start_index = profile.deterioration_start_index or 0
    if step_index < start_index:
        return _move_toward_baseline(current_vitals, profile.baseline, rng)

    remaining_steps = max(1, num_steps - start_index)
    progress = min(1.0, (step_index - start_index + 1) / remaining_steps)
    instability_multiplier = 1.0 + progress
    base_noise = {
        "heart_rate": 1.2,
        "oxygen_saturation": 0.28,
        "systolic_bp": 2.4,
        "diastolic_bp": 1.5,
        "respiratory_rate": 0.45,
        "temperature": 0.04,
    }

    updated = {}
    for vital in VITAL_COLUMNS:
        trend_component = profile.total_deterioration_delta[vital] / remaining_steps
        random_noise = rng.normal(0.0, base_noise[vital] * instability_multiplier)
        updated[vital] = float(current_vitals[vital] + trend_component + random_noise)
    return updated


def _advance_sudden_deterioration(
    current_vitals: dict[str, float],
    profile: PatientProfile,
    step_index: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Apply an abrupt change, then recovery, persistence, or worsening."""
    start_index = profile.deterioration_start_index or 0
    if step_index < start_index:
        return _move_toward_baseline(current_vitals, profile.baseline, rng)

    if step_index == start_index:
        return {
            vital: float(
                current_vitals[vital]
                + profile.total_deterioration_delta[vital]
                + rng.normal(0.0, _sudden_noise_scale(vital))
            )
            for vital in VITAL_COLUMNS
        }

    if profile.sudden_course == "recovery":
        recovery_rate = 0.18
        return {
            vital: float(
                current_vitals[vital]
                + recovery_rate * (profile.baseline[vital] - current_vitals[vital])
                + rng.normal(0.0, _sudden_noise_scale(vital) * 0.55)
            )
            for vital in VITAL_COLUMNS
        }

    if profile.sudden_course == "persistent":
        return {
            vital: float(
                current_vitals[vital] + rng.normal(0.0, _sudden_noise_scale(vital))
            )
            for vital in VITAL_COLUMNS
        }

    worsening_fraction = 0.04
    return {
        vital: float(
            current_vitals[vital]
            + profile.total_deterioration_delta[vital] * worsening_fraction
            + rng.normal(0.0, _sudden_noise_scale(vital) * 1.2)
        )
        for vital in VITAL_COLUMNS
    }


def _sudden_noise_scale(vital: str) -> float:
    """Return per-vital noise scale for abrupt deterioration windows."""
    return {
        "heart_rate": 3.0,
        "oxygen_saturation": 0.8,
        "systolic_bp": 4.5,
        "diastolic_bp": 2.8,
        "respiratory_rate": 1.1,
        "temperature": 0.08,
    }[vital]


def _calculate_severity_score(vitals: dict[str, float]) -> float:
    """Calculate an engineering severity score from latent vitals."""
    score = 0.0
    score += max(0.0, vitals["heart_rate"] - 95.0) / 18.0
    score += max(0.0, 95.0 - vitals["oxygen_saturation"]) / 3.5
    score += max(0.0, 105.0 - vitals["systolic_bp"]) / 14.0
    score += max(0.0, vitals["systolic_bp"] - 155.0) / 25.0
    score += max(0.0, 62.0 - vitals["diastolic_bp"]) / 10.0
    score += max(0.0, vitals["diastolic_bp"] - 96.0) / 18.0
    score += max(0.0, vitals["respiratory_rate"] - 21.0) / 4.2
    score += max(0.0, vitals["temperature"] - 37.8) / 0.7
    return round(float(score), 4)


def _condition_label_from_score(vitals: dict[str, float], severity_score: float) -> str:
    """Translate latent vitals into a simple simulation condition label."""
    critical_signal = (
        vitals["heart_rate"] >= 135.0
        or vitals["oxygen_saturation"] <= 88.0
        or vitals["systolic_bp"] <= 85.0
        or vitals["respiratory_rate"] >= 32.0
        or severity_score >= 3.4
    )
    deteriorating_signal = (
        vitals["heart_rate"] >= 105.0
        or vitals["oxygen_saturation"] <= 93.5
        or vitals["systolic_bp"] <= 100.0
        or vitals["respiratory_rate"] >= 23.0
        or vitals["temperature"] >= 38.0
        or severity_score >= 1.2
    )

    if critical_signal:
        return "critical"
    if deteriorating_signal:
        return "deteriorating"
    return "normal"


def _apply_sensor_noise(
    true_vitals: dict[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, float], bool]:
    """Occasionally inject spikes or implausible sensor artifacts."""
    recorded_vitals = dict(true_vitals)
    if rng.random() >= 0.025:
        return recorded_vitals, False

    noisy_columns = rng.choice(VITAL_COLUMNS, size=int(rng.integers(1, 3)), replace=False)
    for vital in noisy_columns:
        recorded_vitals[str(vital)] = _artifact_value(str(vital), recorded_vitals[str(vital)], rng)
    return recorded_vitals, True


def _artifact_value(
    vital: str,
    current_value: float,
    rng: np.random.Generator,
) -> float:
    """Create a plausible monitoring artifact for one vital sign."""
    artifact_ranges = {
        "heart_rate": [(25.0, 38.0), (185.0, 240.0)],
        "oxygen_saturation": [(48.0, 74.0), (101.0, 110.0)],
        "systolic_bp": [(50.0, 72.0), (205.0, 245.0)],
        "diastolic_bp": [(25.0, 38.0), (120.0, 150.0)],
        "respiratory_rate": [(3.0, 7.0), (40.0, 58.0)],
        "temperature": [(32.0, 34.5), (41.5, 43.0)],
    }
    if rng.random() < 0.65:
        lower, upper = artifact_ranges[vital][int(rng.integers(0, 2))]
        return float(rng.uniform(lower, upper))

    spike_direction = float(rng.choice([-1.0, 1.0]))
    spike_size = {
        "heart_rate": rng.uniform(25.0, 70.0),
        "oxygen_saturation": rng.uniform(8.0, 25.0),
        "systolic_bp": rng.uniform(30.0, 80.0),
        "diastolic_bp": rng.uniform(18.0, 45.0),
        "respiratory_rate": rng.uniform(8.0, 25.0),
        "temperature": rng.uniform(1.2, 4.0),
    }[vital]
    return float(current_value + spike_direction * spike_size)


def _build_missing_plan(
    num_steps: int,
    rng: np.random.Generator,
) -> dict[int, list[str]]:
    """Plan random missing points and occasional short missing blocks."""
    missing_plan: dict[int, list[str]] = {}

    if num_steps >= 6 and rng.random() < 0.60:
        block_start = int(rng.integers(1, max(2, num_steps - 2)))
        block_length = int(rng.integers(2, min(7, max(3, num_steps - block_start + 1))))
        block_columns = rng.choice(
            VITAL_COLUMNS,
            size=int(rng.integers(1, 4)),
            replace=False,
        ).tolist()
        for step_index in range(block_start, min(num_steps, block_start + block_length)):
            missing_plan.setdefault(step_index, []).extend(str(col) for col in block_columns)

    return missing_plan


def _apply_missing_data(
    recorded_vitals: dict[str, float],
    step_index: int,
    missing_plan: dict[int, list[str]],
    rng: np.random.Generator,
) -> tuple[dict[str, float], bool]:
    """Set selected vital readings to NaN and flag the row."""
    missing_columns = set(missing_plan.get(step_index, []))

    if rng.random() < 0.018:
        point_missing_count = int(rng.integers(1, 4))
        missing_columns.update(
            str(col)
            for col in rng.choice(
                VITAL_COLUMNS,
                size=point_missing_count,
                replace=False,
            )
        )

    if not missing_columns:
        return recorded_vitals, False

    updated = dict(recorded_vitals)
    for column in missing_columns:
        updated[column] = np.nan
    return updated, True


def _mark_deterioration_events(rows: list[dict[str, Any]]) -> None:
    """Mark rows where latent severity significantly worsens."""
    previous_label = "normal"
    previous_score = 0.0

    for row in rows:
        label = str(row["patient_condition_label"])
        score = float(row["_severity_score"])
        label_worsened = _severity_rank(label) > _severity_rank(previous_label)
        significant_jump = score - previous_score >= 0.85

        row["deterioration_event"] = bool(
            label != "normal" and (label_worsened or significant_jump)
        )
        previous_label = label
        previous_score = score


def _assign_outcomes(rows: list[dict[str, Any]], frequency_minutes: int) -> None:
    """Attach simple post-alert outcome labels to deterioration events."""
    if len(rows) < 2:
        return

    min_horizon_steps = max(1, int(round(30 / frequency_minutes)))
    max_horizon_steps = max(min_horizon_steps, int(round(120 / frequency_minutes)))

    for index, row in enumerate(rows):
        if not row["deterioration_event"]:
            continue

        if index >= len(rows) - 1:
            row["patient_outcome_after_alert"] = "unknown"
            row["outcome_timestamp"] = None
            row["outcome_severity_change"] = None
            continue

        horizon_steps = min(max_horizon_steps, len(rows) - 1 - index)
        if horizon_steps < min_horizon_steps:
            outcome_index = len(rows) - 1
        else:
            outcome_index = min(len(rows) - 1, index + horizon_steps)

        current_score = float(row["_severity_score"])
        future_row = rows[outcome_index]
        future_score = float(future_row["_severity_score"])
        severity_change = round(future_score - current_score, 3)

        row["patient_outcome_after_alert"] = _outcome_from_severity_change(
            current_label=str(row["patient_condition_label"]),
            future_label=str(future_row["patient_condition_label"]),
            severity_change=severity_change,
        )
        row["outcome_timestamp"] = future_row["timestamp"]
        row["outcome_severity_change"] = severity_change


def _outcome_from_severity_change(
    current_label: str,
    future_label: str,
    severity_change: float,
) -> str:
    """Map a future severity change to a simulated outcome label."""
    if future_label == "normal" and current_label != "normal":
        return "improved"
    if current_label == "critical" and severity_change > -0.35:
        return "worsened"
    if severity_change <= -0.35:
        return "improved"
    if severity_change >= 0.35:
        return "worsened"
    return "unchanged"


def _severity_rank(label: str) -> int:
    """Return ordinal severity for condition label transitions."""
    return {"normal": 0, "deteriorating": 1, "critical": 2}[label]


def _clip_true_vitals(vitals: dict[str, float]) -> dict[str, float]:
    """Keep latent physiology in broad synthetic ranges."""
    bounds = {
        "heart_rate": (42.0, 165.0),
        "oxygen_saturation": (76.0, 99.8),
        "systolic_bp": (70.0, 210.0),
        "diastolic_bp": (35.0, 122.0),
        "respiratory_rate": (8.0, 45.0),
        "temperature": (35.0, 41.5),
    }
    return {
        vital: float(np.clip(value, bounds[vital][0], bounds[vital][1]))
        for vital, value in vitals.items()
    }


def _round_vitals(vitals: dict[str, float]) -> dict[str, float]:
    """Round recorded vitals while preserving NaN missing values."""
    rounded = {}
    for vital, value in vitals.items():
        if pd.isna(value):
            rounded[vital] = np.nan
        else:
            rounded[vital] = round(float(value), 2)
    return rounded


def _ensure_required_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all mandatory columns exist and are ordered consistently."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["patient_condition_label"] = pd.Categorical(
        df["patient_condition_label"],
        categories=CONDITION_LABELS,
    )
    df["patient_outcome_after_alert"] = pd.Categorical(
        df["patient_outcome_after_alert"],
        categories=OUTCOME_LABELS,
    )
    df["deterioration_event"] = df["deterioration_event"].astype(bool)
    df["sensor_noise_flag"] = df["sensor_noise_flag"].astype(bool)
    df["missing_data_flag"] = df["missing_data_flag"].astype(bool)
    return df


def _ensure_observable_artifacts(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Guarantee demo-sized datasets visibly contain noise and missingness.

    Random simulation can legitimately produce no artifacts in tiny datasets.
    For datasets with enough rows, this makes the simulator easier to verify
    while preserving deterministic behavior through the same RNG.
    """
    if len(df) < 3:
        return df

    df = df.copy()

    if not bool(df["sensor_noise_flag"].any()):
        noise_index = int(rng.integers(0, len(df)))
        vital = str(rng.choice(VITAL_COLUMNS))
        df.loc[noise_index, vital] = round(
            _artifact_value(vital, float(df.loc[noise_index, vital]), rng),
            2,
        )
        df.loc[noise_index, "sensor_noise_flag"] = True

    if not bool(df["missing_data_flag"].any()):
        missing_index = int(rng.integers(0, len(df)))
        vital = str(rng.choice(VITAL_COLUMNS))
        df.loc[missing_index, vital] = np.nan
        df.loc[missing_index, "missing_data_flag"] = True

    return df


def _project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]
