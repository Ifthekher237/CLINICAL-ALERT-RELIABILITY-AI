"""Baseline patient risk models for the clinical alert reliability prototype.

This module trains and evaluates two reusable baseline classifiers:

- Logistic Regression with StandardScaler for an interpretable linear baseline.
- Random Forest without scaling for a stronger non-linear baseline.

The models predict the engineered ``future_deterioration_label`` from Step 3
data. Columns that identify patients, encode timestamps, store text labels,
represent future outcomes, or duplicate target labels are removed before
training to avoid leakage.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "future_deterioration_label"
LEGACY_TARGET_COLUMN = "target_label"
TARGET_COLUMNS = {TARGET_COLUMN, LEGACY_TARGET_COLUMN}
RISK_CLASSES = [0, 1]
DEFAULT_PROCESSED_DATA_PATH = Path("data/processed/processed_data.csv")
DEFAULT_MODEL_DIR = Path("models")

IDENTIFIER_COLUMNS = {"patient_id"}
TIMESTAMP_COLUMNS = {"timestamp", "outcome_timestamp"}

# These values should not be model inputs. Some are direct target sources, and
# outcome fields describe future state after an alert.
LEAKAGE_COLUMNS = {
    "patient_condition_label",
    "deterioration_event",
    "patient_outcome_after_alert",
    "outcome_severity_change",
    *TARGET_COLUMNS,
}


class BaselineRiskModel:
    """Convenience wrapper for training and predicting with baseline models."""

    def __init__(self, model_type: str = "random_forest") -> None:
        if model_type not in {"logistic_regression", "random_forest"}:
            raise ValueError("model_type must be 'logistic_regression' or 'random_forest'.")
        self.model_type = model_type
        self.model: LogisticRegression | RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "BaselineRiskModel":
        """Train the selected baseline model."""
        if self.model_type == "logistic_regression":
            self.model, self.scaler = train_logistic_regression(X_train, y_train)
        else:
            self.model = train_random_forest(X_train, y_train)
            self.scaler = None
        return self

    def predict(self, X_input: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Predict numeric risk classes for new engineered feature rows."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return predict_risk(self.model, X_input, scaler=self.scaler)


def load_processed_data(path: str | Path = DEFAULT_PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the Step 3 processed dataset."""
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")
    return pd.read_csv(data_path)


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate model features from the numeric target label.

    The target is the future-looking ``future_deterioration_label``. Non-useful,
    string-based, timestamp, duplicate target, and leakage-prone columns are then
    removed from ``X``.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{TARGET_COLUMN}'. Regenerate processed data "
            "with src.data.preprocessing.prepare_modeling_data before training."
        )

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if y.isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing or non-numeric values.")

    drop_columns = _columns_to_drop_from_features(X)
    X = X.drop(columns=drop_columns, errors="ignore")

    # Keep only numeric and boolean model inputs. Any remaining strings or
    # categorical columns are excluded by design.
    X = X.select_dtypes(include=["number", "bool"]).copy()
    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
        selected_columns = feature_columns
    else:
        selected_columns = list(X.columns)

    return X, y.astype(int), selected_columns


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Create a leakage-safe, time-aware train/test split."""
    from src.data.preprocessing import train_test_split_time_series

    train_df, test_df = train_test_split_time_series(df, test_size=test_size)
    if train_df.empty or test_df.empty:
        raise ValueError("Time-aware split produced an empty train or test set.")

    X_train, y_train, feature_columns = prepare_features_and_target(train_df)
    X_test, y_test, _ = prepare_features_and_target(test_df, feature_columns=feature_columns)
    if y_test.nunique() < 2:
        warnings.warn(
            "Time-aware test split contains fewer than two target classes; "
            "classification metrics and AUROC may be unreliable.",
            stacklevel=2,
        )
    return X_train, X_test, y_train, y_test, feature_columns


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train a scaled Logistic Regression risk classifier."""
    _validate_trainable_target(y_train)
    X_clean, fill_values = _fit_feature_matrix(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, y_train.astype(int))

    _attach_model_metadata(
        model=model,
        feature_columns=list(X_clean.columns),
        fill_values=fill_values,
        requires_scaling=True,
        model_name="logistic_regression",
        scaler=scaler,
    )
    scaler.risk_model_feature_columns_ = list(X_clean.columns)
    scaler.risk_model_fill_values_ = fill_values
    return model, scaler


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train an unscaled Random Forest risk classifier."""
    _validate_trainable_target(y_train)
    X_clean, fill_values = _fit_feature_matrix(X_train)
    min_samples_leaf = 1 if len(X_clean) < 50 else 2

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_clean, y_train.astype(int))

    _attach_model_metadata(
        model=model,
        feature_columns=list(X_clean.columns),
        fill_values=fill_values,
        requires_scaling=False,
        model_name="random_forest",
        scaler=None,
    )
    return model


def evaluate_model(
    model: LogisticRegression | RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a trained model on held-out time-split data."""
    y_true = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(int)
    y_pred = predict_risk(model, X_test)
    auroc = _safe_multiclass_auroc(model, X_test, y_true)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=RISK_CLASSES).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=RISK_CLASSES,
            zero_division=0,
            output_dict=True,
        ),
        "auroc": auroc,
        "auroc_ovr_weighted": auroc,
    }
    metrics["warnings"] = _build_metric_warnings(metrics, y_true)
    return metrics


def predict_risk(
    model: LogisticRegression | RandomForestClassifier,
    X_input: pd.DataFrame | pd.Series | np.ndarray,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """Predict numeric risk classes for engineered input rows.

    If a scaler is provided or stored on the model, it is applied before
    prediction. Random Forest models are intentionally left unscaled.
    """
    X_prepared = _prepare_input_for_prediction(X_input, model=model, scaler=scaler)

    scaler_to_use = scaler or getattr(model, "risk_model_scaler_", None)
    requires_scaling = bool(getattr(model, "risk_model_requires_scaling_", False))
    if scaler_to_use is not None and requires_scaling:
        X_model = scaler_to_use.transform(X_prepared)
    else:
        X_model = X_prepared

    return model.predict(X_model)


def save_models(
    logistic_model: LogisticRegression,
    scaler: StandardScaler,
    random_forest_model: RandomForestClassifier,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Path]:
    """Save trained models and scaler with joblib."""
    output_dir = _resolve_project_path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "logistic_regression": output_dir / "logistic_regression.pkl",
        "random_forest": output_dir / "random_forest.pkl",
        "scaler": output_dir / "scaler.pkl",
    }
    joblib.dump(logistic_model, paths["logistic_regression"])
    joblib.dump(random_forest_model, paths["random_forest"])
    joblib.dump(scaler, paths["scaler"])
    return paths


def train_and_evaluate_models(
    data_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train, evaluate, compare, save, and sample-predict both baselines."""
    df = load_processed_data(data_path)
    X_train, X_test, y_train, y_test, feature_columns = prepare_train_test_data(
        df,
        test_size=test_size,
    )

    logistic_model, scaler = train_logistic_regression(X_train, y_train)
    random_forest_model = train_random_forest(X_train, y_train)

    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
    random_forest_metrics = evaluate_model(random_forest_model, X_test, y_test)
    saved_paths = save_models(logistic_model, scaler, random_forest_model, model_dir)

    sample_predictions = {
        "logistic_regression": predict_risk(logistic_model, X_test.head(5)).tolist(),
        "random_forest": predict_risk(random_forest_model, X_test.head(5)).tolist(),
        "actual": y_test.head(5).astype(int).tolist(),
    }
    train_distribution = y_train.value_counts().sort_index().to_dict()
    test_distribution = y_test.value_counts().sort_index().to_dict()
    split_warnings = _build_split_warnings(y_train, y_test)

    results = {
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "class_distribution_train": train_distribution,
        "class_distribution_test": test_distribution,
        "split_warnings": split_warnings,
        "logistic_regression": {
            "model": logistic_model,
            "scaler": scaler,
            "metrics": logistic_metrics,
        },
        "random_forest": {
            "model": random_forest_model,
            "metrics": random_forest_metrics,
        },
        "sample_predictions": sample_predictions,
        "saved_paths": saved_paths,
    }

    print_model_comparison(
        logistic_metrics,
        random_forest_metrics,
        sample_predictions,
        train_distribution=train_distribution,
        test_distribution=test_distribution,
        split_warnings=split_warnings,
    )
    print("Saved model artifacts:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")

    return results


def print_model_comparison(
    logistic_metrics: dict[str, Any],
    random_forest_metrics: dict[str, Any],
    sample_predictions: dict[str, list[int]] | None = None,
    train_distribution: dict[int, int] | None = None,
    test_distribution: dict[int, int] | None = None,
    split_warnings: list[str] | None = None,
) -> None:
    """Print a compact comparison of both baseline models."""
    if train_distribution is not None:
        print("Train target distribution:")
        print(train_distribution)
    if test_distribution is not None:
        print("Test target distribution:")
        print(test_distribution)
    if split_warnings:
        print("Split warnings:")
        for warning in split_warnings:
            print(f"- {warning}")

    print("Logistic Regression:")
    print(f"Accuracy: {logistic_metrics['accuracy']:.4f}")
    print(f"Precision: {logistic_metrics['precision_weighted']:.4f}")
    print(f"Recall: {logistic_metrics['recall_weighted']:.4f}")
    print(f"F1: {logistic_metrics['f1_weighted']:.4f}")
    print("Confusion Matrix:")
    print(np.array(logistic_metrics["confusion_matrix"]))
    if logistic_metrics.get("auroc") is not None:
        print(f"AUROC: {logistic_metrics['auroc']:.4f}")
    if logistic_metrics.get("warnings"):
        print("Warnings:")
        for warning in logistic_metrics["warnings"]:
            print(f"- {warning}")

    print("\nRandom Forest:")
    print(f"Accuracy: {random_forest_metrics['accuracy']:.4f}")
    print(f"Precision: {random_forest_metrics['precision_weighted']:.4f}")
    print(f"Recall: {random_forest_metrics['recall_weighted']:.4f}")
    print(f"F1: {random_forest_metrics['f1_weighted']:.4f}")
    print("Confusion Matrix:")
    print(np.array(random_forest_metrics["confusion_matrix"]))
    if random_forest_metrics.get("auroc") is not None:
        print(f"AUROC: {random_forest_metrics['auroc']:.4f}")
    if random_forest_metrics.get("warnings"):
        print("Warnings:")
        for warning in random_forest_metrics["warnings"]:
            print(f"- {warning}")

    if sample_predictions is not None:
        print("\nSample Predictions:")
        print(f"Actual: {sample_predictions['actual']}")
        print(f"Logistic Regression: {sample_predictions['logistic_regression']}")
        print(f"Random Forest: {sample_predictions['random_forest']}")


def _columns_to_drop_from_features(X: pd.DataFrame) -> list[str]:
    """Identify non-useful and leakage-prone columns for removal."""
    string_columns = set(X.select_dtypes(include=["object", "string", "category"]).columns)
    datetime_columns = set(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
    explicit_drop_columns = (
        IDENTIFIER_COLUMNS | TIMESTAMP_COLUMNS | LEAKAGE_COLUMNS | datetime_columns
    )
    return sorted(string_columns | explicit_drop_columns)


def _build_split_warnings(y_train: pd.Series, y_test: pd.Series) -> list[str]:
    """Return warnings for class distributions that weaken evaluation."""
    warnings_list = []
    if pd.Series(y_train).nunique() < 2:
        warnings_list.append("Train target distribution contains fewer than two classes.")
    if pd.Series(y_test).nunique() < 2:
        warnings_list.append("Test target distribution contains fewer than two classes.")
    return warnings_list


def _build_metric_warnings(metrics: dict[str, Any], y_true: pd.Series) -> list[str]:
    """Flag evaluation results that look too good for a simulated prototype."""
    warnings_list = []
    if pd.Series(y_true).nunique() < 2:
        warnings_list.append("Evaluation target contains fewer than two classes.")
    if metrics["accuracy"] >= 0.995 and metrics["f1_weighted"] >= 0.995:
        warnings_list.append(
            "Accuracy and F1 are suspiciously perfect; inspect target leakage and class balance."
        )
    if metrics.get("auroc") is not None and metrics["auroc"] >= 0.995:
        warnings_list.append(
            "AUROC is suspiciously perfect; inspect whether the prediction task is too easy."
        )
    return warnings_list


def _fit_feature_matrix(X_train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit training-only missing-value fill values and return cleaned features."""
    X_clean = _coerce_numeric_features(X_train)
    fill_values = X_clean.median(numeric_only=True).fillna(0.0).to_dict()
    X_clean = X_clean.fillna(fill_values).fillna(0.0)
    return X_clean, {column: float(value) for column, value in fill_values.items()}


def _transform_feature_matrix(
    X_input: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    """Apply training feature columns and fill values to new data."""
    X_clean = _coerce_numeric_features(X_input)
    X_clean = X_clean.reindex(columns=feature_columns, fill_value=0.0)
    X_clean = X_clean.fillna(fill_values).fillna(0.0)
    return X_clean


def _coerce_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert booleans to integers and remove numeric infinities."""
    X_clean = X.copy()
    for column in X_clean.select_dtypes(include=["bool"]).columns:
        X_clean[column] = X_clean[column].astype(int)
    return X_clean.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prepare_input_for_prediction(
    X_input: pd.DataFrame | pd.Series | np.ndarray,
    model: LogisticRegression | RandomForestClassifier,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Align arbitrary input data to the model's training feature schema."""
    if isinstance(X_input, pd.Series):
        X_frame = X_input.to_frame().T
    elif isinstance(X_input, pd.DataFrame):
        X_frame = X_input.copy()
    else:
        feature_columns = _get_feature_columns(model, scaler)
        X_array = np.asarray(X_input)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        if feature_columns and X_array.shape[1] == len(feature_columns):
            X_frame = pd.DataFrame(X_array, columns=feature_columns)
        else:
            X_frame = pd.DataFrame(X_array)

    if TARGET_COLUMN in X_frame.columns:
        X_frame = X_frame.drop(columns=[TARGET_COLUMN])

    drop_columns = _columns_to_drop_from_features(X_frame)
    X_frame = X_frame.drop(columns=drop_columns, errors="ignore")

    feature_columns = _get_feature_columns(model, scaler)
    fill_values = _get_fill_values(model, scaler)

    if not feature_columns:
        feature_columns = list(X_frame.select_dtypes(include=["number", "bool"]).columns)

    return _transform_feature_matrix(X_frame, feature_columns, fill_values)


def _get_feature_columns(
    model: LogisticRegression | RandomForestClassifier,
    scaler: StandardScaler | None,
) -> list[str]:
    """Read feature columns stored on the model or scaler."""
    columns = getattr(model, "risk_model_feature_columns_", None)
    if columns is None and scaler is not None:
        columns = getattr(scaler, "risk_model_feature_columns_", None)
    return list(columns) if columns is not None else []


def _get_fill_values(
    model: LogisticRegression | RandomForestClassifier,
    scaler: StandardScaler | None,
) -> dict[str, float]:
    """Read training fill values stored on the model or scaler."""
    values = getattr(model, "risk_model_fill_values_", None)
    if values is None and scaler is not None:
        values = getattr(scaler, "risk_model_fill_values_", None)
    return dict(values) if values is not None else {}


def _safe_multiclass_auroc(
    model: LogisticRegression | RandomForestClassifier,
    X_test: pd.DataFrame,
    y_true: pd.Series,
) -> float | None:
    """Compute AUROC when the class structure supports it."""
    if not hasattr(model, "predict_proba") or y_true.nunique() < 2:
        return None

    try:
        X_prepared = _prepare_input_for_prediction(X_test, model=model)
        if bool(getattr(model, "risk_model_requires_scaling_", False)):
            scaler = getattr(model, "risk_model_scaler_", None)
            if scaler is None:
                return None
            X_model = scaler.transform(X_prepared)
        else:
            X_model = X_prepared

        probabilities = model.predict_proba(X_model)
        model_classes = list(getattr(model, "classes_", []))
        present_classes = sorted(pd.Series(y_true).dropna().astype(int).unique().tolist())

        if len(present_classes) == 2:
            positive_class = present_classes[-1]
            if positive_class not in model_classes:
                return None
            positive_index = model_classes.index(positive_class)
            binary_target = (y_true == positive_class).astype(int)
            return float(roc_auc_score(binary_target, probabilities[:, positive_index]))

        if set(present_classes) != set(model_classes):
            return None

        return float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=model_classes,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        return None


def _attach_model_metadata(
    model: LogisticRegression | RandomForestClassifier,
    feature_columns: list[str],
    fill_values: dict[str, float],
    requires_scaling: bool,
    model_name: str,
    scaler: StandardScaler | None,
) -> None:
    """Store preprocessing metadata directly on trained models."""
    model.risk_model_feature_columns_ = feature_columns
    model.risk_model_fill_values_ = fill_values
    model.risk_model_requires_scaling_ = requires_scaling
    model.risk_model_name_ = model_name
    if scaler is not None:
        model.risk_model_scaler_ = scaler


def _validate_trainable_target(y_train: pd.Series) -> None:
    """Raise clear errors for tiny or single-class training targets."""
    if len(y_train) < 2:
        raise ValueError("At least two training rows are required.")
    if pd.Series(y_train).nunique() < 2:
        raise ValueError("At least two target classes are required for training.")


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return _project_root() / path_obj


def _project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]
