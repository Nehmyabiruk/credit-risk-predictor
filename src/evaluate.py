"""
evaluate.py
-----------
Standalone evaluation of a trained credit-risk model.

Loads the full pipeline + a hold-out set (or a user-supplied CSV),
computes classification metrics, ROC-AUC, PR-AUC, confusion matrix,
and optionally writes a metrics JSON + plots.

Usage:
    python -m src.evaluate
    python -m src.evaluate --test-csv path/to/test.csv --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.train import CATEGORICAL, NUMERIC, RANDOM_STATE, TEST_SIZE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "loan_data_clean.csv"
PIPELINE_PATH = ROOT / "models" / "full_pipeline.pkl"
REPORT_DIR = ROOT / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_holdout(
    clean_path: Path = CLEAN_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Re-create the exact same stratified hold-out used at training time."""
    df = pd.read_csv(clean_path)
    y = df["loan_status"]
    X = df.drop(columns=["loan_status"])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
    )
    return X_test, y_test


def load_external(csv_path: Path, target_col: str = "loan_status"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in {csv_path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": report["accuracy"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "support_0": int(report["0"]["support"]),
        "support_1": int(report["1"]["support"]),
        "threshold": None,  # filled by caller
    }
    return metrics


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default (0)", "Default (1)"],
        yticklabels=["No Default (0)", "Default (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


def plot_roc(y_true, y_proba, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", out_path)


def plot_pr(y_true, y_proba, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curve → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(
    pipeline_path: Path = PIPELINE_PATH,
    test_csv: Path | None = None,
    threshold: float = 0.5,
    save_plots: bool = True,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {pipeline_path}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(pipeline_path)
    logger.info("Loaded pipeline from %s", pipeline_path)

    if test_csv is not None:
        X_test, y_test = load_external(test_csv)
        logger.info("Evaluating on external CSV: %s (%d rows)", test_csv, len(X_test))
    else:
        X_test, y_test = load_holdout()
        logger.info("Evaluating on stratified hold-out (%d rows)", len(X_test))

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    metrics["threshold"] = threshold

    # Pretty print
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))
    logger.info("ROC-AUC : %.4f", metrics["roc_auc"])
    logger.info("PR-AUC  : %.4f", metrics["pr_auc"])
    logger.info("Threshold used: %.2f", threshold)

    # Persist metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", METRICS_PATH)

    if save_plots:
        plot_confusion(y_test, y_pred, REPORT_DIR / "confusion_matrix.png")
        plot_roc(y_test, y_proba, REPORT_DIR / "roc_curve.png")
        plot_pr(y_test, y_proba, REPORT_DIR / "pr_curve.png")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained credit-risk model")
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=PIPELINE_PATH,
        help="Path to full_pipeline.pkl",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Optional external test CSV (must contain loan_status)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for positive class (default 0.5)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving ROC / PR / confusion plots",
    )
    args = parser.parse_args()

    evaluate(
        pipeline_path=args.pipeline,
        test_csv=args.test_csv,
        threshold=args.threshold,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
