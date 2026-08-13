"""Benchmark the calibrated observed-event pipelines used in the manuscript.

The benchmark uses the same 33-feature panel, model factories, scaling,
calibration, and five-fold issuer-grouped evaluation as reanalysis_oof.py. It
reports model-fitting time separately from held-out probability inference time.
Bootstrap confidence intervals, SHAP, lead-time calculations, and file I/O are
excluded from the timed regions.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import cmdf_tree_classify as classification


MODELS = ("XGBoost", "CatBoost", "LightGBM")
N_SPLITS = 5
SEED = 42


def _version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def benchmark(repeats: int, inference_repeats: int) -> tuple[pd.DataFrame, dict]:
    panel, X, y, _ = classification.load_panel(verbose=True)
    A = X.to_numpy(float)
    yv = y.to_numpy(int)
    groups = panel["issuer_code"].to_numpy()
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    splits = list(cv.split(A, yv, groups))
    factories = classification.classifiers(seed=SEED)
    rows: list[dict] = []

    for model_name in MODELS:
        if model_name not in factories:
            raise RuntimeError(f"Required model is unavailable: {model_name}")
        print(f"\n{model_name}")
        for repeat in range(1, repeats + 1):
            for fold, (train_idx, test_idx) in enumerate(splits, 1):
                gc.collect()
                train_start = time.perf_counter()
                scaler = StandardScaler().fit(A[train_idx])
                train_X = scaler.transform(A[train_idx])
                estimator = CalibratedClassifierCV(
                    estimator=factories[model_name](), method="sigmoid", cv=3
                )
                estimator.fit(train_X, yv[train_idx])
                train_seconds = time.perf_counter() - train_start

                test_X = scaler.transform(A[test_idx])
                # Warm up lazy allocations before timing repeated inference.
                estimator.predict_proba(test_X)
                inference_samples = []
                for _ in range(inference_repeats):
                    infer_start = time.perf_counter()
                    estimator.predict_proba(test_X)
                    inference_samples.append(time.perf_counter() - infer_start)
                infer_seconds = float(np.median(inference_samples))

                rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_seconds": train_seconds,
                        "inference_seconds": infer_seconds,
                        "inference_ms_per_1000": (
                            infer_seconds * 1_000_000.0 / len(test_idx)
                        ),
                    }
                )
                print(
                    f"  repeat {repeat} fold {fold}: train {train_seconds:.3f}s, "
                    f"inference {infer_seconds * 1000:.3f}ms "
                    f"for {len(test_idx):,} rows"
                )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("model", sort=False)
        .agg(
            median_train_seconds_per_fold=("train_seconds", "median"),
            mean_train_seconds_per_fold=("train_seconds", "mean"),
            median_inference_ms_per_1000=("inference_ms_per_1000", "median"),
            mean_inference_ms_per_1000=("inference_ms_per_1000", "mean"),
            measured_fits=("train_seconds", "size"),
        )
        .reset_index()
    )
    # Five folds are required to create one complete OOF probability vector.
    summary["estimated_five_fold_train_seconds"] = (
        summary["median_train_seconds_per_fold"] * N_SPLITS
    )

    metadata = {
        "run_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "sklearn": _version("sklearn"),
        "xgboost": _version("xgboost"),
        "lightgbm": _version("lightgbm"),
        "catboost": _version("catboost"),
        "issuer_months": int(len(panel)),
        "issuers": int(panel["issuer_code"].nunique()),
        "features": int(X.shape[1]),
        "positive_rows": int(y.sum()),
        "outer_folds": N_SPLITS,
        "calibration_folds": 3,
        "repeats": repeats,
        "inference_repeats": inference_repeats,
        "timed_scope": (
            "training includes scaling, base learner fitting, and sigmoid calibration; "
            "inference includes held-out predict_proba only"
        ),
    }
    return raw, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inference-repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1 or args.inference_repeats < 1:
        parser.error("repeat counts must be positive")

    raw, summary, metadata = benchmark(args.repeats, args.inference_repeats)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tex_out")
    os.makedirs(out_dir, exist_ok=True)
    raw.to_csv(os.path.join(out_dir, "runtime_benchmark_folds.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "runtime_benchmark_summary.csv"), index=False)
    with open(
        os.path.join(out_dir, "runtime_benchmark_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMetadata")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
