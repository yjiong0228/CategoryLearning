from __future__ import annotations

import argparse
import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunSample:
    subject_id: int
    condition: int
    run_index: int
    mean_error: float
    params: dict[str, Any]
    trajectory: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFT-based clustering for run-level accuracy trajectories")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with subject_*.json and cache/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <input-dir>/analysis)")
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default="sliding_pred_acc",
        help="Metrics key used as trajectory, e.g. sliding_pred_acc or sliding_true_acc",
    )
    parser.add_argument(
        "--fft-keep-ratio",
        type=float,
        default=0.2,
        help="Low-frequency ratio to keep from rfft bins when --fft-keep-bins is not set",
    )
    parser.add_argument(
        "--fft-keep-bins",
        type=int,
        default=None,
        help="Explicit number of low-frequency bins to keep",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("kmeans", "agglomerative", "dbscan", "gmm", "dpmm"),
        default="kmeans",
        help="Clustering method",
    )
    parser.add_argument("--n-clusters", type=int, default=4, help="Cluster count for kmeans/agglomerative/gmm")
    parser.add_argument("--dbscan-eps", type=float, default=0.8, help="DBSCAN eps")
    parser.add_argument("--dbscan-min-samples", type=int, default=8, help="DBSCAN min_samples")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for kmeans/gmm")
    parser.add_argument("--dp-max-components", type=int, default=12, help="Max mixture components for truncated DPMM")
    parser.add_argument(
        "--dp-weight-concentration-prior",
        type=float,
        default=1.0,
        help="DP concentration prior for BayesianGaussianMixture",
    )
    parser.add_argument(
        "--dp-covariance-type",
        type=str,
        choices=("full", "diag"),
        default="full",
        help="Covariance type for BayesianGaussianMixture",
    )
    parser.add_argument("--dp-max-iter", type=int, default=1000, help="Max iterations for DPMM variational fit")
    parser.add_argument("--dp-n-init", type=int, default=3, help="Init count for DPMM variational fit")
    return parser.parse_args()


def _iter_pickle_gz(path: Path):
    with gzip.open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json payload in {path}")
    return data


def _to_float_array(values: Any, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{context} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{context} is empty")
    if np.isnan(arr).all():
        raise ValueError(f"{context} contains only NaN")
    if np.isnan(arr).any():
        fill_value = float(np.nanmean(arr))
        arr = np.where(np.isnan(arr), fill_value, arr)
    return arr


def load_run_samples(input_dir: Path, trajectory_key: str) -> list[RunSample]:
    samples: list[RunSample] = []
    subject_files = sorted(input_dir.glob("subject_*.json"))
    if not subject_files:
        raise RuntimeError(f"No subject_*.json found in {input_dir}")

    for file in subject_files:
        payload = _load_json(file)
        schema_version = int(payload.get("schema_version", 0))
        if schema_version < 3:
            raise ValueError(f"{file} is schema_version={schema_version}, expected >= 3")
        raw_runs_ref = payload.get("raw_runs_ref")
        if not isinstance(raw_runs_ref, dict) or "path" not in raw_runs_ref:
            raise ValueError(f"{file} missing raw_runs_ref")

        cache_path = (input_dir / str(raw_runs_ref["path"])).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing cache file: {cache_path}")

        for run_obj in _iter_pickle_gz(cache_path):
            if not isinstance(run_obj, dict):
                raise ValueError(f"Unexpected run object type in {cache_path}")
            metrics = run_obj.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"Run missing metrics in {cache_path}")
            if trajectory_key not in metrics:
                raise ValueError(f"Run missing metrics['{trajectory_key}'] in {cache_path}")

            trajectory = _to_float_array(
                metrics[trajectory_key],
                context=f"{cache_path} trajectory {trajectory_key}",
            )
            samples.append(
                RunSample(
                    subject_id=int(run_obj["subject_id"]),
                    condition=int(run_obj["condition"]),
                    run_index=int(run_obj["run_index"]),
                    mean_error=float(run_obj["mean_error"]),
                    params=dict(run_obj.get("params", {})),
                    trajectory=trajectory,
                )
            )

    if not samples:
        raise RuntimeError(f"No run samples loaded from {input_dir}")
    return samples


def build_feature_matrix(
    samples: list[RunSample],
    fft_keep_ratio: float,
    fft_keep_bins: int | None,
) -> tuple[np.ndarray, int, int]:
    min_len = min(len(s.trajectory) for s in samples)
    if min_len < 2:
        raise ValueError(f"Trajectory too short for FFT: min_len={min_len}")

    clipped_items = []
    for sample in samples:
        sample.trajectory = sample.trajectory[:min_len]
        clipped_items.append(sample.trajectory)
    clipped = np.stack(clipped_items, axis=0)
    fft_mag = np.abs(np.fft.rfft(clipped, axis=1))
    total_bins = int(fft_mag.shape[1])

    if fft_keep_bins is not None:
        keep_bins = int(fft_keep_bins)
    else:
        if fft_keep_ratio <= 0 or fft_keep_ratio > 1:
            raise ValueError(f"fft_keep_ratio must be in (0,1], got {fft_keep_ratio}")
        keep_bins = int(np.ceil(total_bins * fft_keep_ratio))
    keep_bins = max(1, min(keep_bins, total_bins))

    X = fft_mag[:, :keep_bins].astype(float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    X = (X - mean) / std
    return X, min_len, keep_bins


def cluster_features(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    random_state: int,
    dp_max_components: int,
    dp_weight_concentration_prior: float,
    dp_covariance_type: str,
    dp_max_iter: int,
    dp_n_init: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cluster_probabilities: np.ndarray
    model_info: dict[str, Any] = {"method": method}
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        cluster_probabilities = np.eye(n_clusters, dtype=float)[labels]
        return labels, cluster_probabilities, model_info
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        cluster_probabilities = np.eye(n_clusters, dtype=float)[labels]
        return labels, cluster_probabilities, model_info
    elif method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X)
        unique_labels = sorted(set(int(x) for x in labels))
        label_to_col = {label: i for i, label in enumerate(unique_labels)}
        cluster_probabilities = np.zeros((len(labels), len(unique_labels)), dtype=float)
        for i, label in enumerate(labels):
            cluster_probabilities[i, label_to_col[int(label)]] = 1.0
        model_info["label_space"] = unique_labels
        return labels, cluster_probabilities, model_info
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        model.fit(X)
        cluster_probabilities = model.predict_proba(X)
        labels = cluster_probabilities.argmax(axis=1).astype(int)
        return labels, cluster_probabilities, model_info
    elif method == "dpmm":
        model = BayesianGaussianMixture(
            n_components=dp_max_components,
            covariance_type=dp_covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=dp_weight_concentration_prior,
            max_iter=dp_max_iter,
            n_init=dp_n_init,
            random_state=random_state,
        )
        model.fit(X)
        cluster_probabilities = model.predict_proba(X)
        labels = cluster_probabilities.argmax(axis=1).astype(int)
        model_info.update(
            {
                "active_weight_threshold": 1e-3,
                "component_weights": model.weights_.astype(float).tolist(),
                "active_components": int(np.sum(model.weights_ > 1e-3)),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
            }
        )
        return labels, cluster_probabilities, model_info
    else:
        raise ValueError(f"Unsupported method: {method}")


def save_outputs(
    output_dir: Path,
    samples: list[RunSample],
    X: np.ndarray,
    labels: np.ndarray,
    pca_2d: np.ndarray,
    cluster_probabilities: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "fft_features.npy", X)

    assignment_rows = []
    for i, sample in enumerate(samples):
        assignment_rows.append(
            {
                "sample_id": i,
                "subject_id": sample.subject_id,
                "condition": sample.condition,
                "run_index": sample.run_index,
                "mean_error": sample.mean_error,
                "cluster_label": int(labels[i]),
                "cluster_confidence": float(np.max(cluster_probabilities[i])),
                "cluster_prob_json": json.dumps(cluster_probabilities[i].astype(float).tolist(), ensure_ascii=False),
                "params_json": json.dumps(sample.params, ensure_ascii=False, sort_keys=True),
            }
        )
    pd.DataFrame(assignment_rows).to_csv(output_dir / "cluster_assignments.csv", index=False)

    embedding_rows = []
    for i, sample in enumerate(samples):
        embedding_rows.append(
            {
                "sample_id": i,
                "subject_id": sample.subject_id,
                "condition": sample.condition,
                "run_index": sample.run_index,
                "cluster_label": int(labels[i]),
                "pca_x": float(pca_2d[i, 0]),
                "pca_y": float(pca_2d[i, 1]),
            }
        )
    pd.DataFrame(embedding_rows).to_csv(output_dir / "embedding_pca_2d.csv", index=False)


def plot_cluster_scatter(output_dir: Path, pca_2d: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(int(x) for x in labels))
    for label in unique_labels:
        mask = labels == label
        legend = f"cluster {label}" if label != -1 else "noise(-1)"
        plt.scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=22, alpha=0.8, label=legend)
    plt.title("FFT Feature Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_scatter.png", dpi=180)
    plt.close()


def _cluster_center_representative(
    X: np.ndarray,
    labels: np.ndarray,
    target_label: int,
) -> int:
    idx = np.where(labels == target_label)[0]
    center = X[idx].mean(axis=0, keepdims=True)
    dist = np.linalg.norm(X[idx] - center, axis=1)
    return int(idx[int(np.argmin(dist))])


def plot_cluster_mean_trajectories(output_dir: Path, samples: list[RunSample], labels: np.ndarray) -> None:
    unique_labels = sorted(set(int(x) for x in labels))
    plt.figure(figsize=(12, 7))
    for label in unique_labels:
        mask = labels == label
        traj = np.stack([samples[i].trajectory for i in np.where(mask)[0]], axis=0)
        mean = traj.mean(axis=0)
        std = traj.std(axis=0)
        x = np.arange(mean.shape[0])
        plt.plot(x, mean, label=f"cluster {label} (n={traj.shape[0]})")
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.title("Cluster Mean Accuracy Trajectories")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_mean_trajectories.png", dpi=180)
    plt.close()


def plot_cluster_representatives(
    output_dir: Path,
    samples: list[RunSample],
    labels: np.ndarray,
    X: np.ndarray,
) -> None:
    unique_labels = sorted(set(int(x) for x in labels))
    plt.figure(figsize=(12, 7))
    for label in unique_labels:
        rep_idx = _cluster_center_representative(X, labels, label)
        traj = samples[rep_idx].trajectory
        x = np.arange(traj.shape[0])
        plt.plot(x, traj, label=f"cluster {label} rep: sub{samples[rep_idx].subject_id}-run{samples[rep_idx].run_index}")
    plt.title("Representative Accuracy Trajectories by Cluster")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_representative_trajectories.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = (args.output_dir or (input_dir / "analysis")).resolve()

    samples = load_run_samples(input_dir, args.trajectory_key)
    subject_ids = sorted(set(s.subject_id for s in samples))
    all_assignment_frames: list[pd.DataFrame] = []
    all_embedding_frames: list[pd.DataFrame] = []
    subject_meta: list[dict[str, Any]] = []

    for sid in subject_ids:
        subject_samples = [s for s in samples if s.subject_id == sid]
        subject_output_dir = output_dir / f"subject_{sid}"
        X, min_len, keep_bins = build_feature_matrix(
            subject_samples,
            args.fft_keep_ratio,
            args.fft_keep_bins,
        )
        labels, cluster_probabilities, model_info = cluster_features(
            X=X,
            method=args.method,
            n_clusters=args.n_clusters,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            random_state=args.random_state,
            dp_max_components=args.dp_max_components,
            dp_weight_concentration_prior=args.dp_weight_concentration_prior,
            dp_covariance_type=args.dp_covariance_type,
            dp_max_iter=args.dp_max_iter,
            dp_n_init=args.dp_n_init,
        )
        pca_2d = PCA(n_components=2).fit_transform(X)

        save_outputs(subject_output_dir, subject_samples, X, labels, pca_2d, cluster_probabilities)
        plot_cluster_scatter(subject_output_dir, pca_2d, labels)
        plot_cluster_mean_trajectories(subject_output_dir, subject_samples, labels)
        plot_cluster_representatives(subject_output_dir, subject_samples, labels, X)

        with (subject_output_dir / "clustering_report.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": args.method,
                    "params": {
                        "n_clusters": args.n_clusters,
                        "dbscan_eps": args.dbscan_eps,
                        "dbscan_min_samples": args.dbscan_min_samples,
                        "random_state": args.random_state,
                        "dp_max_components": args.dp_max_components,
                        "dp_weight_concentration_prior": args.dp_weight_concentration_prior,
                        "dp_covariance_type": args.dp_covariance_type,
                        "dp_max_iter": args.dp_max_iter,
                        "dp_n_init": args.dp_n_init,
                    },
                    "num_samples": len(subject_samples),
                    "num_clusters_found": int(len(set(int(x) for x in labels))),
                    "model_info": model_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        assign_df = pd.read_csv(subject_output_dir / "cluster_assignments.csv")
        embed_df = pd.read_csv(subject_output_dir / "embedding_pca_2d.csv")
        all_assignment_frames.append(assign_df)
        all_embedding_frames.append(embed_df)
        subject_meta.append(
            {
                "subject_id": int(sid),
                "num_samples": int(len(subject_samples)),
                "trajectory_min_length": int(min_len),
                "fft_keep_bins": int(keep_bins),
                "num_clusters_found": int(len(set(int(x) for x in labels))),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_assignment_frames, ignore_index=True).to_csv(
        output_dir / "cluster_assignments_all_subjects.csv",
        index=False,
    )
    pd.concat(all_embedding_frames, ignore_index=True).to_csv(
        output_dir / "embedding_pca_2d_all_subjects.csv",
        index=False,
    )

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_samples": int(len(samples)),
        "num_subjects": int(len(subject_ids)),
        "trajectory_key": args.trajectory_key,
        "method": args.method,
        "n_clusters": int(args.n_clusters),
        "subjects": subject_meta,
    }
    with (output_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote per-subject analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
