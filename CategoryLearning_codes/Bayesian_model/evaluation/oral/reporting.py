"""口述—模型对齐结果的汇总、序列化与绘图。"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.anova import AnovaRM
except ImportError:  # pragma: no cover - optional dependency in some environments.
    AnovaRM = None

from .scoring import OralAlignmentScoringMixin


logger = logging.getLogger(__name__)


class OralAlignmentReportingMixin(OralAlignmentScoringMixin):
    """在纯计算结果之上提供汇总、保存与作图方法。"""

    @staticmethod
    def _subjectwise_grid_layout(
        subjects,
        n_cols,
        *,
        panel_width=8.0,
        panel_height=5.0,
    ):
        """Return a subject-wise grid using the oral-mass plot layout scale."""
        n_subjects = len(subjects)
        if n_subjects <= 0:
            raise RuntimeError("No subjects available for subject-wise plot.")

        requested_cols = max(1, int(n_cols))
        actual_cols = max(1, min(requested_cols, n_subjects))
        n_rows = int(np.ceil(n_subjects / actual_cols))
        return n_rows, actual_cols, (actual_cols * float(panel_width), n_rows * float(panel_height))

    def _style_subjectwise_grid_axes(self, axes, n_rows, n_cols, ylabel, xlabel="Normalized trial"):
        axes = np.asarray(axes)
        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=self.SUBJECTWISE_TICK_FONTSIZE)
        for row in range(n_rows):
            axes[row, 0].set_ylabel(ylabel, fontsize=self.SUBJECTWISE_LABEL_FONTSIZE)
        for col in range(n_cols):
            axes[-1, col].set_xlabel(xlabel, fontsize=self.SUBJECTWISE_LABEL_FONTSIZE)

    @staticmethod
    def _sem(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return np.nan
        return float(np.std(arr, ddof=1) / np.sqrt(arr.size))

    @staticmethod
    def _rolling_mean(values, window_size=16):
        """Rolling mean for subject plots with a tolerant valid-sample rule."""
        window = max(1, int(window_size))
        min_periods = max(1, window // 4)
        return pd.Series(values, dtype=float).rolling(window=window, min_periods=min_periods).mean().to_numpy()

    def _attach_target_sampling_bands(
        self,
        target_based_results,
        *,
        window_size=16,
        n_draws=OralAlignmentScoringMixin.DEFAULT_TARGET_BAND_DRAWS,
        base_seed=OralAlignmentScoringMixin.DEFAULT_TARGET_BAND_SEED,
    ):
        """Attach backend-appropriate rolling target intervals to trial rows."""
        df = target_based_results.copy()
        band_columns = (
            "model_target_expected_rolling",
            "model_target_q05_rolling",
            "model_target_q25_rolling",
            "model_target_q50_rolling",
            "model_target_q75_rolling",
            "model_target_q95_rolling",
        )
        for column in band_columns:
            if column not in df:
                df[column] = np.nan
        metadata_defaults = {
            "model_target_band_type": "",
            "model_target_band_n_draws": np.nan,
            "model_target_band_n_runs": np.nan,
            "model_target_band_base_seed": np.nan,
            "model_target_band_subject_seed": np.nan,
            "model_target_band_window_size": int(window_size),
        }
        for column, default in metadata_defaults.items():
            if column not in df:
                df[column] = default

        group_fields = ["subject"]
        if "alignment_space" in df.columns:
            group_fields.append("alignment_space")
        for group_key, group in df.groupby(group_fields, sort=True):
            sid = int(group_key[0] if isinstance(group_key, tuple) else group_key)
            subject_seed = int(
                np.random.SeedSequence([int(base_seed), sid]).generate_state(1)[0]
            )
            ordered = group.sort_values("trial")
            backend = (
                str(ordered["model_inference_backend"].dropna().iloc[0])
                if "model_inference_backend" in ordered
                and not ordered["model_inference_backend"].dropna().empty
                else "particle_filter"
            )
            if backend == "trajectory":
                precomputed_type = ordered["model_target_band_type"].astype(str)
                precomputed_window = pd.to_numeric(
                    ordered["model_target_band_window_size"],
                    errors="coerce",
                )
                has_precomputed_band = (
                    precomputed_type.eq(self.TRAJECTORY_TARGET_BAND_TYPE).all()
                    and precomputed_window.eq(int(window_size)).all()
                    and pd.to_numeric(
                        ordered["model_target_band_n_runs"],
                        errors="coerce",
                    ).notna().all()
                )
                if has_precomputed_band:
                    continue
                if "model_target_repeat_probabilities" in ordered:
                    repeat_rows = [
                        np.asarray(values, dtype=float).reshape(-1)
                        for values in ordered[
                            "model_target_repeat_probabilities"
                        ].tolist()
                    ]
                    run_counts = {row.size for row in repeat_rows}
                    if len(run_counts) != 1 or not run_counts or 0 in run_counts:
                        raise ValueError(
                            "Trajectory target repeats disagree on run count."
                        )
                    probability_runs = np.vstack(repeat_rows).T
                else:
                    probability_runs = ordered[
                        "model_target_prior"
                    ].to_numpy(dtype=float)[None, :]
                band = self.compute_trajectory_target_band(
                    probability_runs,
                    window_size=window_size,
                )
            else:
                band = self.compute_target_sampling_band(
                    ordered["model_target_prior"].to_numpy(dtype=float),
                    window_size=window_size,
                    n_draws=n_draws,
                    seed=subject_seed,
                )
            index = ordered.index
            df.loc[index, "model_target_expected_rolling"] = band["expected"]
            for quantile in ("q05", "q25", "q50", "q75", "q95"):
                df.loc[index, f"model_target_{quantile}_rolling"] = band[quantile]
            df.loc[index, "model_target_band_type"] = band["band_type"]
            if backend == "trajectory":
                df.loc[index, "model_target_band_n_runs"] = band["n_runs"]
            else:
                df.loc[index, "model_target_band_n_draws"] = band["n_draws"]
                df.loc[index, "model_target_band_base_seed"] = int(base_seed)
                df.loc[index, "model_target_band_subject_seed"] = subject_seed
        return df

    @staticmethod
    def _format_p_value(p_value):
        try:
            p = float(p_value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(p):
            return "n/a"
        if p < 0.001:
            return "<.001"
        return f"={p:.3f}"

    @staticmethod
    def _oral_encoder_label(frame):
        """Return a compact encoder label for probability-space figures."""
        df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        if df.empty or "oral_distribution_method" not in df:
            return ""
        methods = df["oral_distribution_method"].dropna()
        if methods.empty:
            return ""
        method = str(methods.iloc[0])
        aggregation = ""
        if "oral_aggregation_method" in df:
            aggregations = df["oral_aggregation_method"].dropna()
            if not aggregations.empty:
                aggregation = str(aggregations.iloc[0])
        aggregation = {
            "latest_by_category_likelihood_product": "latest-category joint",
            "current_report_only": "current-report only",
        }.get(aggregation, aggregation)
        modes = df["oral_mode"].dropna() if "oral_mode" in df else pd.Series(dtype=str)
        mode = str(modes.iloc[0]) if not modes.empty else ""
        scale_column = "oral_center_sigma" if mode == "center" else "oral_region_temperature"
        scale_name = "sigma" if mode == "center" else "temperature"
        if scale_column not in df:
            return f"{aggregation}; {method}" if aggregation else method
        values = pd.to_numeric(df[scale_column], errors="coerce").dropna()
        if values.empty:
            return f"{aggregation}; {method}" if aggregation else method
        encoder = f"{method}, {scale_name}={float(values.iloc[0]):g}"
        return f"{aggregation}; {encoder}" if aggregation else encoder

    @staticmethod
    def _safe_pearson(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            return np.nan
        return float(stats.pearsonr(x, y).statistic)

    @staticmethod
    def _safe_spearman(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            return np.nan
        return float(stats.spearmanr(x, y).statistic)

    @staticmethod
    def _safe_cosine_similarity(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        x = x[mask]
        y = y[mask]
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-12:
            return np.nan
        return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))

    @staticmethod
    def _safe_cohen_kappa(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        xb = x[mask] > 0.5
        yb = y[mask] > 0.5
        observed = float(np.mean(xb == yb))
        px = float(np.mean(xb))
        py = float(np.mean(yb))
        expected = px * py + (1.0 - px) * (1.0 - py)
        denom = 1.0 - expected
        if denom <= 1e-12:
            return np.nan
        return float((observed - expected) / denom)

    @staticmethod
    def _safe_binary_jaccard(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        xb = x[mask] > 0.5
        yb = y[mask] > 0.5
        union = int(np.sum(xb | yb))
        if union <= 0:
            return np.nan
        return float(np.sum(xb & yb) / union)

    @staticmethod
    def _holm_adjust_pvalues(p_values):
        """Holm-adjust a sequence of p-values while preserving NaNs."""
        p = np.asarray(p_values, dtype=float)
        adjusted = np.full(p.shape, np.nan, dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(p))
        if finite_idx.size == 0:
            return adjusted

        ordered = finite_idx[np.argsort(p[finite_idx])]
        m = int(ordered.size)
        running_max = 0.0
        for rank, idx in enumerate(ordered):
            candidate = min(1.0, float(p[idx]) * float(m - rank))
            running_max = max(running_max, candidate)
            adjusted[idx] = running_max
        return adjusted

    @classmethod
    def _paired_distribution_space_stats(cls, subject_space_means, spaces):
        """Return compact paired statistics for the group-level bar panel."""
        pivot = subject_space_means.reindex(columns=list(spaces))
        complete = pivot.dropna()
        n_complete = int(len(complete))

        friedman_p = np.nan
        if n_complete >= 3 and len(spaces) >= 3:
            try:
                samples = [complete[space].to_numpy(dtype=float) for space in spaces]
                friedman_p = float(stats.friedmanchisquare(*samples).pvalue)
            except ValueError:
                friedman_p = np.nan

        pairs = []
        p_values = []
        for left_idx in range(len(spaces)):
            for right_idx in range(left_idx + 1, len(spaces)):
                left = spaces[left_idx]
                right = spaces[right_idx]
                pair = pivot[[left, right]].dropna()
                p_val = np.nan
                if len(pair) >= 2:
                    try:
                        p_val = float(stats.wilcoxon(pair[left], pair[right], zero_method="wilcox").pvalue)
                    except ValueError:
                        diff = pair[left].to_numpy(dtype=float) - pair[right].to_numpy(dtype=float)
                        p_val = 1.0 if np.allclose(diff, 0.0, equal_nan=False) else np.nan
                pairs.append((left, right))
                p_values.append(p_val)

        adjusted = cls._holm_adjust_pvalues(p_values)
        pair_text = []
        for (left, right), p_adj in zip(pairs, adjusted):
            left_label = cls.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(left, left)
            right_label = cls.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(right, right)
            pair_text.append(f"{left_label}-{right_label} p{cls._format_p_value(p_adj)}")

        return {
            "n": n_complete,
            "friedman_p": friedman_p,
            "pair_text": "; ".join(pair_text),
        }

    @classmethod
    def _distribution_space_time_stats(cls, distribution_results, spaces, bins=20):
        """Run a two-way repeated-measures ANOVA over space and normalized time bins."""
        if AnovaRM is None:
            return {"n": 0, "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        df = distribution_results.copy()
        df = df[df["alignment_space"].isin(spaces)]
        df = df[np.isfinite(df["js_similarity"])]
        if df.empty:
            return {"n": 0, "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "alignment_space", "trial_bin"], observed=True)["js_similarity"]
            .mean()
            .reset_index()
        )

        expected_cells = int(len(spaces) * int(bins))
        counts = subject_bin.groupby("subject").size()
        complete_subjects = counts[counts == expected_cells].index
        complete = subject_bin[subject_bin["subject"].isin(complete_subjects)].copy()
        if complete["subject"].nunique() < 3:
            return {"n": int(complete["subject"].nunique()), "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        complete["alignment_space"] = complete["alignment_space"].astype(str)
        complete["trial_bin"] = complete["trial_bin"].astype(str)

        try:
            fit = AnovaRM(
                complete,
                depvar="js_similarity",
                subject="subject",
                within=["alignment_space", "trial_bin"],
            ).fit()
        except Exception:
            return {"n": int(complete["subject"].nunique()), "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        table = fit.anova_table
        out = {
            "n": int(complete["subject"].nunique()),
            "space_p": np.nan,
            "time_p": np.nan,
            "interaction_p": np.nan,
        }
        for idx, row in table.iterrows():
            idx_str = str(idx)
            p_val = float(row.get("Pr > F", np.nan))
            if idx_str == "alignment_space":
                out["space_p"] = p_val
            elif idx_str == "trial_bin":
                out["time_p"] = p_val
            elif "alignment_space" in idx_str and "trial_bin" in idx_str:
                out["interaction_p"] = p_val
        return out

    @staticmethod
    def save_oral_mass_probabilities(oral_mass_results, save_path):
        """Save full oral_t hypothesis distributions to a compressed npz file."""
        results = {int(k): v for k, v in oral_mass_results.items()}
        if not results:
            raise RuntimeError("No oral mass results to save.")

        subjects = np.asarray(sorted(results), dtype=int)
        max_trials = max(np.asarray(results[sid]["oral_mass"]).shape[0] for sid in subjects)
        max_hypos = max(np.asarray(results[sid]["oral_mass"]).shape[1] for sid in subjects)
        oral_mass = np.full((len(subjects), max_trials, max_hypos), np.nan, dtype=float)
        instantaneous_oral_mass = np.full(
            (len(subjects), max_trials, max_hypos),
            np.nan,
            dtype=float,
        )
        valid_oral = np.zeros((len(subjects), max_trials), dtype=bool)
        valid_oral_report = np.zeros((len(subjects), max_trials), dtype=bool)
        n_trials = np.zeros(len(subjects), dtype=int)
        n_hypos = np.zeros(len(subjects), dtype=int)
        conditions = np.zeros(len(subjects), dtype=int)
        target_hypos = np.zeros(len(subjects), dtype=int)
        oral_modes = []
        region_stimulus_sigmas = np.full(len(subjects), np.nan, dtype=float)
        encoder_metadata = {
            field: []
            for field in OralAlignmentScoringMixin.ORAL_ENCODER_METADATA_FIELDS
        }
        trial_diagnostics = {
            field: np.full((len(subjects), max_trials), np.nan, dtype=float)
            for field in OralAlignmentScoringMixin.ORAL_TRIAL_DIAGNOSTIC_FIELDS
        }

        for row_idx, sid in enumerate(subjects):
            info = results[int(sid)]
            arr = np.asarray(info["oral_mass"], dtype=float)
            trials, hypos = arr.shape
            oral_mass[row_idx, :trials, :hypos] = arr
            instantaneous = np.asarray(
                info.get("instantaneous_oral_mass", arr),
                dtype=float,
            )
            if instantaneous.ndim == 2:
                inst_trials = min(trials, instantaneous.shape[0])
                inst_hypos = min(hypos, instantaneous.shape[1])
                instantaneous_oral_mass[
                    row_idx,
                    :inst_trials,
                    :inst_hypos,
                ] = instantaneous[:inst_trials, :inst_hypos]
            valid = np.asarray(info.get("valid_oral", []), dtype=bool).reshape(-1)
            valid_oral[row_idx, : min(trials, valid.size)] = valid[:trials]
            report_valid = np.asarray(
                info.get("valid_oral_report", valid),
                dtype=bool,
            ).reshape(-1)
            valid_oral_report[
                row_idx,
                : min(trials, report_valid.size),
            ] = report_valid[:trials]
            n_trials[row_idx] = trials
            n_hypos[row_idx] = hypos
            conditions[row_idx] = int(info.get("condition"))
            target_hypos[row_idx] = int(info.get("target_hypo"))
            oral_modes.append(str(info.get("oral_mode", "")))
            for field in OralAlignmentScoringMixin.ORAL_ENCODER_METADATA_FIELDS:
                encoder_metadata[field].append(info.get(field, np.nan))
            for field in OralAlignmentScoringMixin.ORAL_TRIAL_DIAGNOSTIC_FIELDS:
                values = np.asarray(info.get(field, []), dtype=float).reshape(-1)
                trial_diagnostics[field][row_idx, : min(trials, values.size)] = values[:trials]
            try:
                region_stimulus_sigmas[row_idx] = float(info.get("region_stimulus_sigma", np.nan))
            except (TypeError, ValueError):
                region_stimulus_sigmas[row_idx] = np.nan

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            subjects=subjects,
            conditions=conditions,
            target_hypos=target_hypos,
            n_trials=n_trials,
            n_hypos=n_hypos,
            valid_oral=valid_oral,
            valid_oral_report=valid_oral_report,
            oral_mass=oral_mass,
            instantaneous_oral_mass=instantaneous_oral_mass,
            oral_modes=np.asarray(oral_modes, dtype=str),
            region_stimulus_sigmas=region_stimulus_sigmas,
            **{
                field: (
                    np.asarray(values, dtype=float)
                    if field in {"oral_center_sigma", "oral_region_temperature"}
                    else np.asarray(values, dtype=str)
                )
                for field, values in encoder_metadata.items()
            },
            **trial_diagnostics,
        )
        logger.info("Oral mass probabilities saved to %s", save_path)
        return save_path

    @staticmethod
    def load_oral_mass_probabilities(path):
        """Load oral_t hypothesis distributions saved by ``save_oral_mass_probabilities``."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            subjects = data["subjects"].astype(int)
            conditions = data["conditions"].astype(int)
            target_hypos = data["target_hypos"].astype(int)
            n_trials = data["n_trials"].astype(int)
            n_hypos = data["n_hypos"].astype(int)
            valid_oral = data["valid_oral"].astype(bool)
            oral_mass = data["oral_mass"].astype(float)
            valid_oral_report = (
                data["valid_oral_report"].astype(bool)
                if "valid_oral_report" in data.files
                else valid_oral.copy()
            )
            instantaneous_oral_mass = (
                data["instantaneous_oral_mass"].astype(float)
                if "instantaneous_oral_mass" in data.files
                else oral_mass.copy()
            )
            oral_modes = data["oral_modes"].astype(str) if "oral_modes" in data.files else np.asarray([""] * len(subjects))
            region_stimulus_sigmas = (
                data["region_stimulus_sigmas"].astype(float)
                if "region_stimulus_sigmas" in data.files
                else np.full(len(subjects), np.nan, dtype=float)
            )
            encoder_metadata = {}
            for field in OralAlignmentScoringMixin.ORAL_ENCODER_METADATA_FIELDS:
                dtype = float if field in {"oral_center_sigma", "oral_region_temperature"} else str
                if field in data.files:
                    encoder_metadata[field] = data[field].astype(dtype)
                else:
                    default = np.nan if field in {"oral_center_sigma", "oral_region_temperature"} else ""
                    encoder_metadata[field] = np.asarray([default] * len(subjects), dtype=dtype)
            trial_diagnostics = {
                field: (
                    data[field].astype(float)
                    if field in data.files
                    else np.full((len(subjects), oral_mass.shape[1]), np.nan, dtype=float)
                )
                for field in OralAlignmentScoringMixin.ORAL_TRIAL_DIAGNOSTIC_FIELDS
            }

            out = {}
            for row_idx, sid in enumerate(subjects):
                trials = int(n_trials[row_idx])
                hypos = int(n_hypos[row_idx])
                out[int(sid)] = {
                    "iSub": int(sid),
                    "condition": int(conditions[row_idx]),
                    "target_hypo": int(target_hypos[row_idx]),
                    "oral_mode": str(oral_modes[row_idx]),
                    "region_stimulus_sigma": float(region_stimulus_sigmas[row_idx]),
                    **{
                        field: (
                            float(values[row_idx])
                            if field in {"oral_center_sigma", "oral_region_temperature"}
                            else str(values[row_idx])
                        )
                        for field, values in encoder_metadata.items()
                    },
                    **{
                        field: values[row_idx, :trials].copy()
                        for field, values in trial_diagnostics.items()
                    },
                    "oral_mass": oral_mass[row_idx, :trials, :hypos].copy(),
                    "valid_oral": valid_oral[row_idx, :trials].tolist(),
                    "instantaneous_oral_mass": instantaneous_oral_mass[
                        row_idx,
                        :trials,
                        :hypos,
                    ].copy(),
                    "valid_oral_report": valid_oral_report[
                        row_idx,
                        :trials,
                    ].tolist(),
                }
        return out

    @staticmethod
    def save_oral_mass_diagnostics(oral_mass_results, save_path):
        """Write inspectable trial-level oral encoder diagnostics."""
        rows = []
        for sid, info in sorted((int(k), v) for k, v in oral_mass_results.items()):
            mass = np.asarray(info.get("oral_mass"), dtype=float)
            valid = np.asarray(info.get("valid_oral", []), dtype=bool).reshape(-1)
            instantaneous = np.asarray(
                info.get("instantaneous_oral_mass", mass),
                dtype=float,
            )
            report_valid = np.asarray(
                info.get("valid_oral_report", valid),
                dtype=bool,
            ).reshape(-1)
            target_hypo = int(info.get("target_hypo", -1))
            for trial_idx in range(mass.shape[0]):
                row = {
                    "iSub": sid,
                    "subject": sid,
                    "condition": int(info.get("condition")),
                    "trial": trial_idx + 1,
                    "oral_mode": str(info.get("oral_mode", "")),
                    "target_hypo": target_hypo,
                    "valid_oral": bool(valid[trial_idx]) if trial_idx < valid.size else False,
                    "valid_oral_state": (
                        bool(valid[trial_idx]) if trial_idx < valid.size else False
                    ),
                    "valid_oral_report": (
                        bool(report_valid[trial_idx])
                        if trial_idx < report_valid.size
                        else False
                    ),
                    "oral_target_mass": (
                        float(mass[trial_idx, target_hypo])
                        if 0 <= target_hypo < mass.shape[1] and np.isfinite(mass[trial_idx, target_hypo])
                        else np.nan
                    ),
                    "instantaneous_oral_target_mass": (
                        float(instantaneous[trial_idx, target_hypo])
                        if (
                            instantaneous.ndim == 2
                            and trial_idx < instantaneous.shape[0]
                            and 0 <= target_hypo < instantaneous.shape[1]
                            and np.isfinite(instantaneous[trial_idx, target_hypo])
                        )
                        else np.nan
                    ),
                }
                row.update(
                    {
                        field: info.get(field, np.nan)
                        for field in OralAlignmentScoringMixin.ORAL_ENCODER_METADATA_FIELDS
                    }
                )
                for field in OralAlignmentScoringMixin.ORAL_TRIAL_DIAGNOSTIC_FIELDS:
                    values = np.asarray(info.get(field, []), dtype=float).reshape(-1)
                    row[field] = float(values[trial_idx]) if trial_idx < values.size else np.nan
                rows.append(row)

        diagnostics = pd.DataFrame(rows)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(save_path, index=False)
        logger.info("Oral mass diagnostics saved to %s", save_path)
        return save_path

    @staticmethod
    def save_oral_equivalence_group_outputs(
        lookup_df,
        trial_df,
        output_dir,
        prefix="oral_equivalence",
    ):
        """Save oral-equivalence lookup/trial tables and a readable report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lookup_path = output_dir / f"{prefix}_group_lookup.csv"
        multi_path = output_dir / f"{prefix}_multi_hypothesis_groups.csv"
        trial_path = output_dir / f"{prefix}_trial_groups.csv"
        report_path = output_dir / f"{prefix}_group_report.md"

        lookup = lookup_df.copy()
        trial = trial_df.copy()
        multi = lookup[lookup["n_hypotheses"].astype(int) > 1].copy() if not lookup.empty else pd.DataFrame()

        lookup.to_csv(lookup_path, index=False)
        multi.to_csv(multi_path, index=False)
        trial.to_csv(trial_path, index=False)

        lines = [
            "# Oral-Equivalence Hypothesis Groups",
            "",
            "Hypotheses are grouped when they have the same oral representation for the current choice.",
            "Use `oral_equivalence_trial_groups.csv` to map each trial to a choice-level lookup key,",
            "and `oral_equivalence_group_lookup.csv` to inspect every group under that key.",
            "",
        ]
        if lookup.empty:
            lines.append("No grouping rows were generated.")
        else:
            for (condition, oral_mode, choice), sub in lookup.groupby(
                ["condition", "oral_mode", "choice"],
                observed=True,
            ):
                n_groups = int(len(sub))
                n_multi = int(np.sum(sub["n_hypotheses"].astype(int) > 1))
                max_size = int(sub["n_hypotheses"].astype(int).max())
                lines.extend(
                    [
                        f"## Condition {int(condition)}, {oral_mode}, choice {int(choice)}",
                        "",
                        f"- groups: {n_groups}",
                        f"- multi-hypothesis groups: {n_multi}",
                        f"- max group size: {max_size}",
                        "",
                    ]
                )
                multi_sub = sub[sub["n_hypotheses"].astype(int) > 1]
                if multi_sub.empty:
                    lines.extend(["No multi-hypothesis groups for this choice.", ""])
                    continue
                lines.append("| group_id | n | hypotheses | target_in_group |")
                lines.append("| --- | ---: | --- | --- |")
                for _, row in multi_sub.iterrows():
                    lines.append(
                        "| "
                        f"{int(row['group_id'])} | "
                        f"{int(row['n_hypotheses'])} | "
                        f"`{row['hypotheses']}` | "
                        f"{bool(row['target_in_group'])} |"
                    )
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "lookup": lookup_path,
            "multi_hypothesis_groups": multi_path,
            "trial_groups": trial_path,
            "report": report_path,
        }

    def plot_oral_mass_probabilities(
        self,
        oral_mass_results,
        subjects=None,
        save_path=None,
        limit=True,
        mass_key="oral_mass",
        title="Oral Category-State Hypothesis Probability by Subject",
        ylabel="Oral Hypothesis Probability",
        target_label="target",
        **kwargs,
    ):
        """Plot trial-by-hypothesis or trial-by-group mass, matching posterior.png layout."""
        results = self._filter_results(oral_mass_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info["condition"]].append((iSub, info))

        if not grouped:
            raise RuntimeError("No oral mass results to plot.")

        first_info = next(iter(results.values()))
        if mass_key == "oral_mass" and "oral_distribution_method" in first_info:
            method = str(first_info.get("oral_distribution_method", ""))
            aggregation = str(first_info.get("oral_aggregation_method", ""))
            aggregation = {
                "latest_by_category_likelihood_product": "latest-category joint",
                "current_report_only": "current-report only",
            }.get(aggregation, aggregation)
            mode = str(first_info.get("oral_mode", ""))
            if mode == "center":
                scale_label = f"sigma={float(first_info.get('oral_center_sigma', np.nan)):g}"
            else:
                scale_label = (
                    f"temperature={float(first_info.get('oral_region_temperature', np.nan)):g}"
                )
            method_label = f"{method}, {scale_label}"
            if aggregation:
                method_label = f"{aggregation}; {method_label}"
            title = f"{title} ({method_label})"

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(
            title,
            fontsize=kwargs.get("fontsize", 16),
            y=kwargs.get("y", 0.99),
        )

        scatter_size = kwargs.get("scatter_size", 3)
        alpha = kwargs.get("alpha", 0.28)
        cmap = kwargs.get("cmap", "viridis")

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, (iSub, info) in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)

                oral_mass = np.asarray(info.get(mass_key), dtype=float)
                if oral_mass.ndim != 2 or oral_mass.size == 0:
                    ax.text(0.5, 0.5, f"No {ylabel.lower()} data", ha="center", va="center", transform=ax.transAxes)
                    continue

                max_k = oral_mass.shape[1]

                mass = oral_mass[:, :max_k]
                n_trials, n_hypos = mass.shape
                x = np.repeat(np.arange(1, n_trials + 1), n_hypos)
                k = np.tile(np.arange(n_hypos), n_trials)
                y = mass.reshape(-1)
                finite = np.isfinite(y)

                if np.any(finite):
                    ax.scatter(
                        x[finite],
                        y[finite],
                        c=k[finite],
                        cmap=cmap,
                        vmin=0,
                        vmax=max(1, n_hypos - 1),
                        s=scatter_size,
                        alpha=alpha,
                        linewidths=0,
                        rasterized=True,
                    )

                    target_hypo = int(info.get("target_hypo", 0 if int(condition) == 1 else 42))
                    target_group = np.asarray(info.get("target_group_per_trial", []), dtype=float).reshape(-1)
                    if target_group.size >= n_trials:
                        target_x = []
                        target_y = []
                        for trial_idx in range(n_trials):
                            group_idx = int(target_group[trial_idx])
                            if 0 <= group_idx < n_hypos and np.isfinite(mass[trial_idx, group_idx]):
                                target_x.append(trial_idx + 1)
                                target_y.append(float(mass[trial_idx, group_idx]))
                        if target_x:
                            ax.scatter(
                                target_x,
                                target_y,
                                color="red",
                                s=max(12, scatter_size * 4),
                                alpha=0.85,
                                linewidths=0,
                                label=f"{target_label} group",
                            )
                    elif 0 <= target_hypo < n_hypos:
                        target_y = mass[:, target_hypo]
                        target_x = np.arange(1, n_trials + 1)
                        target_mask = np.isfinite(target_y)
                        ax.scatter(
                            target_x[target_mask],
                            target_y[target_mask],
                            color="red",
                            s=max(12, scatter_size * 4),
                            alpha=0.85,
                            linewidths=0,
                            label=f"{target_label} k={target_hypo}",
                        )

                    y_max = float(np.nanmax(y[finite]))
                    ax.set_ylim(0, min(1.0, max(0.02, y_max * 1.12)))
                else:
                    ax.text(0.5, 0.5, "No valid oral mass", ha="center", va="center", transform=ax.transAxes)

                ax.set(
                    title=f"Subject {iSub} (Condition {condition})",
                    xlabel="Trial",
                    ylabel=ylabel,
                )
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()

            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("%s plot saved to %s", title, save_path)
        return fig

    def plot_model_distribution_probabilities(
        self,
        model_mass_results,
        model_distribution="prior",
        subjects=None,
        save_path=None,
        limit=True,
        **kwargs,
    ):
        """Plot model belief distributions with the oral-mass plot style."""
        state = str(model_distribution).strip().lower()
        mass_key = f"{state}_mass"
        title = kwargs.pop("title", f"Model {state.capitalize()} for k by Subject")
        ylabel = kwargs.pop("ylabel", f"{state.capitalize()} Probability")
        return self.plot_oral_mass_probabilities(
            model_mass_results,
            subjects=subjects,
            save_path=save_path,
            limit=limit,
            mass_key=mass_key,
            title=title,
            ylabel=ylabel,
            target_label="target",
            **kwargs,
        )

    @staticmethod
    def summarize_distribution_alignment_by_bin(distribution_results, bins=20):
        """Return subject-balanced binned means and SEMs for JS similarity."""
        df = distribution_results.copy()
        if df.empty:
            return pd.DataFrame()

        df = df[np.isfinite(df["js_similarity"])]
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "alignment_space", "alignment_label", "trial_bin"], observed=True)[
                "js_similarity"
            ]
            .mean()
            .reset_index()
        )

        rows = []
        for (space, label, trial_bin), group in subject_bin.groupby(
            ["alignment_space", "alignment_label", "trial_bin"],
            observed=True,
        ):
            values = group["js_similarity"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "alignment_space": str(space),
                    "alignment_label": str(label),
                    "trial_bin": int(trial_bin),
                    "trial_pct": (int(trial_bin) - 0.5) / float(bins),
                    "js_similarity_mean": float(np.mean(values)) if values.size else np.nan,
                    "js_similarity_sem": OralAlignmentReportingMixin._sem(values),
                    "n_subjects": int(values.size),
                }
            )
        return pd.DataFrame(rows)

    def plot_distribution_alignment_group(
        self,
        distribution_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level distribution alignment summary and time course."""
        df = distribution_results.copy()
        if df.empty:
            raise RuntimeError("No distribution alignment results to plot.")

        spaces = [space for space in self.DISTRIBUTION_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            raise RuntimeError("No supported distribution alignment spaces to plot.")

        subject_space_means = (
            df.groupby(["subject", "alignment_space"], observed=True)["js_similarity"]
            .mean()
            .unstack("alignment_space")
            .reindex(columns=spaces)
        )
        binned = self.summarize_distribution_alignment_by_bin(df, bins=bins)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        fig_title = title or f"Condition {condition_label}: oral vs model {model_state} distribution alignment"
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        x = np.arange(len(spaces), dtype=float)
        means = [float(np.nanmean(subject_space_means[space].to_numpy(dtype=float))) for space in spaces]
        sems = [self._sem(subject_space_means[space].to_numpy(dtype=float)) for space in spaces]
        colors = [self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555") for space in spaces]
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.82, capsize=4, edgecolor="white", linewidth=0.8)

        rng = np.random.default_rng(123)
        for subject, row in subject_space_means.iterrows():
            vals = row.to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if np.sum(finite) >= 2:
                ax.plot(x[finite], vals[finite], color="#888888", alpha=0.22, lw=0.8, zorder=1)
            jitter = rng.normal(0.0, 0.035, size=len(spaces))
            ax.scatter(
                x[finite] + jitter[finite],
                vals[finite],
                s=18,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )

        labels = [
            "Full\nhypothesis\nspace" if space == "full" else
            "Model\nactive set" if space == "active" else
            "Active +\noral top-N\nunion"
            for space in spaces
        ]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("JS similarity")
        ax.set_title("Subject means")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)

        stats_bar = self._paired_distribution_space_stats(subject_space_means, spaces)
        ax.text(
            0.02,
            0.98,
            (
                f"Friedman p{self._format_p_value(stats_bar['friedman_p'])}, "
                f"n={stats_bar['n']}\n{stats_bar['pair_text']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        ax = axes[1]
        for space in spaces:
            sub = binned[binned["alignment_space"] == space].sort_values("trial_bin")
            if sub.empty:
                continue
            line_x = sub["trial_pct"].to_numpy(dtype=float)
            mean = sub["js_similarity_mean"].to_numpy(dtype=float)
            sem = sub["js_similarity_sem"].to_numpy(dtype=float)
            self._line_with_sem(
                ax,
                line_x,
                mean,
                sem,
                self.DISTRIBUTION_ALIGNMENT_LABELS.get(space, space),
                self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555"),
            )

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("JS similarity")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        stats_time = self._distribution_space_time_stats(df, spaces, bins=bins)
        ax.text(
            0.02,
            0.02,
            (
                f"RM-ANOVA n={stats_time['n']}\n"
                f"space p{self._format_p_value(stats_time['space_p'])}; "
                f"time p{self._format_p_value(stats_time['time_p'])}; "
                f"space x time p{self._format_p_value(stats_time['interaction_p'])}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Distribution alignment group plot saved to %s", save_path)
        return fig

    def plot_distribution_based_alignment_group(self, *args, **kwargs):
        """Alias for the distribution-based group plot."""
        return self.plot_distribution_alignment_group(*args, **kwargs)

    def plot_distribution_alignment_subjectwise(
        self,
        distribution_results,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling distribution-alignment traces in each subject panel."""
        df = distribution_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No subject-level distribution alignment results to plot.")

        spaces = [space for space in self.DISTRIBUTION_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        fig_title = (
            title
            or f"Condition {condition_label}: subject-wise oral vs model {model_state} distribution alignment"
        )
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid]
            title_bits = []
            for space in spaces:
                one = sub[sub["alignment_space"] == space].sort_values("trial")
                if one.empty:
                    continue
                x = one["trial_pct"].to_numpy(dtype=float)
                y = self._rolling_mean(one["js_similarity"].to_numpy(dtype=float), window_size=window_size)
                ax.plot(
                    x,
                    y,
                    lw=0.9,
                    alpha=0.82,
                    color=self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555"),
                    label=self.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(space, space),
                )
                title_bits.append(
                    f"{self.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(space, space)}={np.nanmean(one['js_similarity']):.2f}"
                )
            ax.set_title(
                f"S{int(sid)}  " + ", ".join(title_bits),
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "JS similarity")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(spaces),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Distribution alignment subject-wise plot saved to %s", save_path)
        return fig

    def plot_distribution_based_alignment_subjectwise(self, *args, **kwargs):
        """Alias for the distribution-based subject-wise plot."""
        return self.plot_distribution_alignment_subjectwise(*args, **kwargs)

    def save_distribution_alignment_outputs(
        self,
        distribution_results,
        output_dir,
        prefix="distribution_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write distribution-alignment CSVs and the group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = distribution_results.copy()
        if df.empty:
            raise RuntimeError("No distribution alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = (
            df.groupby(["subject", "alignment_space", "alignment_label"], observed=True)["js_similarity"]
            .mean()
            .reset_index()
        )
        binned = self.summarize_distribution_alignment_by_bin(df, bins=bins)
        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_distribution_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: oral vs model {model_state} distribution alignment",
        )
        plt.close(fig)
        fig = self.plot_distribution_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise oral vs model {model_state} distribution alignment",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    def save_distribution_based_alignment_outputs(self, *args, **kwargs):
        """Alias for writing distribution-based alignment outputs."""
        kwargs.setdefault("prefix", "distribution_based_alignment")
        return self.save_distribution_alignment_outputs(*args, **kwargs)

    @staticmethod
    def summarize_oral_based_alignment_by_bin(oral_based_results, bins=20):
        """Return subject-balanced binned means and SEMs for oral-based similarity."""
        df = oral_based_results.copy()
        if df.empty:
            return pd.DataFrame()

        df = df[np.isfinite(df["oral_based_similarity"])]
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "trial_bin"], observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )

        rows = []
        for trial_bin, group in subject_bin.groupby("trial_bin", observed=True):
            values = group["oral_based_similarity"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "trial_bin": int(trial_bin),
                    "trial_pct": (int(trial_bin) - 0.5) / float(bins),
                    "oral_based_similarity_mean": float(np.mean(values)) if values.size else np.nan,
                    "oral_based_similarity_sem": OralAlignmentReportingMixin._sem(values),
                    "n_subjects": int(values.size),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def _oral_based_time_stats(cls, oral_based_results, bins=20):
        """Run one-way repeated-measures ANOVA for time bins."""
        if AnovaRM is None:
            return {"n": 0, "time_p": np.nan}

        df = oral_based_results.copy()
        df = df[np.isfinite(df["oral_based_similarity"])]
        if df.empty:
            return {"n": 0, "time_p": np.nan}

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "trial_bin"], observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )

        counts = subject_bin.groupby("subject").size()
        complete_subjects = counts[counts == int(bins)].index
        complete = subject_bin[subject_bin["subject"].isin(complete_subjects)].copy()
        if complete["subject"].nunique() < 3:
            return {"n": int(complete["subject"].nunique()), "time_p": np.nan}

        complete["trial_bin"] = complete["trial_bin"].astype(str)
        try:
            fit = AnovaRM(
                complete,
                depvar="oral_based_similarity",
                subject="subject",
                within=["trial_bin"],
            ).fit()
        except Exception:
            return {"n": int(complete["subject"].nunique()), "time_p": np.nan}

        table = fit.anova_table
        p_val = float(table.loc["trial_bin", "Pr > F"]) if "trial_bin" in table.index else np.nan
        return {"n": int(complete["subject"].nunique()), "time_p": p_val}

    def plot_oral_based_alignment_group(
        self,
        oral_based_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level oral-based alignment summary and time course."""
        df = oral_based_results.copy()
        if df.empty:
            raise RuntimeError("No oral-based alignment results to plot.")

        primary_metric = str(df["primary_metric"].dropna().iloc[0])
        metric_label = self.ORAL_BASED_METRIC_LABELS.get(primary_metric, primary_metric)
        color = self.ORAL_BASED_METRIC_COLORS.get(primary_metric, "#4c78a8")
        subject_means = (
            df.groupby("subject", observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )
        binned = self.summarize_oral_based_alignment_by_bin(df, bins=bins)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: {oral_mode} oral-based alignment ({model_state})"
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        values = subject_means["oral_based_similarity"].to_numpy(dtype=float)
        mean = float(np.nanmean(values)) if values.size else np.nan
        sem = self._sem(values)
        ax.bar([0], [mean], yerr=[sem], color=color, alpha=0.82, capsize=5, edgecolor="white", linewidth=0.8)
        rng = np.random.default_rng(123)
        finite = np.isfinite(values)
        ax.scatter(
            rng.normal(0.0, 0.035, size=int(np.sum(finite))),
            values[finite],
            s=20,
            color="#222222",
            alpha=0.68,
            linewidths=0,
            zorder=3,
        )
        ax.set_xticks([0], [metric_label])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity")
        ax.set_title("Subject means")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.text(
            0.02,
            0.98,
            f"mean={mean:.3f}\nSEM={sem:.3f}\nn={int(np.sum(finite))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        ax = axes[1]
        if not binned.empty:
            x = binned["trial_pct"].to_numpy(dtype=float)
            self._line_with_sem(
                ax,
                x,
                binned["oral_based_similarity_mean"].to_numpy(dtype=float),
                binned["oral_based_similarity_sem"].to_numpy(dtype=float),
                metric_label,
                color,
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("Similarity")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        time_stats = self._oral_based_time_stats(df, bins=bins)
        ax.text(
            0.02,
            0.02,
            f"RM-ANOVA n={time_stats['n']}\ntime p{self._format_p_value(time_stats['time_p'])}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Oral-based alignment group plot saved to %s", save_path)
        return fig

    def plot_oral_based_alignment_subjectwise(
        self,
        oral_based_results,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling oral-based alignment in each subject panel."""
        df = oral_based_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No oral-based subject-level alignment results to plot.")

        primary_metric = str(df["primary_metric"].dropna().iloc[0])
        metric_label = self.ORAL_BASED_METRIC_LABELS.get(primary_metric, primary_metric)
        color = self.ORAL_BASED_METRIC_COLORS.get(primary_metric, "#4c78a8")
        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: subject-wise {oral_mode} oral-based alignment"
        fig.suptitle(
            f"{fig_title} ({model_state})",
            fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE,
            y=0.995,
        )

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            y = self._rolling_mean(sub["oral_based_similarity"].to_numpy(dtype=float), window_size=window_size)
            ax.plot(x, y, lw=0.95, alpha=0.84, color=color, label=metric_label)
            ax.set_title(
                f"S{int(sid)}  mean={np.nanmean(y):.2f}",
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Similarity")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Oral-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_oral_based_alignment_outputs(
        self,
        oral_based_results,
        output_dir,
        prefix="oral_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write oral-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = oral_based_results.copy()
        if df.empty:
            raise RuntimeError("No oral-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = (
            df.groupby("subject", observed=True)[
                [
                    "oral_based_similarity",
                    "expected_center_similarity",
                    "fuzzy_iou_similarity",
                    "fuzzy_cosine_similarity",
                    "model_mass_inside_oral",
                    "oral_region_covered_by_model",
                    "model_expected_volume",
                    "oral_volume",
                ]
            ]
            .mean()
            .reset_index()
        )
        binned = self.summarize_oral_based_alignment_by_bin(df, bins=bins)
        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_oral_based_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: {oral_mode} oral-based alignment ({model_state})",
        )
        plt.close(fig)
        fig = self.plot_oral_based_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise {oral_mode} oral-based alignment",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    def summarize_target_based_alignment(self, target_based_results):
        """Compute subject-level metrics between model/oral target trajectories."""
        df = target_based_results.copy()
        if df.empty:
            return pd.DataFrame()
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if "alignment_label" not in df.columns:
            df["alignment_label"] = df["alignment_space"].map(self.TARGET_ALIGNMENT_LABELS).fillna(df["alignment_space"])

        rows = []
        for (sid, space), sub in df.groupby(["subject", "alignment_space"], observed=True):
            model_vals = sub["model_target_prior"].to_numpy(dtype=float)
            oral_vals = sub["oral_target_mass"].to_numpy(dtype=float)
            valid = np.isfinite(model_vals) & np.isfinite(oral_vals)
            rows.append(
                {
                    "subject": int(sid),
                    "iSub": int(sid),
                    "condition": int(sub["condition"].dropna().iloc[0]),
                    "oral_mode": str(sub["oral_mode"].dropna().iloc[0]),
                    "alignment_space": str(space),
                    "alignment_label": str(sub["alignment_label"].dropna().iloc[0]),
                    "target_hypo": int(sub["target_hypo"].dropna().iloc[0]),
                    "n_trials": int(len(sub)),
                    "n_valid": int(np.sum(valid)),
                    "valid_rate": float(np.mean(valid)) if len(valid) else np.nan,
                    "active_set_size_mean": (
                        float(np.nanmean(sub["active_set_size"].to_numpy(dtype=float)))
                        if "active_set_size" in sub
                        else np.nan
                    ),
                    "comparison_set_size_mean": (
                        float(np.nanmean(sub["comparison_set_size"].to_numpy(dtype=float)))
                        if "comparison_set_size" in sub
                        else np.nan
                    ),
                    "oral_mass_in_comparison_set_mean": (
                        float(np.nanmean(sub["oral_mass_in_comparison_set"].to_numpy(dtype=float)))
                        if "oral_mass_in_comparison_set" in sub
                        else np.nan
                    ),
                    "model_target_prior_mean": (
                        float(np.nanmean(model_vals)) if np.any(np.isfinite(model_vals)) else np.nan
                    ),
                    "oral_target_mass_mean": (
                        float(np.nanmean(oral_vals)) if np.any(np.isfinite(oral_vals)) else np.nan
                    ),
                    "pearson_r": self._safe_pearson(model_vals, oral_vals),
                    "spearman_rho": self._safe_spearman(model_vals, oral_vals),
                    "cosine_similarity": self._safe_cosine_similarity(model_vals, oral_vals),
                    **{
                        field: (
                            sub[field].dropna().iloc[0]
                            if field in sub and not sub[field].dropna().empty
                            else np.nan
                        )
                        for field in self.ORAL_ENCODER_METADATA_FIELDS
                    },
                }
            )
        return pd.DataFrame(rows)

    def plot_target_based_alignment_group(
        self,
        target_subject_metrics,
        save_path=None,
        title=None,
    ):
        """Plot group-level target trajectory metrics for each comparison space."""
        df = target_subject_metrics.copy()
        if df.empty:
            raise RuntimeError("No target-based subject metrics to plot.")
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if "alignment_label" not in df.columns:
            df["alignment_label"] = df["alignment_space"].map(self.TARGET_ALIGNMENT_LABELS).fillna(df["alignment_space"])

        metrics = list(self.TARGET_BASED_METRICS)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: target-based alignment ({oral_mode})"
        encoder_label = self._oral_encoder_label(df)
        spaces = [space for space in self.TARGET_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            spaces = sorted(df["alignment_space"].dropna().unique())
        fig, axes = plt.subplots(1, len(spaces), figsize=(5.1 * len(spaces), 5.2), dpi=170, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        fig.suptitle(fig_title, fontsize=15, y=0.98)
        if encoder_label:
            fig.text(0.995, 0.008, f"Oral encoder: {encoder_label}", ha="right", va="bottom", fontsize=8)

        rng = np.random.default_rng(123)
        for ax, space in zip(axes, spaces):
            sub = df[df["alignment_space"] == space]
            x = np.arange(len(metrics), dtype=float)
            means = []
            sems = []
            for metric in metrics:
                vals = sub[metric].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
                sems.append(self._sem(vals))
            colors = [self.TARGET_BASED_METRIC_COLORS.get(metric, "#555555") for metric in metrics]
            ax.bar(x, means, yerr=sems, color=colors, alpha=0.82, capsize=4, edgecolor="white", linewidth=0.8)

            for idx, metric in enumerate(metrics):
                vals = sub[metric].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                ax.scatter(
                    rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                    vals[finite],
                    s=18,
                    color="#222222",
                    alpha=0.62,
                    linewidths=0,
                    zorder=3,
                )

            ax.axhline(0, color="#333333", lw=0.8, alpha=0.5)
            ax.set_xticks(x, [self.TARGET_BASED_METRIC_LABELS.get(metric, metric) for metric in metrics])
            ax.tick_params(axis="x", labelrotation=12)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(self.TARGET_ALIGNMENT_LABELS.get(space, space))
            ax.grid(axis="y", alpha=0.18, linewidth=0.7)
            ax.text(
                0.02,
                0.02,
                f"n={sub['subject'].nunique()}\nvalid={np.nanmean(sub['valid_rate']):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
            )
        axes[0].set_ylabel("Subject-level metric")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Target-based alignment group plot saved to %s", save_path)
        return fig

    @plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    def plot_target_based_alignment_subjectwise(
        self,
        target_based_results,
        target_subject_metrics=None,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
        alignment_space="full",
    ):
        """Plot expected model target mass, latent-state bands, and oral mass."""
        df = target_based_results.copy()
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if alignment_space is not None:
            df = df[df["alignment_space"] == alignment_space]
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No target-based trial metrics to plot.")

        # Subject-level association metrics remain in the group plot and CSV.
        # The trajectory panels mirror the canonical PF accuracy-band layout.
        del target_subject_metrics

        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(
            subjects_sorted,
            n_cols,
            panel_width=4.2,
            panel_height=3.1,
        )
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=False,
            sharey=False,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        space = str(df["alignment_space"].dropna().iloc[0])
        space_label = self.TARGET_ALIGNMENT_LABELS.get(space, space)
        backend = (
            str(df["model_inference_backend"].dropna().iloc[0])
            if "model_inference_backend" in df
            and not df["model_inference_backend"].dropna().empty
            else "particle_filter"
        )
        is_trajectory = backend == "trajectory"
        n_draws = (
            int(df["model_target_band_n_draws"].dropna().iloc[0])
            if "model_target_band_n_draws" in df.columns
            and not df["model_target_band_n_draws"].dropna().empty
            else self.DEFAULT_TARGET_BAND_DRAWS
        )
        n_runs = (
            int(df["model_target_band_n_runs"].dropna().iloc[0])
            if "model_target_band_n_runs" in df
            and not df["model_target_band_n_runs"].dropna().empty
            else None
        )
        legend_drawn = False
        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial"].to_numpy(dtype=float)
            has_band = all(
                column in sub
                for column in (
                    "model_target_expected_rolling",
                    "model_target_q05_rolling",
                    "model_target_q25_rolling",
                    "model_target_q75_rolling",
                    "model_target_q95_rolling",
                )
            )
            if has_band:
                ax.fill_between(
                    x,
                    sub["model_target_q05_rolling"].to_numpy(dtype=float),
                    sub["model_target_q95_rolling"].to_numpy(dtype=float),
                    color=self.TARGET_BASED_BAND_COLORS["q05_q95"],
                    alpha=0.38,
                    linewidth=0,
                    label=(
                        "Trajectory 90% band"
                        if is_trajectory
                        else "90% latent-state PI"
                    ),
                )
                ax.fill_between(
                    x,
                    sub["model_target_q25_rolling"].to_numpy(dtype=float),
                    sub["model_target_q75_rolling"].to_numpy(dtype=float),
                    color=self.TARGET_BASED_BAND_COLORS["q25_q75"],
                    alpha=0.50,
                    linewidth=0,
                    label=(
                        "Trajectory 50% band"
                        if is_trajectory
                        else "50% latent-state PI"
                    ),
                )
                model_curve = sub["model_target_expected_rolling"].to_numpy(dtype=float)
                oral_curve = (
                    pd.Series(sub["oral_target_mass"].to_numpy(dtype=float))
                    .rolling(window=max(1, int(window_size)), min_periods=max(1, int(window_size)))
                    .mean()
                    .to_numpy()
                )
            else:
                model_curve = self._rolling_mean(
                    sub["model_target_prior"].to_numpy(dtype=float),
                    window_size=window_size,
                )
                oral_curve = self._rolling_mean(
                    sub["oral_target_mass"].to_numpy(dtype=float),
                    window_size=window_size,
                )
            ax.plot(
                x,
                model_curve,
                linewidth=2.0,
                color=self.TARGET_BASED_LINE_COLORS["model"],
                label=(
                    "Trajectory mean target mass"
                    if is_trajectory
                    else "PF expected target mass"
                ),
                zorder=4,
            )
            ax.plot(
                x,
                oral_curve,
                linewidth=2.1,
                color=self.TARGET_BASED_LINE_COLORS["oral"],
                label="Oral target probability",
                zorder=5,
            )
            run_count = (
                int(sub["model_target_n_runs"].dropna().iloc[0])
                if "model_target_n_runs" in sub.columns
                and not sub["model_target_n_runs"].dropna().empty
                else int(sub["model_target_n_pf_runs"].dropna().iloc[0])
                if "model_target_n_pf_runs" in sub.columns
                and not sub["model_target_n_pf_runs"].dropna().empty
                else 1
            )
            ax.set(
                title=(
                    f"Subject {int(sid)} | Trajectory runs={run_count}"
                    if is_trajectory
                    else f"Subject {int(sid)} | PF runs={run_count}"
                ),
                xlabel="Trial",
                ylabel="Rolling target probability",
                ylim=(0.0, 1.0),
            )
            ax.grid(axis="y", alpha=0.20)
            if not legend_drawn:
                ax.legend(loc="best", fontsize=8, frameon=False)
                legend_drawn = True

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        fig_title = title or (
            (
                "Latent-trajectory target-mass ensemble"
                if is_trajectory
                else "Particle-filter conditional latent target occupancy"
            )
            + f" ({oral_mode}; {space_label})"
        )
        interval_note = (
            f"50% and 90% pointwise bands | {int(n_runs or 1):,} trajectory runs"
            if is_trajectory
            else f"50% and 90% pointwise intervals | {n_draws:,} draws"
        )
        fig.suptitle(
            f"{fig_title}\n{interval_note}",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            logger.info("Target-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_target_based_alignment_outputs(
        self,
        target_based_results,
        output_dir,
        prefix="target_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        title_prefix=None,
        target_band_draws=OralAlignmentScoringMixin.DEFAULT_TARGET_BAND_DRAWS,
        target_band_seed=OralAlignmentScoringMixin.DEFAULT_TARGET_BAND_SEED,
    ):
        """Write target-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self._attach_target_sampling_bands(
            target_based_results,
            window_size=window_size,
            n_draws=target_band_draws,
            base_seed=target_band_seed,
        )
        if df.empty:
            raise RuntimeError("No target-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_metrics.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        group_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_metrics = self.summarize_target_based_alignment(df)
        df.to_csv(trial_csv, index=False)
        subject_metrics.to_csv(subject_csv, index=False)
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        spaces = [space for space in self.TARGET_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            spaces = sorted(df["alignment_space"].dropna().unique())

        subjectwise_plots = {}
        for space in spaces:
            suffix = self.TARGET_ALIGNMENT_SUFFIXES.get(space, str(space))
            subjectwise_plot = output_dir / f"{prefix}_{suffix}_subject.png"
            if len(spaces) == 1 and subjectwise_plot_path:
                subjectwise_plot = Path(subjectwise_plot_path)
            subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)
            subjectwise_plots[space] = subjectwise_plot

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_target_based_alignment_group(
            subject_metrics,
            save_path=str(group_plot),
            title=f"{prefix_title}: target-based alignment ({oral_mode})",
        )
        plt.close(fig)
        for space, subjectwise_plot in subjectwise_plots.items():
            fig = self.plot_target_based_alignment_subjectwise(
                df,
                target_subject_metrics=subject_metrics,
                save_path=str(subjectwise_plot),
                window_size=window_size,
                title=title_prefix,
                alignment_space=space,
            )
            plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_metrics": subject_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plots.get("full") or next(iter(subjectwise_plots.values()), None),
            "subjectwise_plots": subjectwise_plots,
        }

    def summarize_hit_based_alignment(self, hit_based_results):
        """Compute subject-level association metrics between binary hit traces."""
        df = hit_based_results.copy()
        if df.empty:
            return pd.DataFrame()

        def finite_mean(values):
            arr = np.asarray(values, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan
            return float(np.mean(arr))

        rows = []
        for sid, sub in df.groupby("subject", observed=True):
            model_hits = sub["model_target_hit"].to_numpy(dtype=float)
            oral_hits = sub["oral_target_hit"].to_numpy(dtype=float)
            valid = np.isfinite(model_hits) & np.isfinite(oral_hits)
            if np.any(valid):
                mh = model_hits[valid]
                oh = oral_hits[valid]
                agreement = float(np.mean(mh == oh))
                joint_hit = float(np.mean((mh > 0.5) & (oh > 0.5)))
                model_hit_rate = float(np.mean(mh > 0.5))
                oral_hit_rate = float(np.mean(oh > 0.5))
            else:
                agreement = np.nan
                joint_hit = np.nan
                model_hit_rate = np.nan
                oral_hit_rate = np.nan

            rows.append(
                {
                    "subject": int(sid),
                    "iSub": int(sid),
                    "condition": int(sub["condition"].dropna().iloc[0]),
                    "oral_mode": str(sub["oral_mode"].dropna().iloc[0]),
                    "hit_rule": str(sub["hit_rule"].dropna().iloc[0]) if "hit_rule" in sub else "active_set_topn",
                    "hit_rule_label": (
                        str(sub["hit_rule_label"].dropna().iloc[0])
                        if "hit_rule_label" in sub
                        else "active_set_topN"
                    ),
                    "rank_top_k": (
                        float(sub["rank_top_k"].dropna().iloc[0])
                        if "rank_top_k" in sub and not sub["rank_top_k"].dropna().empty
                        else np.nan
                    ),
                    "target_hypo": int(sub["target_hypo"].dropna().iloc[0]),
                    "n_trials": int(len(sub)),
                    "n_valid": int(np.sum(valid)),
                    "valid_rate": float(np.mean(valid)) if len(valid) else np.nan,
                    "model_hit_rate": model_hit_rate,
                    "oral_hit_rate": oral_hit_rate,
                    "joint_hit_rate": joint_hit,
                    "active_set_size_mean": finite_mean(sub["active_set_size"]),
                    "oral_topn_mass_mean": finite_mean(sub["oral_topn_mass"]),
                    "active_oral_mass_mean": finite_mean(sub["active_oral_mass"]),
                    "model_target_rank_mean": finite_mean(sub["model_target_rank"]),
                    "oral_target_rank_mean": finite_mean(sub["oral_target_rank"]),
                    "phi_correlation": self._safe_pearson(model_hits, oral_hits),
                    "cohen_kappa": self._safe_cohen_kappa(model_hits, oral_hits),
                    "hit_agreement_rate": agreement,
                    "positive_hit_jaccard": self._safe_binary_jaccard(model_hits, oral_hits),
                    **{
                        field: (
                            sub[field].dropna().iloc[0]
                            if field in sub and not sub[field].dropna().empty
                            else np.nan
                        )
                        for field in self.ORAL_ENCODER_METADATA_FIELDS
                    },
                }
            )
        return pd.DataFrame(rows)

    def plot_hit_based_alignment_group(
        self,
        hit_subject_metrics,
        save_path=None,
        title=None,
    ):
        """Plot group-level metrics for binary hit-based alignment."""
        df = hit_subject_metrics.copy()
        if df.empty:
            raise RuntimeError("No hit-based subject metrics to plot.")

        metrics = list(self.HIT_BASED_METRICS)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        hit_rule_label = (
            str(df["hit_rule_label"].dropna().iloc[0])
            if "hit_rule_label" in df and not df["hit_rule_label"].dropna().empty
            else "active_set_topN"
        )
        fig_title = title or f"Condition {condition_label}: hit-based alignment ({oral_mode})"
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.3), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.98)

        x = np.arange(len(metrics), dtype=float)
        means = []
        sems = []
        for metric in metrics:
            vals = df[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
            sems.append(self._sem(vals))
        colors = [self.HIT_BASED_METRIC_COLORS.get(metric, "#555555") for metric in metrics]
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.84, capsize=4, edgecolor="white", linewidth=0.8)

        rng = np.random.default_rng(123)
        for idx, metric in enumerate(metrics):
            vals = df[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            ax.scatter(
                rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                vals[finite],
                s=20,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0, color="#333333", lw=0.8, alpha=0.5)
        ax.set_xticks(x, [self.HIT_BASED_METRIC_LABELS.get(metric, metric) for metric in metrics])
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Subject-level metric")
        if hit_rule_label.startswith("top"):
            ax.set_title(f"Model {hit_rule_label} target hit vs oral {hit_rule_label} target hit")
        else:
            ax.set_title("Model active-set target hit vs oral top-N target hit")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.text(
            0.02,
            0.02,
            (
                f"n={df['subject'].nunique()}\n"
                f"valid rate={np.nanmean(df['valid_rate']):.2f}\n"
                f"model hit={np.nanmean(df['model_hit_rate']):.2f}, oral hit={np.nanmean(df['oral_hit_rate']):.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Hit-based alignment group plot saved to %s", save_path)
        return fig

    def plot_hit_based_alignment_subjectwise(
        self,
        hit_based_results,
        hit_subject_metrics=None,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling binary target-hit rates in each subject panel."""
        df = hit_based_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No hit-based trial metrics to plot.")

        if hit_subject_metrics is None:
            hit_subject_metrics = self.summarize_hit_based_alignment(df)
        metric_lookup = {
            int(row["subject"]): row
            for _, row in hit_subject_metrics.iterrows()
        }

        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        hit_rule_label = (
            str(df["hit_rule_label"].dropna().iloc[0])
            if "hit_rule_label" in df and not df["hit_rule_label"].dropna().empty
            else "active_set_topN"
        )
        if hit_rule_label.startswith("top"):
            model_line_label = f"Model {hit_rule_label} target hit"
            oral_line_label = f"Oral {hit_rule_label} target hit"
        else:
            model_line_label = "Model active-set target hit"
            oral_line_label = "Oral top-N target hit"
        fig_title = title or f"Condition {condition_label}: hit-based alignment ({oral_mode})"
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            ax.plot(
                x,
                self._rolling_mean(sub["model_target_hit"].to_numpy(dtype=float), window_size=window_size),
                lw=1.05,
                alpha=0.88,
                color=self.HIT_BASED_LINE_COLORS["model"],
                label=model_line_label,
            )
            ax.plot(
                x,
                self._rolling_mean(sub["oral_target_hit"].to_numpy(dtype=float), window_size=window_size),
                lw=1.05,
                alpha=0.88,
                color=self.HIT_BASED_LINE_COLORS["oral"],
                label=oral_line_label,
            )
            metrics = metric_lookup.get(int(sid))
            if metrics is not None:
                ax.set_title(
                    f"S{int(sid)}  phi={metrics.get('phi_correlation', np.nan):.2f}, "
                    f"agr={metrics.get('hit_agreement_rate', np.nan):.2f}",
                    fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
                )
            else:
                ax.set_title(f"S{int(sid)}", fontsize=self.SUBJECTWISE_TITLE_FONTSIZE)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Rolling hit rate")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Hit-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_hit_based_alignment_outputs(
        self,
        hit_based_results,
        output_dir,
        prefix="hit_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        title_prefix=None,
    ):
        """Write hit-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = hit_based_results.copy()
        if df.empty:
            raise RuntimeError("No hit-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_metrics.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_metrics = self.summarize_hit_based_alignment(df)
        df.to_csv(trial_csv, index=False)
        subject_metrics.to_csv(subject_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_hit_based_alignment_group(
            subject_metrics,
            save_path=str(group_plot),
            title=f"{prefix_title}: hit-based alignment ({oral_mode})",
        )
        plt.close(fig)
        fig = self.plot_hit_based_alignment_subjectwise(
            df,
            hit_subject_metrics=subject_metrics,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: hit-based alignment ({oral_mode})",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_metrics": subject_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    def summarize_coverage_based_alignment(self, coverage_results):
        """Return subject means for the two coverage-based alignment metrics."""
        df = coverage_results.copy()
        if df.empty:
            return pd.DataFrame()

        metrics = list(self.COVERAGE_BASED_METRICS) + [
            "active_oral_mass",
            "oracle_topn_oral_mass",
            "random_expected_mass",
            "n_active",
            "active_fraction",
        ]
        present_metrics = [metric for metric in metrics if metric in df.columns]
        subject_means = (
            df.groupby("subject", observed=True)[present_metrics]
            .mean()
            .reset_index()
        )
        meta = (
            df.groupby("subject", observed=True)[["iSub", "condition", "oral_mode"]]
            .first()
            .reset_index()
        )
        return meta.merge(subject_means, on="subject", how="left")

    @staticmethod
    def summarize_coverage_based_alignment_by_bin(coverage_results, bins=20):
        """Return subject-balanced binned means and SEMs for coverage alignment."""
        df = coverage_results.copy()
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        metrics = [
            "active_capture_ratio",
            "active_topn_overlap",
            "active_oral_mass",
            "oracle_topn_oral_mass",
            "random_expected_mass",
            "n_active",
            "active_fraction",
        ]
        subject_bin = df.groupby(["subject", "trial_bin"], observed=True)[metrics].mean().reset_index()

        rows = []
        for trial_bin, group in subject_bin.groupby("trial_bin", observed=True):
            item = {"trial_bin": int(trial_bin), "trial_pct": (int(trial_bin) - 0.5) / int(bins)}
            for metric in metrics:
                vals = group[metric].to_numpy(dtype=float)
                valid = vals[~np.isnan(vals)]
                item[f"{metric}_mean"] = float(np.mean(valid)) if valid.size else np.nan
                item[f"{metric}_sem"] = (
                    float(np.std(valid, ddof=1) / np.sqrt(valid.size))
                    if valid.size > 1
                    else np.nan
                )
            rows.append(item)
        return pd.DataFrame(rows)

    @staticmethod
    def _line_with_sem(ax, x, mean, sem, label, color):
        ax.plot(x, mean, lw=2.2, label=label, color=color)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)

    def plot_coverage_based_alignment_group(
        self,
        coverage_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level coverage alignment: subject bars and time course."""
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to plot.")

        binned = self.summarize_coverage_based_alignment_by_bin(df, bins=bins)
        subject_means = self.summarize_coverage_based_alignment(df)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: coverage-based alignment ({oral_mode})"
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        metric_names = list(self.COVERAGE_BASED_METRICS)
        x_bar = np.arange(len(metric_names), dtype=float)
        means = []
        sems = []
        for metric in metric_names:
            vals = subject_means[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
            sems.append(self._sem(vals))
        colors = [self.COVERAGE_BASED_COLORS[metric] for metric in metric_names]
        ax.bar(x_bar, means, yerr=sems, color=colors, alpha=0.84, capsize=4, edgecolor="white", linewidth=0.8)
        rng = np.random.default_rng(123)
        for idx, metric in enumerate(metric_names):
            vals = subject_means[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            ax.scatter(
                rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                vals[finite],
                s=20,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )
        ax.set_xticks(x_bar, [self.COVERAGE_BASED_LABELS[metric] for metric in metric_names])
        ax.tick_params(axis="x", labelrotation=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Subject mean")
        ax.set_title("Group mean")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)

        ax = axes[1]
        x = binned["trial_pct"].to_numpy(dtype=float)
        for metric in metric_names:
            self._line_with_sem(
                ax,
                x,
                binned[f"{metric}_mean"].to_numpy(dtype=float),
                binned[f"{metric}_sem"].to_numpy(dtype=float),
                self.COVERAGE_BASED_LABELS[metric],
                self.COVERAGE_BASED_COLORS[metric],
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("Coverage")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info("Coverage-based alignment group plot saved to %s", save_path)
        return fig

    def plot_coverage_based_alignment_subjectwise(
        self,
        coverage_results,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling coverage-based alignment traces in each subject panel."""
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to plot.")

        subjects = sorted(df["subject"].unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: subject-wise coverage-based alignment ({oral_mode})"
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            for metric in self.COVERAGE_BASED_METRICS:
                ax.plot(
                    x,
                    self._rolling_mean(sub[metric], window_size),
                    lw=1.05,
                    alpha=0.88,
                    color=self.COVERAGE_BASED_COLORS[metric],
                    label=self.COVERAGE_BASED_LABELS[metric],
                )
            ax.set_title(
                (
                    f"S{int(sid)}  "
                    f"cap={np.nanmean(sub['active_capture_ratio']):.2f}, "
                    f"ov={np.nanmean(sub['active_topn_overlap']):.2f}"
                ),
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Coverage")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Coverage-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_coverage_based_alignment_outputs(
        self,
        coverage_results,
        output_dir,
        prefix="coverage_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write coverage-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = self.summarize_coverage_based_alignment(df)
        binned = self.summarize_coverage_based_alignment_by_bin(df, bins=bins)

        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_coverage_based_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: coverage-based alignment ({oral_mode})",
        )
        plt.close(fig)
        fig = self.plot_coverage_based_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise coverage-based alignment ({oral_mode})",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }


__all__ = ["OralAlignmentReportingMixin"]
