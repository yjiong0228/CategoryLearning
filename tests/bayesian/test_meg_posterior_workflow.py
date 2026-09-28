from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.Bayesian import run_meg_posterior as workflow


def _subject_frame(subject_id: int = 334) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iSub": [subject_id, subject_id, subject_id],
            "condition": [3, 3, 3],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.2, 0.3, 0.4],
            "feature3": [0.3, 0.4, 0.5],
            "feature4": [0.4, 0.5, 0.6],
            "category": [1, 2, 3],
            "choice": [1, 2, 1],
            "rating": [1, 2, 3],
            "feedback": [1.0, 1.0, 0.0],
        }
    )


def test_prepare_run_paths_uses_yymmdd_and_refuses_overwrite(tmp_path: Path) -> None:
    paths = workflow.build_run_paths(tmp_path, 334, date(2026, 9, 28))

    assert paths.output_dir == (
        tmp_path
        / "results/model_static/model_results_meg"
        / "Model_results_sub334_260928"
    )
    assert paths.posterior_csv.name == "Task3b_Sub334_M6_MH_model_posterior.csv"

    workflow.prepare_output_directory(paths)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        workflow.prepare_output_directory(paths)


def test_effective_fit_jobs_is_bounded_by_cpu_and_ready_tasks() -> None:
    assert workflow.effective_fit_jobs(120, available_cpus=128, ready_tasks=400) == 120
    assert workflow.effective_fit_jobs(120, available_cpus=32, ready_tasks=400) == 32
    assert workflow.effective_fit_jobs(120, available_cpus=128, ready_tasks=7) == 7


def test_source_file_hashes_cover_legacy_and_shared_hypothesis_code(
    tmp_path: Path,
) -> None:
    legacy_file = tmp_path / "src/Bayesian/model.py"
    shared_file = tmp_path / "src/Bayesian_state/hypothesis_space/catalog.py"
    package_file = tmp_path / "src/Bayesian_state/__init__.py"
    requirements_file = tmp_path / "requirements.txt"
    for path in (legacy_file, shared_file, package_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text("legacy\n", encoding="utf-8")
    shared_file.write_text("shared\n", encoding="utf-8")
    package_file.write_text("package\n", encoding="utf-8")
    requirements_file.write_text("numpy==2.2.6\n", encoding="utf-8")

    hashes = workflow._source_file_hashes(tmp_path)

    assert hashes == {
        "requirements.txt": hashlib.sha256(b"numpy==2.2.6\n").hexdigest(),
        "src/Bayesian/model.py": hashlib.sha256(b"legacy\n").hexdigest(),
        "src/Bayesian_state/__init__.py": hashlib.sha256(b"package\n").hexdigest(),
        "src/Bayesian_state/hypothesis_space/catalog.py": hashlib.sha256(
            b"shared\n"
        ).hexdigest(),
    }


def test_build_posterior_table_preserves_trial_semantics_and_normalizes() -> None:
    subject_data = _subject_frame()
    prediction = {
        "pred_choice": np.array([1, 2, 1]),
        "pred_probs": np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.6, 0.1, 0.2, 0.1],
            ]
        ),
        "original_posterior": [
            {1: 1.0},
            {7: 2.0, 9: 2.0},
            {1: 0.2, 5: 0.3, 9: 0.5},
        ],
    }

    table = workflow.build_posterior_table(subject_data, prediction)

    assert list(table.columns) == workflow.POSTERIOR_COLUMNS
    assert table["trial"].tolist() == [2, 3]
    assert table["true_choice"].tolist() == [2, 3]
    assert table["predicted_choice"].tolist() == [2, 1]
    assert table["hit"].tolist() == [1, 0]
    assert table["nonhit"].tolist() == [0, 1]
    assert table["origin_true_choice_probability"].tolist() == pytest.approx(
        [0.7, 0.2]
    )
    assert json.loads(table.loc[0, "origin_posterior"]) == {"7": 0.5, "9": 0.5}
    assert table.loc[0, "origin_entropy"] == pytest.approx(np.log(2.0))
    assert table.loc[1, "origin_top1_hypothesis"] == 9
    assert table.loc[1, "origin_top1_probability"] == pytest.approx(0.5)
    assert table.loc[1, "origin_top3_mass"] == pytest.approx(1.0)


def test_build_posterior_table_allows_undefined_first_trial() -> None:
    subject_data = _subject_frame()
    prediction = {
        "pred_choice": np.array([0, 2, 1]),
        "pred_probs": np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.1, 0.7, 0.1, 0.1],
                [0.6, 0.1, 0.2, 0.1],
            ]
        ),
        "original_posterior": [
            None,
            {7: 0.5, 9: 0.5},
            {1: 0.2, 5: 0.3, 9: 0.5},
        ],
    }

    table = workflow.build_posterior_table(subject_data, prediction)

    assert table["trial"].tolist() == [2, 3]
    assert np.all(np.isfinite(table["origin_true_choice_probability"]))


def test_build_posterior_table_preserves_nonconsecutive_itrial_ids() -> None:
    subject_data = _subject_frame()
    subject_data["iTrial"] = [169, 172, 173]
    prediction = {
        "pred_choice": np.array([0, 2, 1]),
        "pred_probs": np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.1, 0.7, 0.1, 0.1],
                [0.6, 0.1, 0.2, 0.1],
            ]
        ),
        "original_posterior": [
            None,
            {7: 0.5, 9: 0.5},
            {1: 0.2, 5: 0.3, 9: 0.5},
        ],
    }

    table = workflow.build_posterior_table(subject_data, prediction)

    assert table["trial"].tolist() == [172, 173]


def test_missing_subject_is_rejected_before_output_is_created(tmp_path: Path) -> None:
    processed_csv = tmp_path / "Task3b_processed.csv"
    _subject_frame(subject_id=333).to_csv(processed_csv, index=False)

    with pytest.raises(ValueError, match="No behavioral rows found for subject 334"):
        workflow.run_meg_posterior(
            334,
            project_root=tmp_path,
            processed_csv=processed_csv,
            run_date=date(2026, 9, 28),
        )

    assert not (
        tmp_path
        / "results/model_static/model_results_meg"
        / "Model_results_sub334_260928"
    ).exists()


def test_export_rejects_fit_and_behavior_length_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_csv = tmp_path / "Task3b_processed.csv"
    _subject_frame().to_csv(processed_csv, index=False)
    joblib.dump(
        {334: {"condition": 3, "best_step_results": [{}, {}]}},
        tmp_path / "M6_MH.joblib",
    )

    class UnexpectedModel:
        def __init__(self, *args: object, **kwargs: object):
            raise AssertionError("model should not run for a misaligned fit")

    monkeypatch.setattr(workflow, "StandardModel", UnexpectedModel)

    with pytest.raises(ValueError, match="Fit/data trial mismatch"):
        workflow.export_m6_mh_model_posterior_csv(334, tmp_path, processed_csv)


def test_export_rejects_fit_and_behavior_condition_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_csv = tmp_path / "Task3b_processed.csv"
    _subject_frame().to_csv(processed_csv, index=False)
    joblib.dump(
        {334: {"condition": 2, "best_step_results": [{}, {}, {}]}},
        tmp_path / "M6_MH.joblib",
    )

    class UnexpectedModel:
        def __init__(self, *args: object, **kwargs: object):
            raise AssertionError("model should not run for a misaligned fit")

    monkeypatch.setattr(workflow, "StandardModel", UnexpectedModel)

    with pytest.raises(ValueError, match="Fit/data condition mismatch"):
        workflow.export_m6_mh_model_posterior_csv(334, tmp_path, processed_csv)


def test_export_rejects_prediction_crosscheck_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_csv = tmp_path / "Task3b_processed.csv"
    _subject_frame().to_csv(processed_csv, index=False)
    joblib.dump(
        {334: {"condition": 3, "best_step_results": [{}, {}, {}]}},
        tmp_path / "M6_MH.joblib",
    )
    joblib.dump(
        {334: {"pred_acc": np.array([np.nan, 0.6, 0.2])}},
        tmp_path / "M6_MH_prediction.joblib",
    )

    class LightweightModel:
        def __init__(self, *args: object, **kwargs: object):
            self.condition = kwargs["condition"]

        def predict_probs(
            self,
            data: tuple[np.ndarray, ...],
            step_results: list[dict],
            **kwargs: object,
        ) -> dict:
            return {
                "true_choice": np.array([0, 2, 3]),
                "pred_choice": np.array([0, 2, 1]),
                "top1_choice": np.array([0, 2, 1]),
                "pred_probs": np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [0.1, 0.7, 0.1, 0.1],
                        [0.6, 0.1, 0.2, 0.1],
                    ]
                ),
                "true_acc": np.array([np.nan, 1.0, 0.0]),
                "pred_acc": np.array([np.nan, 0.7, 0.2]),
                "sliding_true_acc": [],
                "sliding_pred_acc": [],
                "sliding_pred_acc_std": [],
                "original_posterior": [
                    None,
                    {7: 0.5, 9: 0.5},
                    {1: 0.2, 5: 0.3, 9: 0.5},
                ],
                "choice_constrained_posterior": [None, None, None],
                "original_true_choice_prob": np.array([np.nan, 0.7, 0.2]),
                "cc_true_choice_prob": np.array([np.nan, np.nan, np.nan]),
                "posterior_shift_tv": np.array([np.nan, np.nan, np.nan]),
            }

    monkeypatch.setattr(workflow, "StandardModel", LightweightModel)

    with pytest.raises(ValueError, match="Posterior/prediction crosscheck failed"):
        workflow.export_m6_mh_model_posterior_csv(334, tmp_path, processed_csv)

    assert not (tmp_path / "Task3b_Sub334_M6_MH_model_posterior.csv").exists()


def test_run_meg_posterior_creates_complete_artifact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_csv = tmp_path / "data/meg/processed/Task3b_processed.csv"
    processed_csv.parent.mkdir(parents=True)
    _subject_frame().to_csv(processed_csv, index=False)

    class LightweightOptimizer:
        def __init__(self, module_config: dict, n_jobs: int):
            self.module_config = module_config
            self.n_jobs = n_jobs
            self.optimize_params_dict = {
                334: {"gamma": [0.1, 0.2, 0.3], "w0": [0.1, 0.2, 0.3]}
            }

        def prepare_data(self, data_path: Path) -> None:
            self.learning_data = pd.read_csv(data_path)

        def optimize_params_with_subs_parallel(
            self,
            model_config: dict,
            subjects: list[int],
            window_size: dict[int, int],
            grid_repeat: int,
            mc_samples: int,
        ) -> dict:
            if self.n_jobs != 2:
                raise AssertionError("fit workers were not capped by available CPUs")
            if subjects != [334] or window_size[334] != 16:
                raise AssertionError("runner changed subject or window size")
            if grid_repeat != 64 or mc_samples != 1024:
                raise AssertionError("runner changed the frozen fitting budget")
            return {
                334: {
                    "condition": 3,
                    "best_params": {"gamma": 0.05, "w0": 0.075},
                    "best_error": 0.1,
                    "best_step_results": [{"hypo_details": {}}] * 3,
                    "raw_step_results": ["raw"],
                    "grid_errors": {(0.05, 0.075): [0.1]},
                    "sample_errors": [0.1],
                }
            }

        def save_results(self, results: dict, name: str, output_dir: Path) -> None:
            cache_file = Path(output_dir) / "cache" / name / "334.gz"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"raw-step-cache")
            joblib.dump(results, Path(output_dir) / f"{name}.joblib")

        def set_results(self, results: dict) -> None:
            self.results = results

        def predict_with_subs_parallel(
            self, model_config: dict, subjects: list[int]
        ) -> dict:
            if self.n_jobs != 1:
                raise AssertionError("single-subject prediction must use one worker")
            return {
                334: {
                    "condition": 3,
                    "true_acc": np.ones(3),
                    "pred_acc": np.ones(3),
                    "sliding_true_acc": [],
                    "sliding_pred_acc": [],
                    "sliding_pred_acc_std": [],
                }
            }

    class LightweightEvaluator:
        @staticmethod
        def _write(save_path: Path) -> None:
            Path(save_path).write_bytes(b"png")

        def plot_posterior_probabilities(self, results: dict, save_path: Path) -> None:
            self._write(save_path)

        def plot_accuracy_comparison(self, results: dict, save_path: Path) -> None:
            self._write(save_path)

        def plot_error_grids(
            self, results: dict, fname: list[str], save_path: Path
        ) -> None:
            self._write(save_path)

        def plot_cluster_amount(
            self, results: dict, window_size: int, save_path: Path
        ) -> None:
            self._write(save_path)

    def lightweight_exporter(
        subject_id: int, result_dir: Path, processed_csv: Path
    ) -> tuple[Path, pd.DataFrame, dict]:
        output = (
            Path(result_dir)
            / f"Task3b_Sub{subject_id}_M6_MH_model_posterior.csv"
        )
        table = pd.DataFrame({"trial": [2]})
        table.to_csv(output, index=False)
        return output, table, {"n_rows": 1}

    monkeypatch.setattr(workflow, "Optimizer", LightweightOptimizer)
    monkeypatch.setattr(workflow, "ModelEval", LightweightEvaluator)
    monkeypatch.setattr(workflow.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        workflow, "export_m6_mh_model_posterior_csv", lightweight_exporter
    )

    completed = workflow.run_meg_posterior(
        334,
        project_root=tmp_path,
        processed_csv=processed_csv,
        run_date=date(2026, 9, 28),
    )

    expected_files = {
        "M6_MH.joblib",
        "M6_MH_prediction.joblib",
        "M6_MH_acc.png",
        "M6_MH_amount.png",
        "M6_MH_grid.png",
        "M6_MH_post.png",
        "Task3b_Sub334_M6_MH_model_posterior.csv",
        "run_manifest.json",
    }
    assert {path.name for path in completed.output_dir.iterdir() if path.is_file()} == (
        expected_files
    )
    assert (completed.output_dir / "cache/M6_MH/334.gz").is_file()
    manifest = json.loads((completed.output_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["subject_id"] == 334
    assert manifest["fit"]["window_size"] == 16
    assert manifest["fit"]["grid_repeat"] == 64
    assert manifest["fit"]["mc_samples"] == 1024
    assert manifest["fit"]["requested_n_jobs"] == 120
    assert manifest["fit"]["effective_fit_jobs"] == 2
    assert manifest["fit"]["prediction_jobs"] == 1
    assert manifest["fit"]["inner_max_num_threads"] == 1
    assert completed.posterior_csv == (
        completed.output_dir / "Task3b_Sub334_M6_MH_model_posterior.csv"
    )
