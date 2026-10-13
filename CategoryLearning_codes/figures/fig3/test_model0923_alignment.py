"""Accuracy and oral panels preserve their distinct evidence and failure states."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3 import model0923_alignment as renderer


def _curves() -> pd.DataFrame:
    return pd.DataFrame({
        "valid_trial": [5, 6, 7], "window_start_valid_trial": [2, 3, 4],
        "scored_window": [True, True, True],
        "observed_rolling_exact_accuracy": [.25, .5, .75],
        "conditional_rolling_correct_probability": [.4, .5, .6],
        "rolling_accuracy_residual": [-.15, 0., .15],
        "iSession": [1, 1, 2],
    })


def _oral() -> pd.DataFrame:
    return pd.DataFrame({
        "trial": [1, 2, 3, 4, 5], "parse_status": ["coded", "coded", "needs_review", "missing", "coded"],
        "scored": [False, True, True, True, True],
        "oral_score_mask": [False, True, False, False, True],
        "report_probability": [np.nan, .3, np.nan, np.nan, 0.],
        "report_top1_hit": [np.nan, 1., np.nan, np.nan, 0.],
        "report_top3_hit": [np.nan, 1., np.nan, np.nan, 0.],
        "report_zero_probability": [False, False, False, False, True],
        "rolling_report_probability": [np.nan, np.nan, np.nan, np.nan, .15],
        "rolling_top1_hit": [np.nan, np.nan, np.nan, np.nan, .5],
        "rolling_top3_hit": [np.nan, np.nan, np.nan, np.nan, .5],
    })


def _summary() -> dict:
    return {
        "status": "completed", "window_size": 4, "min_coded_reports": 2,
        "metrics": {"parser_coverage": .5, "vocabulary_coverage": .5,
                    "positive_support_coverage": .5, "mean_report_nll": float("inf")},
    }


def test_accuracy_panel_retains_trial_identity_and_signed_residual(tmp_path: Path, monkeypatch):
    seen = []
    original = renderer.save

    def capture(fig, path):
        seen.append(fig)
        original(fig, path)

    monkeypatch.setattr(renderer, "save", capture)
    curves = _curves()
    image, source = renderer.behavior_alignment_figures(307, 3, curves, tmp_path)
    assert image.name == "behavior_alignment.png" and image.read_bytes().startswith(b"\x89PNG")
    pd.testing.assert_frame_equal(pd.read_csv(source), curves)
    np.testing.assert_array_equal(seen[0].axes[0].lines[0].get_xdata(), [5, 6, 7])
    np.testing.assert_allclose(seen[0].axes[1].lines[1].get_ydata(), [-.15, 0, .15])
    with pytest.raises(FileExistsError, match="already exists"):
        renderer.behavior_alignment_figures(307, 3, curves, tmp_path)


@pytest.mark.parametrize("patch", [
    {"valid_trial": [5, 7, 6]},
    {"scored_window": ["True"] * 3},
    {"conditional_rolling_correct_probability": [.4, 1.2, .6]},
    {"rolling_accuracy_residual": [.15, 0, .15]},
])
def test_invalid_accuracy_source_cannot_render(tmp_path: Path, patch: dict):
    curves = _curves().assign(**patch)
    with pytest.raises(ValueError):
        renderer.behavior_alignment_figures(307, 3, curves, tmp_path)
    assert not list(tmp_path.glob("*.png"))


def test_condition3_family_and_reward_are_separate_panels(tmp_path: Path, monkeypatch):
    seen = []
    monkeypatch.setattr(renderer, "save", lambda fig, path: seen.append(fig))
    curves = _curves().assign(
        observed_rolling_family_accuracy=[.75, 1., 1.],
        conditional_rolling_family_probability=[.7, .8, .9],
        rolling_family_residual=[.05, .2, .1],
        observed_rolling_mean_reward=[.5, .75, .875],
        conditional_rolling_expected_reward=[.55, .65, .75],
        rolling_reward_residual=[-.05, .1, .125],
    )
    renderer.behavior_alignment_figures(307, 3, curves, tmp_path)
    assert len(seen[0].axes) == 6
    assert "种正确率" in seen[0].axes[0].get_ylabel()
    assert "科正确率" in seen[0].axes[2].get_ylabel()
    assert "平均得分" in seen[0].axes[4].get_ylabel()
    renderer.plt.close(seen[0])


def test_unscored_windows_keep_their_coordinates_and_leave_curve_gaps(tmp_path: Path, monkeypatch):
    seen = []
    monkeypatch.setattr(renderer, "save", lambda fig, path: seen.append(fig))
    curves = _curves().assign(scored_window=[True, False, True])
    _, source = renderer.behavior_alignment_figures(307, 3, curves, tmp_path)
    np.testing.assert_array_equal(seen[0].axes[0].lines[0].get_xdata(), [5, 6, 7])
    assert np.isnan(seen[0].axes[0].lines[0].get_ydata()[1])
    pd.testing.assert_frame_equal(pd.read_csv(source), curves)
    renderer.plt.close(seen[0])


def test_no_complete_window_has_an_explicit_panel_without_data(tmp_path: Path, monkeypatch):
    seen = []
    monkeypatch.setattr(renderer, "save", lambda fig, path: seen.append(fig))
    _, source = renderer.behavior_alignment_figures(307, 3, _curves().iloc[:0], tmp_path)
    assert pd.read_csv(source).empty
    assert "当前序列没有完整的计分窗口" in [label.get_text() for label in seen[0].axes[0].texts]
    renderer.plt.close(seen[0])


def test_oral_panel_keeps_zero_support_and_excludes_unscored_or_unparsed_reports(tmp_path: Path, monkeypatch):
    seen = []
    original = renderer.save

    def capture(fig, path):
        seen.append(fig)
        original(fig, path)

    monkeypatch.setattr(renderer, "save", capture)
    trials = _oral()
    image, source = renderer.oral_alignment_figures(307, 3, trials, _summary(), tmp_path)
    assert image.name == "oral_alignment.png" and image.read_bytes().startswith(b"\x89PNG")
    pd.testing.assert_frame_equal(pd.read_csv(source), trials)
    np.testing.assert_array_equal(seen[0].axes[0].collections[0].get_offsets(), [[2, .3], [5, 0.]])
    text = "\n".join(label.get_text() for label in seen[0].axes[3].texts)
    assert "完整编码 2" in text and "缺失 1" in text and "待复核 1" in text
    assert "∞（含零概率报告）" in text


def test_unavailable_oral_panel_has_reason_and_no_fabricated_trend(tmp_path: Path, monkeypatch):
    seen = []
    original = renderer.save

    def capture(fig, path):
        seen.append(fig)
        original(fig, path)

    monkeypatch.setattr(renderer, "save", capture)
    reason = "当前拟合尚未接入口述预测，行为拟合结果已经完成。"
    image, source = renderer.oral_alignment_figures(
        307, 3, None, {"status": "not_available", "reason": reason}, tmp_path,
    )
    assert image.is_file() and pd.read_csv(source).empty
    assert reason in "\n".join(label.get_text() for label in seen[0].axes[0].texts)
    assert not seen[0].axes[0].lines and not seen[0].axes[0].collections
    with pytest.raises(ValueError, match="Unavailable"):
        renderer.oral_alignment_figures(307, 3, _oral(), {"status": "not_available"}, tmp_path / "other")


@pytest.mark.parametrize("column,value", [
    ("oral_score_mask", [False, True, True, False, True]),
    ("report_probability", [np.nan, .3, np.nan, np.nan, -.1]),
    ("report_zero_probability", [False] * 5),
    ("rolling_top3_hit", [np.nan] * 4 + [1.1]),
])
def test_inconsistent_oral_scores_cannot_render(tmp_path: Path, column: str, value):
    trials = _oral().assign(**{column: value})
    with pytest.raises(ValueError):
        renderer.oral_alignment_figures(307, 3, trials, _summary(), tmp_path)
    assert not list(tmp_path.glob("*.png"))
