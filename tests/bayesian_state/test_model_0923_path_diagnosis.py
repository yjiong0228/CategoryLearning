"""Check residence timing, censoring and observation-wrapper isolation."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_path_diagnosis as diagnosis


def episode_arrays():
    ids = np.array([[[2, -1]], [[2, 3]], [[3, -1]], [[2, 3]], [[2, -1]]])
    a = {key: np.where(ids >= 0, .5, np.nan) for key in diagnosis.SLOT_FLOATS}
    a.update(rule=ids, proposed=np.array([[-1], [3], [-1], [2], [-1]]),
             source=np.array([[-1], [1], [-1], [2], [-1]]),
             capacity_dropped=np.array([[-1], [-1], [-1], [-1], [3]]),
             exits=np.array([[[-1, -1]], [[-1, -1]], [[2, -1]], [[-1, -1]], [[-1, -1]]]))
    return a


def test_episodes_preserve_reentry_exit_cause_and_right_censoring():
    rows = diagnosis.extract_episodes(episode_arrays(), 335, "B", "first")
    assert [(r["rule"], r["start_trial"], r["last_trial"], r["duration"]) for r in rows] == [
        (2, 1, 2, 2), (3, 2, 4, 3), (2, 4, 5, 2)]
    assert [r["exit_cause"] for r in rows] == ["autonomous_exit", "capacity", "censored"]
    assert [r["proposal_source"] for r in rows] == ["initial", "global", "structural"]
    assert rows[-1]["right_censored"]


def test_episode_rejects_unexplained_entry():
    a = episode_arrays()
    a["proposed"][1, 0] = 4
    with pytest.raises(ValueError, match="entry lacks"):
        diagnosis.extract_episodes(a, 335, "B", "first")


def test_episode_rejects_unexplained_loss():
    a = episode_arrays()
    a["exits"][2, 0, 0] = -1
    with pytest.raises(ValueError, match="disappeared"):
        diagnosis.extract_episodes(a, 335, "B", "first")


def test_recording_wrapper_restores_exports_even_on_failure():
    original = diagnosis.model_package.StateModel
    original_auto = diagnosis.autonomous.StateModel
    recorder = diagnosis.PathRecorder(np.array([1]), 1, 2)
    with pytest.raises(RuntimeError, match="deliberate"):
        with diagnosis.record_lifecycle(recorder):
            assert diagnosis.model_package.StateModel is diagnosis.autonomous.StateModel
            assert diagnosis.model_package.StateModel is not original
            raise RuntimeError("deliberate")
    assert diagnosis.model_package.StateModel is original
    assert diagnosis.autonomous.StateModel is original_auto


def test_recorder_rejects_incomplete_paths():
    with pytest.raises(ValueError, match="Incomplete"):
        diagnosis.PathRecorder(np.array([1]), 1, 2).validate()


def test_task_pairing_is_only_a_scoring_coordinate():
    assert diagnosis.true_pairing_index() == 0


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="new output"):
        diagnosis.run(diagnosis.Path("configs/shared/model0923_path_diagnosis.yaml"), tmp_path)
    assert sentinel.read_text() == "keep"
