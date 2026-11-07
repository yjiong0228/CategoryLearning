"""Portable handoff tests use synthetic data only, never model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "evaluation/collaboration_io.py"


@pytest.fixture
def io():
    assert HELPER.is_file(), "portable collaboration loader is not implemented"
    spec = importlib.util.spec_from_file_location("python1031_io", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "portable bundle with spaces"
    root.mkdir()
    for condition, groups, categories in [(1, 1, 2), (2, 1, 4), (3, 3, 4)]:
        subject = root / f"subjects/cond{condition}/subject_101"
        native = subject / "native"
        native.mkdir(parents=True)
        metadata = dict(condition=condition, subject=101, n_categories=categories,
                        feature_names=["tail", "leg", "head", "neck"], pf_repeats=16,
                        autonomous_repeats=1, conditional_particle_file="native/conditional_particle_contents.npz",
                        conditional_particle_contents=dict(pf_repeat_index=0, seed=81))
        (subject / "python1031_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez(subject / "python1031_fit.npz", choice_probabilities=np.full((2, categories), 1 / categories),
                 choices=[1, 0], categories=[1, 2], feedback=[1., .5], valid=[True, False])
        np.savez(subject / "python1031_generation.npz", feedback=[[1., .5]], choices=[[1, 2]], seeds=[81])
        np.savez(native / "conditional_particle_contents.npz",
                 particle_weights=[[.25, .75], [.5, .5]],
                 particle_predictive_weights=[[.5, .5], [.5, .5]],
                 particle_template_category_belief=np.full((2, 2, 3, groups, categories), 1 / categories),
                 particle_pairing=np.full((2, 2, 3, groups), 1 / groups),
                 particle_template_mask=np.ones((2, 2, 3, 4), dtype=np.int64),
                 particle_active=np.ones((2, 2, 3), dtype=np.int64),
                 particle_representation_kind=np.broadcast_to([0, 1, 2], (2, 2, 3)),
                 particle_representation_uid=np.arange(12).reshape(2, 2, 3),
                 particle_catalogue_rule=np.full((2, 2, 3), -1),
                 particle_future_field=[42], stimulus=np.zeros((2, 4)), choices=[0, -1], valid=[1, 1])
    write_manifest(root)
    return root


def write_manifest(root, files=None):
    if files is None:
        files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    (root / "manifest.json").write_text(json.dumps(dict(files=files, subjects=[
        dict(condition=c, subject_id=101) for c in (1, 2, 3)])), encoding="utf-8")


def test_load_subject_preserves_metadata_and_arrays(io, bundle):
    result = io.load_subject(bundle, "cond2", "subject_101")
    assert result["condition"] == 2
    assert result["pf_repeats"] == 16
    assert result["conditional_particle_contents"]["pf_repeat_index"] == 0
    assert result["conditional_particle_file"] == "native/conditional_particle_contents.npz"
    assert result["n_categories"] == 4
    assert "files" not in result
    np.testing.assert_array_equal(result["fit"]["choices"], [1, 0])
    np.testing.assert_array_equal(result["generation"]["feedback"], [[1., .5]])
    assert "ess" not in result["fit"]


@pytest.mark.parametrize("condition,groups,categories", [(1, 1, 2), (2, 1, 4), (3, 3, 4)])
def test_particle_names_dimensions_and_types(io, bundle, condition, groups, categories):
    with io.load_particles(bundle, condition, 101) as p:
        conditional = p["conditional_template_category_belief"]
        assert conditional.shape == (2, 2, 3, groups, categories)
        assert p["template_category_belief"].shape == ((2, 2, 3, categories) if groups == 1 else conditional.shape)
        assert p["pairing"].shape == (2, 2, 3, groups)
        assert p["template_mask"].dtype == np.bool_
        assert p["active"].dtype == np.bool_
        np.testing.assert_array_equal(p["representation_kind"][0, 0], [0, 1, 2])
        np.testing.assert_array_equal(p["weights"][0], [.25, .75])
        assert p["representation_uid"][1, 1, 2] == 11
        assert p["future_field"][0] == 42
        np.testing.assert_array_equal(p["choices"], [0, -1])
        assert "stimulus" in p and "catalogue_rule" in p
        for absent in ("ess", "dominant", "prediction", "search", "particle_weights", "rule_uid"):
            assert absent not in p
            with pytest.raises(KeyError):
                p[absent]
        p.validate_weights()
    with pytest.raises(ValueError, match="closed"):
        p["weights"]


def test_particle_open_and_inventory_are_lazy(io, bundle, monkeypatch):
    accessed = []
    original = np.lib.npyio.NpzFile.__getitem__
    def tracked(self, key):
        accessed.append(key)
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    with io.load_particles(bundle, 3, 101) as p:
        assert len(p) > 10
        assert "template_category_belief" in list(p)
        assert "template_category_belief" in p
        assert accessed == []
        assert p["weights"].shape == (2, 2)
        assert accessed == ["particle_weights"]


def test_missing_packet_fields_remain_absent(io, bundle):
    path = bundle / "subjects/cond1/subject_101/native/conditional_particle_contents.npz"
    np.savez(path, particle_weights=[[1.]])
    with io.load_particles(bundle, 1, 101) as p:
        assert set(p) == {"weights"}
        assert "conditional_template_category_belief" not in p


def test_explicit_weight_validation_rejects_bad_weights(io, bundle):
    path = bundle / "subjects/cond1/subject_101/native/conditional_particle_contents.npz"
    np.savez(path, particle_weights=[[.2, .2]], valid=[True])
    with io.load_particles(bundle, 1, 101) as p:
        with pytest.raises(ValueError, match="weights"):
            p.validate_weights()


def test_verify_and_tampering(io, bundle):
    report = io.verify_bundle(bundle)
    assert report["files_checked"] == 12
    assert report["bytes_checked"] > 0
    path = bundle / "subjects/cond1/subject_101/python1031_metadata.json"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        io.verify_bundle(bundle)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../escape", "C:/absolute", "C:relative", "a\\b", "//host/share", "", "a//b", "./a"])
def test_manifest_rejects_unsafe_paths(io, bundle, path):
    write_manifest(bundle, [dict(path=path, sha256="0" * 64, bytes=0)])
    with pytest.raises(ValueError, match="path"):
        io.verify_bundle(bundle)


@pytest.mark.parametrize("change", [dict(sha256="z" * 64), dict(sha256="abc"), dict(bytes=-1), dict(bytes=True), dict(bytes=1.5)])
def test_manifest_rejects_invalid_records(io, bundle, change):
    record = json.loads((bundle / "manifest.json").read_text())["files"][0]
    record.update(change)
    write_manifest(bundle, [record])
    with pytest.raises(ValueError):
        io.verify_bundle(bundle)


def test_manifest_rejects_duplicate_paths_and_size_mismatch(io, bundle):
    record = json.loads((bundle / "manifest.json").read_text())["files"][0]
    write_manifest(bundle, [record, record])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        io.verify_bundle(bundle)
    record["bytes"] += 1
    write_manifest(bundle, [record])
    with pytest.raises(ValueError, match="[Bb]ytes|[Ss]ize"):
        io.verify_bundle(bundle)


def test_manifest_rejects_symlinks(io, bundle, tmp_path):
    target = tmp_path / "outside"
    target.write_bytes(b"secret")
    link = bundle / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted")
    write_manifest(bundle, [dict(path="link", bytes=6, sha256=hashlib.sha256(b"secret").hexdigest())])
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        io.verify_bundle(bundle)


def test_manifest_rejects_symlink_directory(io, bundle, tmp_path):
    outside = tmp_path / "outside directory"
    outside.mkdir()
    (outside / "file").write_bytes(b"secret")
    try:
        (bundle / "linked directory").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted")
    write_manifest(bundle, [dict(path="linked directory/file", bytes=6,
                                 sha256=hashlib.sha256(b"secret").hexdigest())])
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        io.verify_bundle(bundle)


def test_verify_streams_large_file(io, bundle, monkeypatch):
    payload = b"x" * (2 * 1024 * 1024 + 17)
    (bundle / "large synthetic file").write_bytes(payload)
    write_manifest(bundle, [dict(path="large synthetic file", bytes=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest())])
    original = Path.open
    reads = []
    class Reader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.stream.close()
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            reads.append(size)
            return self.stream.read(size)
    def opened(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return Reader(stream) if mode == "rb" else stream
    monkeypatch.setattr(Path, "open", opened)
    assert io.verify_bundle(bundle)["bytes_checked"] == len(payload)
    assert len(reads) == 4


def test_weight_check_excludes_invalid_trials(io, bundle):
    path = bundle / "subjects/cond1/subject_101/native/conditional_particle_contents.npz"
    np.savez(path, particle_weights=[[.25, .75], [np.nan, np.nan]], valid=[1, 0])
    with io.load_particles(bundle, 1, 101) as p:
        p.validate_weights()
        assert np.isnan(p["weights"][1]).all()
    p.close()


def test_particle_key_collision_is_rejected(io, bundle):
    path = bundle / "subjects/cond1/subject_101/native/conditional_particle_contents.npz"
    np.savez(path, particle_weights=[[1.]], weights=[[.5]])
    with pytest.raises(ValueError, match="Duplicate"):
        io.load_particles(bundle, 1, 101)


@pytest.mark.parametrize("manifest", [[], {}, {"files": {}}, {"files": [None]}])
def test_invalid_manifest_structure(io, bundle, manifest):
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        io.verify_bundle(bundle)


@pytest.mark.parametrize("condition,subject", [("../cond1", 101), (1, "../101"), (4, 101), (True, 101)])
def test_subject_identifiers_are_local_components(io, bundle, condition, subject):
    with pytest.raises(ValueError):
        io.load_subject(bundle, condition, subject)


def test_copied_helper_has_no_repository_dependencies(io, bundle):
    helper = bundle / "python1031_io.py"
    shutil.copyfile(HELPER, helper)
    write_manifest(bundle)
    code = ("import sys; import python1031_io as io; "
            "assert io.load_subject('.', 1, 101)['subject'] == 101; "
            "p=io.load_particles('.', 3, 101); assert p['pairing'].shape[-1]==3; p.close(); "
            "assert not any(n.split('.')[0] in {'scipy','matplotlib','src','evaluation'} for n in sys.modules)")
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=bundle, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    run = subprocess.run([sys.executable, "-B", str(helper), "--verify", str(bundle)],
                         cwd=bundle, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["files_checked"] == 13
    (bundle / "subjects/cond1/subject_101/python1031_fit.npz").write_bytes(b"broken")
    run = subprocess.run([sys.executable, "-B", str(helper), "--verify", str(bundle)],
                         cwd=bundle, capture_output=True, text=True)
    assert run.returncode != 0
