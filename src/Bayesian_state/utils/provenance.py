"""Content provenance for resumable scientific runs (separate from output schemas).

Inputs must remain immutable during a run. Snapshots are taken before scores are
read or written; modifying a live worker's input files is unsupported.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import inspect
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .datasets import resolve_dataset_paths
from .paths import BAYESIAN_STATE_DIR, ROOT_DIR
from .subjects import deep_update, resolve_subject_config

FINGERPRINT_SCHEMA_VERSION = 2


def file_digest(path: Path) -> str | None:
    """Hash bytes, recording absence for optional inputs too."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class DependencySnapshot:
    """Collect source, referenced config, data and effective rule-resource content."""

    def __init__(self) -> None:
        self.files: dict[str, str | None] = {}
        self.resolved: list[dict[str, Any]] = []
        self.similarities: dict[str, str] = {}
        self._visited: set[Path] = set()

    def add_file(self, path: Path, *, follow_yaml: bool = False) -> None:
        path = path.resolve()
        if path in self._visited:
            return
        self._visited.add(path)
        self.files[str(path)] = file_digest(path)
        if follow_yaml and path.is_file() and path.suffix in {'.yaml', '.yml'}:
            self.scan(yaml.safe_load(path.read_text(encoding='utf-8')), path.parent)

    def scan(self, value: Any, base: Path, *, key: str = '') -> None:
        """Follow file references, including nested profile/candidate inputs."""
        if isinstance(value, Mapping):
            for name, item in value.items():
                # Generated artifacts must never become their own dependencies.
                if str(name) in {'output_dir', 'output_root', 'cache_dir', 'save_path'}:
                    continue
                self.scan(item, base, key=str(name))
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.scan(item, base, key=key)
        elif isinstance(value, (str, Path)):
            if key == 'class':
                from ..model.assembly import resolve_class
                cls = resolve_class(str(value))
                # Include base classes used by compatibility wrappers as well.
                for parent in cls.__mro__:
                    if parent is object:
                        continue
                    source = inspect.getsourcefile(parent)
                    if source:
                        self.add_file(Path(source))
                return
            path = Path(value)
            is_file_reference = path.suffix.lower() in {
                '.yaml', '.yml', '.json', '.csv', '.npy', '.npz', '.py', '.tex', '.pkl'
            } or key.endswith('_path') or key == 'path'
            if is_file_reference:
                self.add_file(path if path.is_absolute() else base / path, follow_yaml=True)
                # Provenance references commonly use repository-relative paths.
                if not path.is_absolute() and (ROOT_DIR / path).is_file():
                    self.add_file(ROOT_DIR / path, follow_yaml=True)

    def simulation(self, config: Mapping[str, Any], base: Path, subjects: Sequence[int]) -> None:
        from ..simulation.config import resolve_engine_config
        from ..model.modules.perception import _resolve_data_paths

        self.scan(config, base)
        for subject in subjects:
            sim = resolve_subject_config(config, subject)
            datasets = resolve_dataset_paths(sim, base)
            for key, path in datasets.items():
                if key != 'processed_dir':
                    self.add_file(path)
            engine = None
            if sim.get('engine_config') is not None or sim.get('engine_config_path'):
                engine = resolve_engine_config(sim, base, subject_id=subject)
                self.scan(engine, base)
                for module in (engine.get('modules') or {}).values():
                    kwargs = module.get('kwargs', {}) or {}
                    # Perception allows explicit kwargs to override runner datasets.
                    paths = kwargs.get('dataset_paths', datasets)
                    processed = kwargs.get('processed_data_dir', datasets['processed_dir'])
                    for path in _resolve_data_paths(processed, paths):
                        self.add_file(path)
                self._similarity(engine)
            self.resolved.append({'subject': int(subject), 'simulation': sim, 'engine': engine})

    def _similarity(self, engine: Mapping[str, Any]) -> None:
        from ..model.assembly import build_partition
        from ..hypothesis_space.observation_model import ContinuousPartition

        # The standard partition is deterministic; hash the matrix actually used,
        # including a runtime-cache override. Materialize once before checkpointing
        # so a newly generated cache does not invalidate this run's own checkpoint.
        partition = build_partition(engine, int(engine.get('condition', 1)))
        if not isinstance(partition, ContinuousPartition):
            return
        mode = (engine.get('likelihood') or {}).get('distance_mode', 'boundary')
        similarity = partition.similarity
        key = similarity._cache_key(
            distance_mode=mode, n_samples=similarity.n_samples,
            random_state=similarity.DEFAULT_RANDOM_STATE, sample_distribution='uniform',
        )
        name = repr((key, str(similarity.runtime_cache_dir.resolve())))
        if name in self.similarities:
            return
        # Do not let an earlier run's in-memory matrix hide a changed disk input.
        disk = similarity._load_valid_matrix(similarity._cache_path(key))
        if disk is None:
            disk = similarity._load_valid_matrix(similarity._compatible_resource_path(
                mode, similarity.n_samples, similarity.DEFAULT_RANDOM_STATE))
        memory = similarity._memory_cache.get(key)
        if disk is not None and memory is not None and not np.array_equal(disk, memory):
            raise ValueError('Similarity content changed while cached in memory; restart the process')
        matrix = similarity.get_matrix(distance_mode=mode)
        self.similarities[name] = hashlib.sha256(np.asarray(matrix, dtype='<f8').tobytes()).hexdigest()

    def payload(self) -> dict[str, Any]:
        # Include the complete maintained implementation, not a manually selected
        # subset that can omit memory, beta, transition or helper dependencies.
        for path in sorted(BAYESIAN_STATE_DIR.rglob('*.py')):
            self.add_file(path)
        for path in sorted((BAYESIAN_STATE_DIR / 'hypothesis_space/resources').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                self.add_file(path)
        versions = {'python': platform.python_version()}
        for package in ('numpy', 'scipy', 'pandas', 'numba', 'joblib', 'PyYAML'):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = 'not-installed'
        return {'fingerprint_schema_version': FINGERPRINT_SCHEMA_VERSION,
                'files': self.files, 'resolved': self.resolved,
                'similarity_matrices': self.similarities, 'versions': versions}


def search_dependencies(search: Mapping[str, Any], base_sim: Mapping[str, Any],
                        subjects: Sequence[int], config_dir: Path) -> dict[str, Any]:
    """Snapshot all configured stages, including final scoring and subject overrides."""
    config_dir = config_dir.resolve()
    snapshot = DependencySnapshot()
    snapshot.scan(search, config_dir)
    base_path = Path(search.get('base_sim_config_path', config_dir / 'base.yaml'))
    if not base_path.is_absolute():
        base_path = config_dir / base_path
    sim_dir = base_path.resolve().parent
    snapshot.simulation(base_sim, sim_dir, subjects)
    for stage in (search.get('stages') or {}).values():
        snapshot.simulation(deep_update(dict(base_sim), stage.get('simulation_overrides') or {}), sim_dir, subjects)
    final = search.get('final_rescore') or {}
    if final.get('simulation_overrides'):
        snapshot.simulation(deep_update(dict(base_sim), final['simulation_overrides']), sim_dir, subjects)
    # Numeric cognitive coordinates are supported. Changing external input or
    # geometry definitions across candidates needs an explicit dependency design.
    def check_coordinates(value: Any) -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                text = str(name)
                if text in {'engine', 'engine.likelihood', 'engine.n_dims', 'engine.n_cats',
                            'engine.modules', 'simulation'} or text.startswith(('engine.partition', 'engine.likelihood.distance_mode',
                                    'simulation.dataset', 'simulation.data_path',
                                    'simulation.engine_config', 'simulation.subject_overrides',
                                    'simulation.subject_configs', 'simulation.per_subject')) or (
                    text.startswith('engine.modules.') and len(text.split('.')) <= 4
                ) or (
                    text.startswith('engine.') and any(part in text for part in
                    ('.dataset_paths', '.processed_data_dir', '.class'))
                ):
                    raise ValueError(f'Resumable search requires fixed input/geometry dependencies: {text}')
                check_coordinates(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_coordinates(item)
    check_coordinates(search)
    # Candidate profiles may live in JSON rather than inline YAML.
    def check_imported_profiles(value: Any) -> None:
        if isinstance(value, Mapping):
            if 'values_from_json' in value:
                from ..optimization.artifacts import values_from_json
                candidates = values_from_json(value, config_dir)
                check_coordinates(candidates)
                snapshot.scan(candidates, config_dir)
            for item in value.values():
                check_imported_profiles(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_imported_profiles(item)
    check_imported_profiles(search)
    return snapshot.payload()
