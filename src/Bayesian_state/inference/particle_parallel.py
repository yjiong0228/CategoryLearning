"""Particle blocks of one filter, with global weights and resampling.

Each process owns a fixed set of global particle indices. Per-trial predictions
and feedback summaries are assembled in that index order; every block then uses
the same weights and ancestor indices. The shared executor is the sole process
parallel layer for candidate, repeat and particle computations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
import multiprocessing as mp
from threading import Condition
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from joblib.externals.loky import ProcessPoolExecutor
from joblib.parallel import get_active_backend
from threadpoolctl import threadpool_limits


NUMERIC_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)}


class ProcessAdmission:
    """Reserve a whole block group so partial filters cannot exhaust the pool."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.available = workers
        self.ready_filters = 0
        self.condition = Condition()

    @contextmanager
    def reserve(self, count: int) -> Iterator[None]:
        if count > self.workers:
            raise ValueError("Particle blocks exceed the shared process budget")
        with self.condition:
            self.condition.wait_for(lambda: self.available >= count)
            self.available -= count
        try:
            yield
        finally:
            with self.condition:
                self.available += count
                self.condition.notify_all()

    @contextmanager
    def plan(self, filters: int, block_limit: int, adaptive: bool) -> Iterator[int]:
        with self.condition:
            self.ready_filters += filters
            capacity = max(1, self.workers // self.ready_filters)
            limit = min(block_limit, capacity) if adaptive else block_limit
            blocks = block_limit
            if adaptive:
                blocks = 1
                while blocks * 2 <= limit:
                    blocks *= 2
        try:
            yield blocks
        finally:
            with self.condition:
                self.ready_filters -= filters


_runtime: ContextVar[Any] = ContextVar("particle_executor", default=None)
_block: ContextVar[Any] = ContextVar("particle_block", default=None)


def initialize_particle_pool(executor: Any, workers: int) -> None:
    executor.particle_admission = ProcessAdmission(workers)


def run_in_particle_pool(function: Any, args: tuple, kwargs: dict,
                         executor: Any) -> Any:
    token = _runtime.set((executor, executor.particle_admission))
    try:
        return function(*args, **kwargs)
    finally:
        _runtime.reset(token)


def evaluate_particle_jobs(jobs: list[tuple], executor: Any,
                           workers: int, blocks: int) -> list[Any]:
    def evaluate(job: tuple) -> Any:
        function, args, kwargs = job
        if blocks == 1:
            with executor.particle_admission.reserve(1):
                return executor.submit(function, *args, **kwargs).result()
        return run_in_particle_pool(function, args, kwargs, executor)
    with ThreadPoolExecutor(max_workers=min(len(jobs), max(1, workers // blocks))) as controllers:
        futures = [controllers.submit(evaluate, job) for job in jobs]
        return [future.result() for future in futures]


@dataclass
class ParticleBlock:
    connection: Any
    indices: np.ndarray
    particle_count: int
    buffers: dict = field(default_factory=dict)

    def gather_arrays(self, frame: Mapping[str, Any], names: Sequence[str]) -> None:
        phase = "pre" if "particle_predictions" in names else "post"
        if phase not in self.buffers:
            values = {name: frame[name][self.indices].copy() for name in names
                      if isinstance(frame.get(name), np.ndarray)
                      and frame[name].shape[0] == self.particle_count}
            self.connection.send(("initialize_" + phase, self.indices, values))
            descriptor = self.connection.recv()
            memory = np.memmap(descriptor["path"], mode="r+", dtype=np.uint8)
            views = {name: np.ndarray(shape, dtype=dtype, buffer=memory, offset=offset)
                     for name, (shape, dtype, offset) in descriptor["arrays"].items()}
            self.buffers[phase] = memory, views
        else:
            _, views = self.buffers[phase]
            for name, view in views.items():
                view[self.indices] = frame[name][self.indices]
            self.connection.send(("ready_" + phase, self.indices, None))
            self.connection.recv()
        for name, value in self.buffers[phase][1].items():
            frame[name][:] = value

    def gather_snapshots(self, models: Sequence[Any], ancestors: np.ndarray) -> list[Any]:
        from .backends.particle_filter import _snapshot
        selected = set(int(index) for index in ancestors)
        values = {int(index): _snapshot(models[index]) for index in self.indices
                  if int(index) in selected}
        needed = set(int(ancestors[index]) for index in self.indices)
        self.connection.send(("snapshots", self.indices, (values, needed)))
        snapshots = self.connection.recv()
        return [snapshots.get(int(index)) for index in ancestors]


def current_particle_block() -> ParticleBlock | None:
    return _block.get()


def particle_items(values: Sequence[Any]) -> Any:
    block = current_particle_block()
    return enumerate(values) if block is None else ((int(i), values[i]) for i in block.indices)


def gather_particle_arrays(frame: Mapping[str, Any], names: Sequence[str]) -> None:
    block = current_particle_block()
    if block is not None:
        block.gather_arrays(frame, names)


def _run_block(arguments: dict, connection: Any, indices: np.ndarray) -> Any:
    from .backends.particle_filter import run_state_model_particle_filter
    token = _block.set(ParticleBlock(connection, indices, int(arguments["particle_count"])))
    try:
        result = run_state_model_particle_filter(**arguments)
        connection.send(("done", indices, None))
        return result if int(indices[0]) == 0 else None
    finally:
        block = _block.get()
        for memory, _ in block.buffers.values():
            memory._mmap.close()
        _block.reset(token)
        connection.close()


def _run_group(arguments: dict, executor: Any, admission: ProcessAdmission) -> Any:
    count = int(arguments["particle_blocks"])
    particles = int(arguments["particle_count"])
    with admission.reserve(count), TemporaryDirectory(prefix="model_pf_blocks_") as temporary:
        connections, futures = [], []
        memories = []
        try:
            for indices in np.array_split(np.arange(particles), count):
                parent, remote = mp.Pipe()
                connections.append(parent)
                futures.append(executor.submit(_run_block, arguments, remote, indices))
                # Keep the sending endpoint alive until loky's feeder serializes it.
                connections.append(remote)
            parents = connections[::2]
            while True:
                packets = []
                for connection, future in zip(parents, futures):
                    while not connection.poll(1):
                        if future.done():
                            future.result()
                            raise RuntimeError("Particle block ended before synchronization")
                    packets.append(connection.recv())
                modes = {packet[0] for packet in packets}
                if len(modes) != 1:
                    raise RuntimeError("Particle blocks disagree on synchronization phase")
                if modes == {"done"}:
                    return futures[0].result()
                mode = packets[0][0]
                if mode.startswith("ready_"):
                    for connection in parents:
                        connection.send(None)
                    continue
                merged: dict = {}
                for mode, indices, values in packets:
                    if mode == "snapshots":
                        merged.update(values[0])
                    else:
                        for name, rows in values.items():
                            if name not in merged:
                                merged[name] = np.empty((particles, *rows.shape[1:]), dtype=rows.dtype)
                            merged[name][indices] = rows
                if mode == "snapshots":
                    for connection, packet in zip(parents, packets):
                        connection.send({index: merged[index] for index in packet[2][1]})
                else:
                    descriptor = {"path": str(Path(temporary) / (mode + ".dat")), "arrays": {}}
                    total = 0
                    for name, value in merged.items():
                        descriptor["arrays"][name] = (value.shape, value.dtype.str, total)
                        total += ((value.nbytes + 63) // 64) * 64
                    # A one-byte map also supports empty post-feedback summaries.
                    memory = np.memmap(descriptor["path"], mode="w+", dtype=np.uint8,
                                       shape=(max(1, total),))
                    memories.append(memory)
                    for name, (shape, dtype, offset) in descriptor["arrays"].items():
                        np.ndarray(shape, dtype=dtype, buffer=memory, offset=offset)[:] = merged[name]
                    for connection in parents:
                        connection.send(descriptor)
        finally:
            for connection in connections:
                connection.close()
            for future in futures:
                if not future.done():
                    future.cancel()
            for memory in memories:
                memory._mmap.close()


def execute_particle_blocks(function: Any, frame: Mapping[str, Any]) -> Any:
    arguments = {name: frame[name] for name in inspect.signature(function).parameters}
    count = int(arguments["particle_blocks"])
    if count > int(arguments["particle_count"]):
        raise ValueError("particle_blocks cannot exceed particle_count")
    if arguments.get("choice_transmission_audit"):
        raise ValueError("Particle blocks currently require choice transmission audit disabled")
    if arguments["engine_config"].get("oral_readout") is not None:
        raise ValueError("Particle blocks currently require oral_readout disabled")
    runtime = _runtime.get()
    if runtime is not None:
        return _run_group(arguments, *runtime)
    backend, _ = get_active_backend()
    if (getattr(backend, "nesting_level", 0) or 0) > 0:
        raise ValueError("Particle blocks require a shared executor inside a process-parallel fit")
    from ..utils.parallel import parallel_job_count
    if parallel_job_count(count, count) < count:
        raise ValueError("particle_blocks exceeds the available CPU allocation")
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=count, env=NUMERIC_ENV) as executor:
        return _run_group(arguments, executor, ProcessAdmission(count))
