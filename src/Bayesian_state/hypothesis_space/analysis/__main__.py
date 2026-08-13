"""Command-line entry point for the Task2 oral-evidence audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .oral_evidence import (
    DEFAULT_DATA,
    DEFAULT_DIAGNOSTICS,
    DEFAULT_OUTPUT_DIR,
    add_primitive_flags,
    candidate_catalog,
    cooccurrence_summary,
    coverage_scenarios,
    equality_tolerance_sensitivity,
    evidence_examples,
    load_and_validate,
    partition_inventory,
    subject_primitive_counts,
    summarize_primitives,
)
def run(args: argparse.Namespace) -> None:
    from .reporting import audit_sample, write_outputs

    data_path = Path(args.data)
    diagnostics_path = Path(args.diagnostics)
    output_dir = Path(args.output_dir)
    frame = add_primitive_flags(load_and_validate(data_path, diagnostics_path))
    subject_counts = subject_primitive_counts(frame)
    prevalence = pd.concat(
        [
            summarize_primitives(frame, subject_counts, ["task_arity"], level_kind="task_arity"),
            summarize_primitives(frame, subject_counts, ["task_arity", "condition"], level_kind="condition"),
        ],
        ignore_index=True,
        sort=False,
    )
    inventory, family_summary = partition_inventory()
    cooccurrence = cooccurrence_summary(frame)
    write_outputs(
        frame=frame,
        subject_counts=subject_counts,
        prevalence=prevalence,
        inventory=inventory,
        family_summary=family_summary,
        candidates=candidate_catalog(prevalence, cooccurrence),
        coverage=coverage_scenarios(frame, subject_counts),
        cooccurrence=cooccurrence,
        sensitivity=equality_tolerance_sensitivity(frame),
        examples=evidence_examples(frame),
        sample=audit_sample(frame, per_primitive=args.audit_sample_per_primitive),
        output_dir=output_dir,
        data_path=data_path,
        diagnostics_path=diagnostics_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-sample-per-primitive", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
