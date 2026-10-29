"""Compatibility imports and task-selectable CLI; see README.md for usage."""
from importlib import import_module

if not __package__:  # Preserve direct execution of the historical script path.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.Cohen"

_EXPORTS = {
    "all_binary_patterns": "features",
    "feature_subsets": "features",
    "transform_features": "features",
    "check_pm1": "features",
    "normalize": "features",
    "logsumexp": "features",
    "sigmoid_neg_logodds": "features",
    "feature_name": "features",
    "FeatureMixtureStrict": "binary",
    "ContinuousFeatureMixture": "continuous",
    "create_model": "api",
    "FitConfig": "fitting",
    "_pack": "fitting",
    "_unpack": "fitting",
    "nll": "fitting",
    "fit_paper_style": "fitting",
    "fit_continuous": "fitting",
    "_fmt_float": "trace",
    "_named_vector": "trace",
    "_plain_vector": "trace",
    "_make_inference_formula": "trace",
    "run_session_with_formula_trace": "trace",
    "sti_to_bits": "data",
    "prepare_dataframe": "data",
    "load_csv": "data",
    "sort_subjects": "data",
    "subject_id_values": "data",
    "id_columns": "data",
    "save_alpha_trajectories": "exports",
    "make_wide_df": "exports",
    "make_alpha_long_df": "exports",
    "concat_run_df": "exports",
    "plot_feature_trajectories": "plotting",
    "plot_subject_learning_curves": "plotting",
    "plot_subject_feature_grid": "plotting",
    "model_feature_to_csv_column": "legacy_oral",
    "oral_feature_matrix": "legacy_oral",
    "save_oral_trajectories": "legacy_oral",
    "row_cosine": "legacy_oral",
    "row_pearson": "legacy_oral",
    "save_similarity_outputs": "legacy_oral",
    "make_oral_long_df": "legacy_oral",
    "make_similarity_df": "legacy_oral",
    "plot_subject_similarity_grid": "legacy_oral",
    "plot_subject_target_grid": "legacy_oral",
    "run_one_subject": "binary_workflow",
    "write_combined_outputs": "binary_workflow",
    "main": "cli",
    "DEFAULT_EXP5_CSV": "data"
}

__all__ = [name for name in _EXPORTS if not name.startswith("_")]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __package__), name)
    globals()[name] = value
    return value


if __name__ == "__main__":
    __getattr__("main")()
