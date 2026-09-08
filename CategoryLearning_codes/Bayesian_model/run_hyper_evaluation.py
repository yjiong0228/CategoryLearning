"""Journal evaluation defaults; evaluation implementation is shared."""
from src.Bayesian_state import run_hyper_evaluation as _shared
from src.Bayesian_state.utils.paths import RESULTS_DIR

DEFAULT_INPUT_DIR = RESULTS_DIR / "model_0826" / "hyper"
DEFAULT_CANDIDATES_JSON = None


def __getattr__(name):
    return getattr(_shared, name)


def parse_args(argv=None):
    return _shared.parse_args(argv, default_input_dir=DEFAULT_INPUT_DIR)


def infer_candidate_source(**kwargs):
    kwargs.setdefault("default_candidates_json", DEFAULT_CANDIDATES_JSON)
    return _shared.infer_candidate_source(**kwargs)


def main(argv=None):
    return _shared.main(argv, default_input_dir=DEFAULT_INPUT_DIR,
                        default_candidates_json=DEFAULT_CANDIDATES_JSON)


if __name__ == "__main__":
    main()
