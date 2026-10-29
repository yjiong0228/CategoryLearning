"""Cohen–Schneidman binary model and continuous task extension."""
from .binary import FeatureMixtureStrict
from .fitting import FitConfig, fit_paper_style, fit_continuous
from .continuous import ContinuousFeatureMixture

__all__ = ["FeatureMixtureStrict", "FitConfig", "fit_paper_style", "ContinuousFeatureMixture", "fit_continuous"]
