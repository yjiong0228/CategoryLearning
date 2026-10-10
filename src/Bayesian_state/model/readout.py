"""Compatibility imports for the shared observation readout package."""

from .modules.readout.choice import *  # noqa: F401,F403
from .modules.readout.choice import __all__ as _choice_exports
from .modules.readout.choice import DecisionRuleSnapshot, _one_hot, _output_noise_target_vector
from .modules.readout.shared import _bool_from_mapping, _float_from_mapping, _mapping_get_path
from .modules.readout.oral_kernel import OralReportReadoutResult, read_oral_report
from .modules.readout.reaction_time import ReactionTimeReadoutResult, read_reaction_time

__all__ = [*_choice_exports, "OralReportReadoutResult", "read_oral_report",
           "ReactionTimeReadoutResult", "read_reaction_time"]
