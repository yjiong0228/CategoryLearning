"""Validate the opt-in experience-workspace research candidate."""
from collections.abc import Mapping


def validate_model_1031_config(config: Mapping) -> None:
    """Keep the representation, feedback and update-timing contracts together."""
    provenance = config.get("provenance", {})
    version = provenance.get("specification_version")
    revised = version == "1031-unified-v2"
    condition = provenance.get("condition")
    if (version not in ("1031-experience-v1", "1031-unified-v2")
            or condition not in ((1, 2, 3) if revised else (1, 2))):
        raise ValueError("Model 1031 requires a supported condition and declared specification version")
    expected = {
        "perception_mod": "src.Bayesian_state.model.modules.perception.PerceptionModule",
        "hypo_transitions_mod": "src.Bayesian_state.model.modules.hypothesis_transition.experience_workspace.ExperienceWorkspaceTransition",
        "memory_mod": "src.Bayesian_state.model.modules.hypothesis_transition.experience_workspace.ExperienceWorkspaceMemory",
        "beta_mod": "src.Bayesian_state.model.modules.hypothesis_transition.experience_workspace.ExperienceWorkspaceBeta",
    }
    if condition == 3:
        expected["memory_mod"] = "src.Bayesian_state.model.modules.hypothesis_transition.experience_workspace.ExperienceWorkspacePairingMemory"
    if config.get("agenda") != list(expected):
        raise ValueError("Model 1031 requires perception, workspace, memory, then precision/label learning")
    modules = config.get("modules", {})
    if set(modules) != set(expected) or any(modules[name].get("class") != cls for name, cls in expected.items()):
        raise ValueError("Model 1031 requires its shared workspace modules")
    partition = config.get("partition", {})
    if (partition.get("class") != "src.Bayesian_state.hypothesis_space.observation_model.experience_workspace.ExperienceWorkspacePartition"
            or partition.get("kwargs", {}).get("n_cats") != (2 if condition == 1 else 4)
            or partition.get("kwargs", {}).get("conditional_pairing", False) != (condition == 3)):
        raise ValueError("Model 1031 partition must agree with its condition")
    if config.get("likelihood", {}).get("class") != "src.Bayesian_state.hypothesis_space.observation_model.experience_workspace.ExperienceWorkspaceLikelihood":
        raise ValueError("Model 1031 requires workspace feedback likelihood")
    if config.get("oral_readout") is not None:
        raise ValueError("Model 1031 oral observation model is not implemented; export representation contents")
    beta = modules["beta_mod"].get("kwargs", {})
    if beta.get("beta_max") != 25. or beta.get("beta_min") != .1:
        raise ValueError("Model 1031 v1 retains precision bounds [0.1, 25]")
    transition = modules["hypo_transitions_mod"].get("kwargs", {})
    if transition.get("revision", "v1") != ("v2" if revised else "v1"):
        raise ValueError("Experience transition revision must match the model specification")
    if not revised and (transition.get("marginalize_execution", False)
                        or transition.get("retrieval_mode", "global") != "global"
                        or modules["memory_mod"].get("kwargs", {}).get("retention_mode", "outcome") != "outcome"):
        raise ValueError("Experience v1 retains its historical learning and execution policy")
