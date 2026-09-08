# Category learning analysis and figures

- `figures/fig1/`: Fig1 code, configuration, tests and review notes.
- `figures/fig2/`: Fig2 code, diagnostics, plans and review notes.
- `figures/outputs/fig1/`: Fig1 versions, source snapshots and data-update audits.
- `figures/outputs/fig2/`: Fig2 versions and oral alignment diagnostics.
- [Figure output index](figures/outputs/README.md): current review images and directory map.
- `tests/`: integrated preprocessing regression tests.
- `CategoryLearning_paper/figures/Figure1.png`: confirmed exact copy of Fig1 v10.

The one-time data_preparation utility was removed after its work was integrated into
src/preprocess_b.py. Existing update backups and audit files remain under figures/outputs/fig1/.

Fig2 content and implementation prerequisites: figures/fig2/FIG2_PLAN.md.
The shared Model 0826 implementation is maintained in `src/Bayesian_state/` for both
journal and dissertation work. `Bayesian_model/` contains journal configurations,
entrypoints, regression tests and compatibility imports, rather than duplicate algorithms.
See [Bayesian_model/README.md](Bayesian_model/README.md) for ownership and publication
version freezes. Four-category/partial-feedback extension and formal fitting remain
separate future work.
