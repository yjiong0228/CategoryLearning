# Model 1031 C++

This subtree maintains the ten-parameter `1031-unified-v2` C++ implementation.
Read README.md before changing configuration, CLI or output schemas.

- Keep model execution, PF, autonomous simulation and optimization in C++20/double.
  Python is for offline evaluation and explicit reference validation.
- Shared data lives at the main repository's `data/`; do not create a copy here.
  Paths in configs resolve relative to the config file. Resources and build
  dependencies remain local to this subtree.
- Other repository components must not require this subtree or its build.
  Keep CMake, Python dependencies, configs and tests independent.
- Preserve trial timing, pre-feedback Beta, conditional pairing/category learning
  and the distinction between predictive and choice-filtered state distributions.
  Workspace slots/UIDs are not shared identities across particles.
- Preserve frozen numerical fixtures. Never regenerate them to hide differences.
- Run CTest and `python -B -m pytest -c src/cpp/pytest.ini src/cpp/tests` after
  relevant changes. Use fresh outputs and 1-2 workers for small checks.
- Keep compiler downloads, binaries, logs and scientific outputs out of Git.
  Results already in the independent repository remain there.
