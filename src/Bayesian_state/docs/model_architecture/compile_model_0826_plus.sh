#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_0826_plus.sh [NEW_OUTPUT.pdf]
# The source is standalone. Keep existing proofs and temporary build logs.
model0826plus_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_0826_plus.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model0826plus_destination="${1:-$model0826plus_script_dir/model_0826_plus.pdf}"
model0826plus_destination="$(realpath -m -- "$model0826plus_destination")"
if [[ -e "$model0826plus_destination" || -L "$model0826plus_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model0826plus_destination" >&2
    exit 1
fi
model0826plus_build="$(mktemp -d "${TMPDIR:-/tmp}/model0826plus-build.XXXXXXXX")"
echo "Build logs: $model0826plus_build"
cd "$model0826plus_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model0826plus_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model0826plus_build" xelatex.ini
    model0826plus_compiler=(xetex "-fmt=$model0826plus_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model0826plus_pass in 1 2; do
    "${model0826plus_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model0826plus_build" model_0826_plus.tex
done
mkdir -p -- "$(dirname -- "$model0826plus_destination")"
# noclobber protects against a destination created during compilation.
(set -o noclobber; cat "$model0826plus_build/model_0826_plus.pdf" > "$model0826plus_destination")
echo "Compiled: $model0826plus_destination"
