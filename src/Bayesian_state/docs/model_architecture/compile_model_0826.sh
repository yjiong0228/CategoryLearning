#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_0826.sh [NEW_OUTPUT.pdf]
# The source is standalone. Keep existing proofs and temporary build logs.
model0826_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_0826.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model0826_destination="${1:-$model0826_script_dir/model_0826.pdf}"
model0826_destination="$(realpath -m -- "$model0826_destination")"
if [[ -e "$model0826_destination" || -L "$model0826_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model0826_destination" >&2
    exit 1
fi
model0826_build="$(mktemp -d "${TMPDIR:-/tmp}/model0826-build.XXXXXXXX")"
echo "Build logs: $model0826_build"
cd "$model0826_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model0826_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model0826_build" xelatex.ini
    model0826_compiler=(xetex "-fmt=$model0826_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model0826_pass in 1 2; do
    "${model0826_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model0826_build" model_0826.tex
done
mkdir -p -- "$(dirname -- "$model0826_destination")"
# noclobber protects against a destination created during compilation.
(set -o noclobber; cat "$model0826_build/model_0826.pdf" > "$model0826_destination")
echo "Compiled: $model0826_destination"
