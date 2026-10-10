#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_0923.sh [NEW_OUTPUT.pdf]
# Standalone source; preserve existing PDFs and retain temporary build logs.
model0923_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_0923.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model0923_destination="${1:-$model0923_script_dir/model_0923.pdf}"
model0923_destination="$(realpath -m -- "$model0923_destination")"
if [[ -e "$model0923_destination" || -L "$model0923_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model0923_destination" >&2
    exit 1
fi
model0923_build="$(mktemp -d "${TMPDIR:-/tmp}/model0923-build.XXXXXXXX")"
echo "Build logs: $model0923_build"
cd "$model0923_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model0923_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model0923_build" xelatex.ini
    model0923_compiler=(xetex "-fmt=$model0923_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model0923_pass in 1 2 3; do
    "${model0923_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model0923_build" model_0923.tex
done
mkdir -p -- "$(dirname -- "$model0923_destination")"
# Protect against a destination created during compilation.
(set -o noclobber; cat "$model0923_build/model_0923.pdf" > "$model0923_destination")
echo "Compiled: $model0923_destination"
