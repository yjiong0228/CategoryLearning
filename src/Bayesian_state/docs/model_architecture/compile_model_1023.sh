#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_1023.sh [NEW_OUTPUT.pdf]
# Compile in a temporary directory and preserve existing PDF documents.
model1023_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_1023.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model1023_destination="${1:-$model1023_script_dir/model_1023.pdf}"
model1023_destination="$(realpath -m -- "$model1023_destination")"
if [[ -e "$model1023_destination" || -L "$model1023_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model1023_destination" >&2
    exit 1
fi
model1023_build="$(mktemp -d "${TMPDIR:-/tmp}/model1023-build.XXXXXXXX")"
echo "Build logs: $model1023_build"
cd "$model1023_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model1023_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model1023_build" xelatex.ini \
        > "$model1023_build/format.log" 2>&1
    model1023_compiler=(xetex "-fmt=$model1023_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model1023_pass in 1 2 3; do
    "${model1023_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model1023_build" model_1023.tex \
        > "$model1023_build/pass-$model1023_pass.log" 2>&1
    echo "Completed pass $model1023_pass"
done
mkdir -p -- "$(dirname -- "$model1023_destination")"
(set -o noclobber; cat "$model1023_build/model_1023.pdf" > "$model1023_destination")
echo "Compiled: $model1023_destination"
