#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_1024.sh [NEW_OUTPUT.pdf]
# Compile in a temporary directory and preserve existing PDF documents.
model1024_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_1024.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model1024_destination="${1:-$model1024_script_dir/model_1024.pdf}"
model1024_destination="$(realpath -m -- "$model1024_destination")"
if [[ -e "$model1024_destination" || -L "$model1024_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model1024_destination" >&2
    exit 1
fi
model1024_build="$(mktemp -d "${TMPDIR:-/tmp}/model1024-build.XXXXXXXX")"
echo "Build logs: $model1024_build"
cd "$model1024_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model1024_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model1024_build" xelatex.ini \
        > "$model1024_build/format.log" 2>&1
    model1024_compiler=(xetex "-fmt=$model1024_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model1024_pass in 1 2 3; do
    "${model1024_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model1024_build" model_1024.tex \
        > "$model1024_build/pass-$model1024_pass.log" 2>&1
    echo "Completed pass $model1024_pass"
done
mkdir -p -- "$(dirname -- "$model1024_destination")"
(set -o noclobber; cat "$model1024_build/model_1024.pdf" > "$model1024_destination")
echo "Compiled: $model1024_destination"
