#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_1031.sh [NEW_OUTPUT.pdf]
# Compile in a temporary directory and preserve existing PDF documents.
model1031_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo 'Usage: bash compile_model_1031.sh [NEW_OUTPUT.pdf]'
    echo 'Requires XeLaTeX or XeTeX and Noto CJK fonts; preserves existing output files.'
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_1031.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
model1031_destination="${1:-$model1031_script_dir/model_1031.pdf}"
model1031_destination="$(realpath -m -- "$model1031_destination")"
if [[ -e "$model1031_destination" || -L "$model1031_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $model1031_destination" >&2
    exit 1
fi
model1031_build="$(mktemp -d "${TMPDIR:-/tmp}/model1031-build.XXXXXXXX")"
echo "Build logs: $model1031_build"
cd "$model1031_script_dir"
if command -v xelatex >/dev/null 2>&1; then
    model1031_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$model1031_build" xelatex.ini \
        > "$model1031_build/format.log" 2>&1
    model1031_compiler=(xetex "-fmt=$model1031_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for model1031_pass in 1 2 3; do
    "${model1031_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$model1031_build" model_1031.tex \
        > "$model1031_build/pass-$model1031_pass.log" 2>&1
    echo "Completed pass $model1031_pass"
done
mkdir -p -- "$(dirname -- "$model1031_destination")"
(set -o noclobber; cat "$model1031_build/model_1031.pdf" > "$model1031_destination")
echo "Compiled: $model1031_destination"
