#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compile_model_1024_exp4.sh [NEW_OUTPUT.pdf]
# Keep build artifacts in a temporary directory and preserve existing PDFs.
exp4_doc_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
    echo 'Usage: compile_model_1024_exp4.sh [NEW_OUTPUT.pdf]' >&2
    exit 2
fi
exp4_doc_destination="${1:-$exp4_doc_dir/model_1024_exp4.pdf}"
exp4_doc_destination="$(realpath -m -- "$exp4_doc_destination")"
if [[ -e "$exp4_doc_destination" || -L "$exp4_doc_destination" ]]; then
    echo "Output already exists; choose a new PDF path: $exp4_doc_destination" >&2
    exit 1
fi
exp4_doc_build="$(mktemp -d "${TMPDIR:-/tmp}/model1024-exp4-build.XXXXXXXX")"
echo "Build logs: $exp4_doc_build"
cd "$exp4_doc_dir"
if command -v xelatex >/dev/null 2>&1; then
    exp4_doc_compiler=(xelatex)
elif command -v xetex >/dev/null 2>&1; then
    xetex -ini -etex -jobname=xelatex -output-directory="$exp4_doc_build" xelatex.ini \
        > "$exp4_doc_build/format.log" 2>&1
    exp4_doc_compiler=(xetex "-fmt=$exp4_doc_build/xelatex.fmt")
else
    echo 'XeLaTeX or XeTeX is required.' >&2
    exit 1
fi
for exp4_doc_pass in 1 2 3; do
    "${exp4_doc_compiler[@]}" -interaction=nonstopmode -halt-on-error \
        -file-line-error -output-directory="$exp4_doc_build" model_1024_exp4.tex \
        > "$exp4_doc_build/pass-$exp4_doc_pass.log" 2>&1
    echo "Completed pass $exp4_doc_pass"
done
mkdir -p -- "$(dirname -- "$exp4_doc_destination")"
(set -o noclobber; cat "$exp4_doc_build/model_1024_exp4.pdf" > "$exp4_doc_destination")
echo "Compiled: $exp4_doc_destination"
