#!/usr/bin/env bash
set -euo pipefail

modelv14_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$modelv14_script_dir"

modelv14_source="model_v14_standalone.tex"
modelv14_jobname="model_v14"
modelv14_output_dir="/tmp/model_v14_build"
mkdir -p "$modelv14_output_dir"

if command -v xelatex >/dev/null 2>&1; then
    for modelv14_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv14_output_dir" \
            -jobname="$modelv14_jobname" \
            "$modelv14_source"
    done
else
    modelv14_format_dir="/tmp/model_v14_xelatex_format"
    mkdir -p "$modelv14_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$modelv14_format_dir" \
        xelatex.ini

    for modelv14_pass in 1 2; do
        xetex \
            -fmt="$modelv14_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv14_output_dir" \
            -jobname="$modelv14_jobname" \
            "$modelv14_source"
    done
fi

cp "$modelv14_output_dir/$modelv14_jobname.pdf" "$modelv14_script_dir/$modelv14_jobname.pdf"
echo "Compiled: $modelv14_script_dir/$modelv14_jobname.pdf"
