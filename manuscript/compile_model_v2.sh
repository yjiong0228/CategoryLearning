#!/usr/bin/env bash
set -euo pipefail

modelv2_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$modelv2_script_dir"

modelv2_source="model_v2_standalone.tex"
modelv2_jobname="model_v2"
modelv2_output_dir="/tmp/model_v2_build"
mkdir -p "$modelv2_output_dir"

if command -v xelatex >/dev/null 2>&1; then
    for modelv2_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv2_output_dir" \
            -jobname="$modelv2_jobname" \
            "$modelv2_source"
    done
else
    modelv2_format_dir="/tmp/model_v2_xelatex_format"
    mkdir -p "$modelv2_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$modelv2_format_dir" \
        xelatex.ini

    for modelv2_pass in 1 2; do
        xetex \
            -fmt="$modelv2_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv2_output_dir" \
            -jobname="$modelv2_jobname" \
            "$modelv2_source"
    done
fi

cp "$modelv2_output_dir/$modelv2_jobname.pdf" "$modelv2_script_dir/$modelv2_jobname.pdf"
echo "Compiled: $modelv2_script_dir/$modelv2_jobname.pdf"
