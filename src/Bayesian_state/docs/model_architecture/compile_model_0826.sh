#!/usr/bin/env bash
set -euo pipefail

model0826_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0826_script_dir"

model0826_jobname="model_0826"
model0826_output_dir="/tmp/model_0826_build"
mkdir -p "$model0826_output_dir"
model0826_source="$model0826_output_dir/model_0826_build.tex"

# Keep model_0826.tex embeddable in the manuscript. The standalone wrapper is
# generated only in the temporary proof-build directory.
printf '%s\n' \
    '% !TeX program = xelatex' \
    '\documentclass[12pt]{article}' \
    '\usepackage[a4paper,margin=2.5cm]{geometry}' \
    '\usepackage{amsmath,amssymb,bm}' \
    '\usepackage{graphicx}' \
    '\usepackage{etoolbox}' \
    '\usepackage{fontspec}' \
    '\usepackage[unicode,hidelinks]{hyperref}' \
    '\defaultfontfeatures{Ligatures=TeX}' \
    '\setmainfont[AutoFakeSlant=0.2]{Noto Serif CJK SC}' \
    '\setsansfont{Noto Sans CJK SC}' \
    '\setmonofont{Noto Sans Mono CJK SC}' \
    '\XeTeXlinebreaklocale "zh"' \
    '\XeTeXlinebreakskip=0pt plus 1pt minus 0.1pt' \
    '\setlength{\parindent}{2em}' \
    '\setlength{\parskip}{0.25em}' \
    '\setlength{\emergencystretch}{3em}' \
    '\renewcommand{\thesubsubsection}{\arabic{subsubsection}}' \
    '\allowdisplaybreaks' \
    '\sloppy' \
    '\AtBeginEnvironment{align}{\fontsize{9}{11}\selectfont}' \
    '\begin{document}' \
    '\input{model_0826.tex}' \
    '\end{document}' \
    > "$model0826_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0826_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0826_output_dir" \
            -jobname="$model0826_jobname" \
            "$model0826_source"
    done
else
    model0826_format_dir="/tmp/model_0826_xelatex_format"
    mkdir -p "$model0826_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0826_format_dir" \
        xelatex.ini

    for model0826_pass in 1 2; do
        xetex \
            -fmt="$model0826_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0826_output_dir" \
            -jobname="$model0826_jobname" \
            "$model0826_source"
    done
fi

cp "$model0826_output_dir/$model0826_jobname.pdf" "$model0826_script_dir/$model0826_jobname.pdf"
echo "Compiled: $model0826_script_dir/$model0826_jobname.pdf"
