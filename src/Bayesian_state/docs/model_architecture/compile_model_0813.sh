#!/usr/bin/env bash
set -euo pipefail

model0813_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0813_script_dir"

model0813_jobname="model_0813"
model0813_output_dir="/tmp/model_0813_build"
mkdir -p "$model0813_output_dir"
model0813_source="$model0813_output_dir/model_0813_build.tex"

# Keep model_0813.tex embeddable in the manuscript.  The standalone wrapper is
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
    '\input{model_0813.tex}' \
    '\end{document}' \
    > "$model0813_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0813_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0813_output_dir" \
            -jobname="$model0813_jobname" \
            "$model0813_source"
    done
else
    model0813_format_dir="/tmp/model_0813_xelatex_format"
    mkdir -p "$model0813_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0813_format_dir" \
        xelatex.ini

    for model0813_pass in 1 2; do
        xetex \
            -fmt="$model0813_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0813_output_dir" \
            -jobname="$model0813_jobname" \
            "$model0813_source"
    done
fi

cp "$model0813_output_dir/$model0813_jobname.pdf" "$model0813_script_dir/$model0813_jobname.pdf"
echo "Compiled: $model0813_script_dir/$model0813_jobname.pdf"
