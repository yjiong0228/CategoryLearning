#!/usr/bin/env bash
set -euo pipefail

model0809_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0809_script_dir"

model0809_jobname="model_0809"
model0809_output_dir="/tmp/model_0809_build"
mkdir -p "$model0809_output_dir"
model0809_source="$model0809_output_dir/model_0809_build.tex"

# Keep model_0809.tex embeddable in the manuscript.  The standalone wrapper is
# generated only in the temporary proof-build directory.
printf '%s\n' \
    '% !TeX program = xelatex' \
    '\documentclass[12pt]{article}' \
    '\usepackage[a4paper,margin=2.5cm]{geometry}' \
    '\usepackage{amsmath,amssymb,bm}' \
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
    '\input{model_0809.tex}' \
    '\end{document}' \
    > "$model0809_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0809_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0809_output_dir" \
            -jobname="$model0809_jobname" \
            "$model0809_source"
    done
else
    model0809_format_dir="/tmp/model_0809_xelatex_format"
    mkdir -p "$model0809_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0809_format_dir" \
        xelatex.ini

    for model0809_pass in 1 2; do
        xetex \
            -fmt="$model0809_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0809_output_dir" \
            -jobname="$model0809_jobname" \
            "$model0809_source"
    done
fi

cp "$model0809_output_dir/$model0809_jobname.pdf" "$model0809_script_dir/$model0809_jobname.pdf"
echo "Compiled: $model0809_script_dir/$model0809_jobname.pdf"
