#!/usr/bin/env bash
set -euo pipefail

model_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model_script_dir"

model_jobname="model"
model_output_dir="/tmp/model_build"
mkdir -p "$model_output_dir"
model_source="$model_output_dir/model_build.tex"

# Keep model.tex as an embeddable manuscript section.  For a standalone
# proof PDF, create the document wrapper only inside the temporary build tree.
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
    '\input{model.tex}' \
    '\end{document}' \
    > "$model_source"

if command -v xelatex >/dev/null 2>&1; then
    for model_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model_output_dir" \
            -jobname="$model_jobname" \
            "$model_source"
    done
else
    model_format_dir="/tmp/model_xelatex_format"
    mkdir -p "$model_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model_format_dir" \
        xelatex.ini

    for model_pass in 1 2; do
        xetex \
            -fmt="$model_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model_output_dir" \
            -jobname="$model_jobname" \
            "$model_source"
    done
fi

cp "$model_output_dir/$model_jobname.pdf" "$model_script_dir/$model_jobname.pdf"
echo "Compiled: $model_script_dir/$model_jobname.pdf"
