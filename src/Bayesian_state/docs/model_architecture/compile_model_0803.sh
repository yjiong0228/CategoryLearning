#!/usr/bin/env bash
set -euo pipefail

model0803_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0803_script_dir"

model0803_jobname="model_0803"
model0803_output_dir="/tmp/model_0803_build"
mkdir -p "$model0803_output_dir"
model0803_source="$model0803_output_dir/model_0803_build.tex"

# Keep model_0803.tex as an embeddable manuscript section.  For a standalone
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
    '\input{model_0803.tex}' \
    '\end{document}' \
    > "$model0803_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0803_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0803_output_dir" \
            -jobname="$model0803_jobname" \
            "$model0803_source"
    done
else
    model0803_format_dir="/tmp/model_0803_xelatex_format"
    mkdir -p "$model0803_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0803_format_dir" \
        xelatex.ini

    for model0803_pass in 1 2; do
        xetex \
            -fmt="$model0803_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0803_output_dir" \
            -jobname="$model0803_jobname" \
            "$model0803_source"
    done
fi

cp "$model0803_output_dir/$model0803_jobname.pdf" "$model0803_script_dir/$model0803_jobname.pdf"
echo "Compiled: $model0803_script_dir/$model0803_jobname.pdf"
