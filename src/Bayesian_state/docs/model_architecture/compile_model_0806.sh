#!/usr/bin/env bash
set -euo pipefail

model0806_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0806_script_dir"

model0806_jobname="model_0806"
model0806_output_dir="/tmp/model_0806_build"
mkdir -p "$model0806_output_dir"
model0806_source="$model0806_output_dir/model_0806_build.tex"

# Keep model_0806.tex as an embeddable manuscript section. For a standalone
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
    '\input{model_0806.tex}' \
    '\end{document}' \
    > "$model0806_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0806_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0806_output_dir" \
            -jobname="$model0806_jobname" \
            "$model0806_source"
    done
else
    model0806_format_dir="/tmp/model_0806_xelatex_format"
    mkdir -p "$model0806_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0806_format_dir" \
        xelatex.ini

    for model0806_pass in 1 2; do
        xetex \
            -fmt="$model0806_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0806_output_dir" \
            -jobname="$model0806_jobname" \
            "$model0806_source"
    done
fi

cp "$model0806_output_dir/$model0806_jobname.pdf" "$model0806_script_dir/$model0806_jobname.pdf"
echo "Compiled: $model0806_script_dir/$model0806_jobname.pdf"
