#!/usr/bin/env bash
set -euo pipefail

model0804_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$model0804_script_dir"

model0804_jobname="model_0804"
model0804_output_dir="/tmp/model_0804_build"
mkdir -p "$model0804_output_dir"
model0804_source="$model0804_output_dir/model_0804_build.tex"

# Keep model_0804.tex as an embeddable manuscript section.  For a standalone
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
    '\input{model_0804.tex}' \
    '\end{document}' \
    > "$model0804_source"

if command -v xelatex >/dev/null 2>&1; then
    for model0804_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0804_output_dir" \
            -jobname="$model0804_jobname" \
            "$model0804_source"
    done
else
    model0804_format_dir="/tmp/model_0804_xelatex_format"
    mkdir -p "$model0804_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$model0804_format_dir" \
        xelatex.ini

    for model0804_pass in 1 2; do
        xetex \
            -fmt="$model0804_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$model0804_output_dir" \
            -jobname="$model0804_jobname" \
            "$model0804_source"
    done
fi

cp "$model0804_output_dir/$model0804_jobname.pdf" "$model0804_script_dir/$model0804_jobname.pdf"
echo "Compiled: $model0804_script_dir/$model0804_jobname.pdf"
