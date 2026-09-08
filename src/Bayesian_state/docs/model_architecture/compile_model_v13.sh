#!/usr/bin/env bash
set -euo pipefail

modelv13_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$modelv13_script_dir"

modelv13_jobname="model_v13"
modelv13_output_dir="/tmp/model_v13_build"
mkdir -p "$modelv13_output_dir"
modelv13_source="$modelv13_output_dir/model_v13_build.tex"

# Keep model_v13.tex as an embeddable manuscript section.  For a standalone
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
    '\input{model_v13.tex}' \
    '\end{document}' \
    > "$modelv13_source"

if command -v xelatex >/dev/null 2>&1; then
    for modelv13_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv13_output_dir" \
            -jobname="$modelv13_jobname" \
            "$modelv13_source"
    done
else
    modelv13_format_dir="/tmp/model_v13_xelatex_format"
    mkdir -p "$modelv13_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$modelv13_format_dir" \
        xelatex.ini

    for modelv13_pass in 1 2; do
        xetex \
            -fmt="$modelv13_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelv13_output_dir" \
            -jobname="$modelv13_jobname" \
            "$modelv13_source"
    done
fi

cp "$modelv13_output_dir/$modelv13_jobname.pdf" "$modelv13_script_dir/$modelv13_jobname.pdf"
echo "Compiled: $modelv13_script_dir/$modelv13_jobname.pdf"
