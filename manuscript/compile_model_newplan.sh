#!/usr/bin/env bash
set -euo pipefail

modelplan_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$modelplan_script_dir"

modelplan_jobname="model_newplan"
modelplan_output_dir="/tmp/model_newplan_build"
mkdir -p "$modelplan_output_dir"
modelplan_source="$modelplan_output_dir/model_newplan_build.tex"

# Keep model_newplan.tex as an embeddable manuscript section. For a standalone
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
    '\input{model_newplan.tex}' \
    '\end{document}' \
    > "$modelplan_source"

if command -v xelatex >/dev/null 2>&1; then
    for modelplan_pass in 1 2; do
        xelatex \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelplan_output_dir" \
            -jobname="$modelplan_jobname" \
            "$modelplan_source"
    done
else
    modelplan_format_dir="/tmp/model_newplan_xelatex_format"
    mkdir -p "$modelplan_format_dir"

    xetex \
        -ini \
        -etex \
        -jobname=xelatex \
        -output-directory="$modelplan_format_dir" \
        xelatex.ini

    for modelplan_pass in 1 2; do
        xetex \
            -fmt="$modelplan_format_dir/xelatex.fmt" \
            -interaction=nonstopmode \
            -halt-on-error \
            -file-line-error \
            -output-directory="$modelplan_output_dir" \
            -jobname="$modelplan_jobname" \
            "$modelplan_source"
    done
fi

cp "$modelplan_output_dir/$modelplan_jobname.pdf" "$modelplan_script_dir/$modelplan_jobname.pdf"
echo "Compiled: $modelplan_script_dir/$modelplan_jobname.pdf"
