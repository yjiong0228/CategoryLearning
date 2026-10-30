"""Assemble the academic poster from the supplied template and saved figures.

Run from the repository root. Existing figure pixels and scientific results are
read-only; PowerPoint cropping selects complete panels without resampling them.
Each build uses a new directory and exports an editable PPTX, PDF and PNG preview.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from PIL import Image
import fitz
import pptx
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "CategoryLearning_poster/CategoryLearning_poster.pptx"
FIGURES = {
    "task": ROOT / "CategoryLearning_codes/figures/outputs/fig1/journal_20260921_v2/Figure1_journal.png",
    "framework": ROOT / "CategoryLearning_codes/figures/outputs/fig2/journal_20260921_v2/Figure2_journal.png",
    "dynamics": ROOT / "CategoryLearning_codes/figures/outputs/fig3/journal_20260921_v3/Figure3.png",
}
SUBJECT_TABLE = FIGURES["framework"].parent / "subjects.csv"
INK, NAVY, TEAL = "233544", "183D57", "197C83"
MUTED, LINE, PALE, AMBER = "596671", "D6E0E5", "EFF5F7", "A9702A"
FONT = "Arial"


class Poster:
    def __init__(self) -> None:
        self.prs = Presentation(TEMPLATE)
        self.slide = self.prs.slides[0]
        self.panels: list[dict] = []
        self.textboxes: list[dict] = []
        self.width = self.prs.slide_width / 914400
        self.height = self.prs.slide_height / 914400

    def box(self, x: float, y: float, w: float, h: float,
            fill: str = "FFFFFF", line: str | None = None):
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        if line:
            shape.line.color.rgb = RGBColor.from_string(line)
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    def text(self, value: str, x: float, y: float, w: float, h: float,
             size: float = 29, color: str = INK, bold: bool = False,
             align=PP_ALIGN.LEFT):
        shape = self.slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.name = f"Text {len(self.textboxes) + 1}: {value[:48]}"
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.TOP
        for i, line in enumerate(value.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.alignment = align
            p.space_before = Pt(0)
            p.space_after = Pt(3)
            p.line_spacing = 1.08
            p.font.name = FONT
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
        self.textboxes.append({
            "name": shape.name, "text": value, "font_pt": size,
            "bounds_in": [x, y, w, h],
        })
        return shape

    def section(self, number: str, title: str, x: float, y: float,
                w: float, h: float) -> None:
        self.box(x, y + .34, w, h - .34, line=LINE)
        self.box(x, y, w, .82, NAVY)
        self.text(number, x + .25, y + .13, .75, .55, 31, "A8DCE2", True)
        self.text(title, x + 1.1, y + .13, w - 1.3, .59, 33, "FFFFFF", True)

    def picture(self, key: str, crop: tuple[float, float, float, float],
                x: float, y: float, w: float, label: str) -> float:
        """Place an unaltered source with native PPT crop, preserving aspect."""
        path = FIGURES[key]
        with Image.open(path) as im:
            iw, ih = im.size
        left, top, right, bottom = crop
        h = w * ih * (bottom - top) / (iw * (right - left))
        pic = self.slide.shapes.add_picture(
            str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h)
        )
        pic.crop_left, pic.crop_top = left, top
        pic.crop_right, pic.crop_bottom = 1 - right, 1 - bottom
        pic.name = label
        self.panels.append({"name": label, "source": str(path.relative_to(ROOT)),
                            "crop_ltrb": crop, "bounds_in": [x, y, w, h]})
        return h

    def assemble(self, second_affiliation: str, email: str | None,
                 draft_version: int = 1) -> None:
        x1, x2, cw = .85, 17.0, 15.25
        # The supplied slide layout retains the conference banner and sponsors.
        self.text("Beyond behavioural fit", 1, 3.25, 31.1, 1.6, 84, NAVY, True,
                  PP_ALIGN.CENTER)
        self.text("Reconstructing belief dynamics in human category learning",
                  1, 4.9, 31.1, .9, 49, NAVY, True, PP_ALIGN.CENTER)
        self.text("Jiong Yang¹, Junqi Wang², Ran Zhu¹, Jiayu Zhan¹",
                  1, 5.87, 31.1, .65, 35, INK, False, PP_ALIGN.CENTER)
        self.text("¹ School of Psychological and Cognitive Sciences, Peking University",
                  1, 6.56, 31.1, .55, 28, MUTED, False, PP_ALIGN.CENTER)
        self.text(f"² {second_affiliation}", 1, 7.12, 31.1, .53, 26, MUTED,
                  False, PP_ALIGN.CENTER)
        self.box(.85, 7.95, 31.4, .84, PALE)
        self.text("Which rules are considered, believed and used as a person learns?",
                  1.05, 8.1, 31.0, .59, 34, TEAL, True, PP_ALIGN.CENTER)

        self.section("01", "Task and individual learning", x1, 9.22, cw, 10.25)
        self.text("96 learners  /  3 tasks  /  4 stimulus features",
                  x1 + .35, 10.3, 14.55, .62, 32, TEAL, True)
        self.text("Learners infer category rules from choices and feedback.",
                  x1 + .35, 11.02, 14.55, .65, 28)
        self.picture("task", (.015, .012, .99, .357),
                     x1 + .28, 11.96, 14.65, "Figure 1a-b: tasks and trial sequence")
        self.text("Two or four categories; Task 2 includes partial feedback.\n"
                  "Reports describe the current choice, before feedback.",
                  x1 + .35, 18.12, 14.55, 1.1, 24, MUTED)

        self.section("02", "A model of learning processes", x1, 20.02, cw, 10.15)
        self.text("Search proposes rules; feedback updates beliefs; beliefs guide choice.\n"
                  "Fitting choices yields a distribution of possible internal trajectories.",
                  x1 + .35, 21.14, 14.55, 1.12, 27)
        self.picture("framework", (.035, .013, .985, .345),
                     x1 + .29, 22.48, 14.65, "Figure 2a: illustrative learning cycle")
        self.box(x1 + .35, 29.1, 7.0, .65, PALE)
        self.box(x1 + 7.6, 29.1, 7.3, .65, "EEF6F3")
        self.text("Resources: capacity, memory, perception",
                  x1 + .49, 29.23, 6.75, .4, 19.5, NAVY, True)
        self.text("Biases: search preferences, rule persistence",
                  x1 + 7.75, 29.23, 7.0, .4, 19.5, TEAL, True)

        self.section("03", "Choice fit and the next comparison", x1, 30.72, cw, 10.42)
        self.text("Existing fits track observed choice accuracy.",
                  x1 + .35, 31.84, 14.55, .68, 31, TEAL, True)
        self.picture("framework", (.018, .804, .345, .988),
                     x1 + .3, 33.0, 7.25, "Figure 2e: choice calibration")
        self.text("12 fitted learners\n7,924 scored choices",
                  x1 + 8.0, 33.2, 6.85, 1.45, 31, NAVY, True)
        self.text("Calibration compares predicted and observed accuracy.\n"
                  "Each small point is one participant in a probability bin; "
                  "the black line pools trials.",
                  x1 + 8.0, 35.0, 6.7, 2.8, 25)
        self.text("Full-record fits; displayed scores describe the fitted records.",
                  x1 + .35, 38.28, 14.55, .65, 24, MUTED)
        self.box(x1 + .35, 39.15, 14.55, 1.7, "F9F3E9")
        self.text("MODEL COMPARISON  ·  TO ADD",
                  x1 + .6, 39.34, 14.0, .5, 24, AMBER, True)
        self.text("Bayesian and alternative neural-network models: matched data,\n"
                  "evaluation splits and choice scores.",
                  x1 + .6, 39.83, 14.0, .95, 23, INK)

        self.section("04", "Reconstructing reported strategies", x2, 9.22, cw, 17.12)
        self.text("Choice-derived beliefs show preliminary correspondence\n"
                  "with verbal reports that were excluded from fitting.",
                  x2 + .35, 10.32, 14.55, 1.12, 29, TEAL, True)
        self.picture("framework", (.018, .366, .995, .792),
                     x2 + .28, 11.7, 14.65, "Figure 2b-d: all three task examples")
        self.picture("framework", (.349, .804, .994, .988),
                     x2 + 1.35, 20.0, 12.6, "Figure 2f-g: all 12 participants")
        self.text("Dynamic vs time-averaged beliefs: higher compatibility in 11/12 learners.\n"
                  "7,600 reports; comparison uses pre-choice beliefs and report content.\n"
                  "Development results; oral coding awaits final manual review.",
                  x2 + .35, 24.8, 14.55, 1.35, 22, MUTED)

        self.section("05", "Different paths to better performance", x2, 26.89, cw, 14.25)
        self.text("Rule availability, belief and use can evolve at different times.",
                  x2 + .35, 28.03, 14.55, 1.1, 30, TEAL, True)
        self.picture("dynamics", (.009, .0, .991, .567),
                     x2 + .45, 29.3, 14.3, "Figure 3a-c: earlier, gradual and abrupt improvement")
        self.text("Earlier improvement: target-rule support rises early.\n"
                  "Gradual improvement: support fluctuates across many trials.\n"
                  "Abrupt improvement: useful belief can precede reliable use.",
                  x2 + .35, 38.2, 14.55, 1.65, 26)
        self.text("Illustrative full-record reconstructions at fixed fitted parameters.\n"
                  "Bands show numerical replay variation. S122 has weak oral agreement.",
                  x2 + .35, 39.98, 14.55, .95, 22, MUTED)

        self.box(.85, 41.82, 31.4, 2.55, NAVY)
        self.text("TAKE-HOME MESSAGE", 1.25, 42.12, 5.8, .55, 27, "A8DCE2", True)
        self.text("Learning differs in when rules are found, retained and used.",
                  7.35, 42.04, 24.1, .72, 39, "FFFFFF", True)
        self.text("Reconstructing these processes offers testable explanations of individual learning.\n"
                  "The proposed link to “aha” moments requires stronger state recovery and independent evidence.",
                  7.35, 43.02, 24.0, 1.0, 25, "FFFFFF")
        footer = (f"DRAFT v{draft_version}  ·  30 October 2026  |  "
                  "Figures: earlier Model 0826 development results")
        if email:
            footer += f"  |  {email}"
        self.text(footer, .95, 44.73, 31.2, .46, 19, MUTED, False, PP_ALIGN.CENTER)
        self.slide.notes_slide.notes_text_frame.text = (
            "Poster draft based on learning_paper_v1.tex abstract. All charts are "
            "existing September 2026 figures, sourced from Model 0826. They do not "
            "establish Model 1024 validation. Figures use full-record fitted parameters. "
            "Oral reports were excluded from these fits but remain development material "
            "pending manual review. Alternative-model comparisons and the awareness/aha "
            "mechanism are unconfirmed. Figure 2a contains illustrative numbers. "
            "Native picture cropping preserves source pixels. Full provenance is in manifest.json."
        )

    def validate_geometry(self) -> None:
        for shape in self.slide.shapes:
            x, y = shape.left / 914400, shape.top / 914400
            w, h = shape.width / 914400, shape.height / 914400
            if x < 0 or y < 3.1 or x + w > self.width or y + h > 45.45:
                raise ValueError(f"Content outside template's available area: {shape.name}")
        for panel in self.panels:
            x, y, w, h = panel["bounds_in"]
            for text in self.textboxes:
                tx, ty, tw, th = text["bounds_in"]
                overlap_w = min(x + w, tx + tw) - max(x, tx)
                overlap_h = min(y + h, ty + th) - max(y, ty)
                if overlap_w > .01 and overlap_h > .01:
                    raise ValueError(f"Picture/text overlap: {panel['name']} / {text['name']}")


def verify_counts() -> dict:
    with SUBJECT_TABLE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {"participants": len(rows),
              "reports": sum(int(r["n_reports"]) for r in rows),
              "scored_choices": sum(int(r["n_scored"]) for r in rows),
              "dynamic_above_static": sum(float(r["compatibility"]) >
                                          float(r["static_compatibility"]) for r in rows)}
    if counts != {"participants": 12, "reports": 7600, "scored_choices": 7924,
                  "dynamic_above_static": 11}:
        raise ValueError(f"Source counts changed; review the poster text: {counts}")
    return counts


def export_pdf(pptx_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="categorylearning-poster-lo-") as profile:
        subprocess.run(
            ["libreoffice", f"-env:UserInstallation={Path(profile).as_uri()}",
             "--headless", "--convert-to", "pdf", "--outdir", str(pptx_path.parent),
             str(pptx_path)],
            env={**os.environ, "DCONF_PROFILE": "/dev/null", "SAL_USE_VCLPLUGIN": "svp"},
            check=True, timeout=120, capture_output=True, text=True,
        )


def verify_render(poster: Poster, pdf_path: Path) -> dict:
    """Check text survival and overflow against each editable PPT text box."""
    doc = fitz.open(pdf_path)
    if len(doc) != 1:
        raise ValueError(f"Expected one poster page, found {len(doc)}")
    page = doc[0]
    normalize = lambda s: "".join(s.split()).replace("’", "'")
    full_text = normalize(page.get_text())
    missing = [t["name"] for t in poster.textboxes
               if normalize(t["text"]) not in full_text]
    overflow = []
    # Follow the complete text sequence rather than selecting spans by origin:
    # overflowing final lines can have origins beyond the declared text box.
    spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
             for line in block["lines"] for span in line["spans"]]
    for t in poster.textboxes:
        x, y, w, h = [v * 72 for v in t["bounds_in"]]
        target = normalize(t["text"])
        own = []
        for start, span in enumerate(spans):
            if abs(span["size"] - t["font_pt"]) >= 1:
                continue
            joined, matched = "", []
            for candidate in spans[start:]:
                joined += normalize(candidate["text"])
                matched.append(candidate)
                if joined == target:
                    own = matched
                    break
                if not target.startswith(joined):
                    break
            if own:
                break
        if not own:
            missing.append(t["name"] + " (span matching)")
        if any(s["bbox"][0] < x - 3 or s["bbox"][1] < y - 3 or
               s["bbox"][2] > x + w + 3 or s["bbox"][3] > y + h + 3 for s in own):
            overflow.append(t["name"])
    page.get_pixmap(matrix=fitz.Matrix(2400 / page.rect.width, 2400 / page.rect.width),
                    alpha=False).save(pdf_path.with_name("poster_preview.png"))
    result = {"page_count": len(doc), "page_size_pt": list(page.rect)[2:],
              "text_boxes": len(poster.textboxes), "missing_text": missing,
              "potential_text_overflow": overflow}
    if missing or overflow:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise ValueError("Inspect the PDF: missing or overflowing text")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--second-affiliation", default="Affiliation to be confirmed (Junqi Wang)")
    parser.add_argument("--email")
    parser.add_argument("--draft-version", type=int, default=1)
    parser.add_argument("--pptx-only", action="store_true",
                        help="Skip LibreOffice conversion and PDF validation")
    args = parser.parse_args()
    if args.draft_version < 1:
        parser.error("--draft-version must be a positive integer")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite an existing output directory: {out}")
    poster = Poster()
    counts = verify_counts()
    poster.assemble(args.second_affiliation, args.email, args.draft_version)
    poster.validate_geometry()
    out.mkdir(parents=True)
    path = out / f"CategoryLearning_poster_draft_v{args.draft_version}.pptx"
    poster.prs.save(path)
    sources = [TEMPLATE, *FIGURES.values(), SUBJECT_TABLE, Path(__file__).resolve(),
               ROOT / "CategoryLearning_paper/learning_paper_v1/learning_paper_v1.tex"]
    manifest = {
        "draft_date": "2026-10-30", "draft_version": args.draft_version,
        "page_size_in": [poster.width, poster.height],
        "python": platform.python_version(), "python_pptx": pptx.__version__,
        "pymupdf": fitz.__version__, "requested_font": FONT,
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "second_affiliation": args.second_affiliation, "email": args.email,
        "verified_source_counts": counts,
        "panels": poster.panels, "text_boxes": poster.textboxes,
        "scientific_scope": "Existing Model 0826 development figures; no new analysis or fitting",
    }
    if not args.pptx_only:
        export_pdf(path)
        manifest["validation"] = verify_render(poster, path.with_suffix(".pdf"))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
