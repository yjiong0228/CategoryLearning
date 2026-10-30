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
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

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
    "task": ROOT / "CategoryLearning_codes/figures/outputs/fig1/poster_arial_20261030_v1/Figure1_journal.png",
    "framework": ROOT / "CategoryLearning_codes/figures/outputs/fig2/poster_arial_20261030_v1/Figure2_journal.png",
    "dynamics": ROOT / "CategoryLearning_codes/figures/outputs/fig3/poster_arial_20261030_v1/Figure3.png",
}
SUBJECT_TABLE = FIGURES["framework"].parent / "subjects.csv"
REGULAR_FONT = ROOT / "src/Arial.ttf"
BOLD_FONT = ROOT / "src/Arial Bold.ttf"
TEMPLATE_FONT = ROOT / "CategoryLearning_poster/fonts/Arial Black.ttf"
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
             align=PP_ALIGN.LEFT, lead: str | None = None):
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
            if lead and line.startswith(lead):
                p.clear()
                run = p.add_run()
                run.text = lead
                run.font.bold = True
                p.add_run().text = line[len(lead):]
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
                 draft_version: int = 2) -> None:
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
        self.text("01  Background and Motivation", 1.05, 7.95, 31, .72,
                  35, NAVY, True, PP_ALIGN.CENTER)
        background = [
            ("Learning differences:", " People learning the same categories can improve early, gradually, or abruptly."),
            ("Hidden processes:", " Similar choices can arise from different rules and beliefs, making internal states difficult to infer."),
            ("Our approach:", " Reconstruct rule search, belief updating and rule use from choices, then compare them with verbal reports."),
        ]
        for i, (lead, body) in enumerate(background):
            self.text(lead + body, 1.05, 8.76 + .75 * i, 31, .65, 27, lead=lead)

        self.section("02", "Task and individual learning", x1, 11.55, cw, 8.45)
        self.text("96 learners  /  3 tasks  /  4 stimulus features",
                  x1 + .35, 12.58, 14.55, .62, 29, TEAL, True)
        self.picture("task", (.015, .012, .99, .357),
                     x1 + 1.12, 13.4, 13.0, "Figure 1a-b: tasks and trial sequence")
        self.text("Two or four categories; Task 2 includes partial feedback.\n"
                  "Reports describe the current choice, before feedback.",
                  x1 + .35, 19.0, 14.55, .85, 23, MUTED)

        self.section("03", "A model of learning processes", x1, 20.55, cw, 9.7)
        self.text("Search proposes rules; feedback updates beliefs; beliefs guide choice.\n"
                  "Fitting choices yields a distribution of possible internal trajectories.",
                  x1 + .35, 21.65, 14.55, 1.12, 27)
        self.picture("framework", (.035, .013, .985, .345),
                     x1 + .77, 22.95, 13.7, "Figure 2a: illustrative learning cycle")
        self.box(x1 + .35, 29.2, 7.0, .65, PALE)
        self.box(x1 + 7.6, 29.2, 7.3, .65, "EEF6F3")
        self.text("Resources: capacity, memory, perception",
                  x1 + .49, 29.33, 6.75, .4, 19.5, NAVY, True)
        self.text("Biases: search preferences, rule persistence",
                  x1 + 7.75, 29.33, 7.0, .4, 19.5, TEAL, True)

        self.section("04", "Choice fit and the next comparison", x1, 30.8, cw, 10.34)
        self.text("Existing fits track observed choice accuracy.",
                  x1 + .35, 31.9, 14.55, .68, 31, TEAL, True)
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

        self.section("05", "Reconstructing reported strategies", x2, 11.55, cw, 16.42)
        self.text("Choice-derived beliefs show preliminary correspondence\n"
                  "with verbal reports that were excluded from fitting.",
                  x2 + .35, 12.68, 14.55, 1.12, 29, TEAL, True)
        self.picture("framework", (.018, .366, .995, .792),
                     x2 + .77, 14.0, 13.7, "Figure 2b-d: all three task examples")
        self.picture("framework", (.349, .804, .994, .988),
                     x2 + 1.58, 21.68, 12.1, "Figure 2f-g: all 12 participants")
        self.text("Dynamic vs time-averaged beliefs: higher compatibility in 11/12 learners.\n"
                  "7,600 reports; comparison uses pre-choice beliefs and report content.\n"
                  "Development results; oral coding awaits final manual review.",
                  x2 + .35, 26.36, 14.55, 1.25, 22, MUTED)

        self.section("06", "Different paths to better performance", x2, 28.52, cw, 12.62)
        self.text("Rule availability, belief and use can evolve at different times.",
                  x2 + .35, 29.56, 14.55, .85, 29, TEAL, True)
        self.picture("dynamics", (.009, .0, .991, .567),
                     x2 + 1.4, 30.7, 12.45, "Figure 3a-c: earlier, gradual and abrupt improvement")
        self.text("Earlier improvement: target-rule support rises early.\n"
                  "Gradual improvement: support fluctuates across many trials.\n"
                  "Abrupt improvement: useful belief can precede reliable use.",
                  x2 + .35, 38.6, 14.55, 1.45, 24)
        self.text("Illustrative full-record reconstructions at fixed fitted parameters.\n"
                  "Bands show numerical replay variation. S122 has weak oral agreement.",
                  x2 + .35, 40.15, 14.55, .8, 21.5, MUTED)

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


def preserve_template_parts(pptx_path: Path) -> dict:
    """Keep the template's masters, layouts, themes and media byte-for-byte."""
    prefixes = ("ppt/slideLayouts/", "ppt/slideMasters/", "ppt/theme/", "ppt/media/")
    with ZipFile(TEMPLATE) as source, ZipFile(pptx_path) as rendered:
        protected = {name: source.read(name) for name in source.namelist()
                     if name.startswith(prefixes) and not name.endswith("/")}
        items = {name: rendered.read(name) for name in rendered.namelist()}
    items.update(protected)
    with ZipFile(pptx_path, "w", ZIP_DEFLATED) as archive:
        for name, data in items.items():
            archive.writestr(name, data)
    with ZipFile(pptx_path) as archive:
        assert all(archive.read(name) == data for name, data in protected.items())
    return {"protected_parts": len(protected), "changed_parts": [],
            "original_theme_heading_font": "Arial Black"}


def export_pdf(pptx_path: Path, template_font: Path) -> dict:
    """Render with the supplied files; compare against the untouched template."""
    with tempfile.TemporaryDirectory(prefix="categorylearning-poster-lo-") as temporary:
        work = Path(temporary)
        fonts = work / "fonts"
        fonts.mkdir()
        files = {"Arial:style=Regular": REGULAR_FONT,
                 "Arial:style=Bold": BOLD_FONT, "Arial Black": template_font}
        for index, source in enumerate(files.values()):
            shutil.copy2(source, fonts / f"font{index}.ttf")
        config = work / "fonts.conf"
        config.write_text(
            '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
            '<fontconfig><include>/etc/fonts/fonts.conf</include>'
            f'<dir>{escape(str(fonts))}</dir><cachedir>{escape(str(work / "cache"))}</cachedir>'
            '</fontconfig>'
        )
        env = {**os.environ, "FONTCONFIG_FILE": str(config), "DCONF_PROFILE": "/dev/null",
               "SAL_USE_VCLPLUGIN": "svp"}
        matches = {}
        for pattern, source in files.items():
            matched = Path(subprocess.check_output(
                ["fc-match", "-f", "%{file}", pattern], env=env, text=True))
            if matched.read_bytes() != source.read_bytes():
                raise ValueError(f"Unexpected font substitution for {pattern}: {matched}")
            matches[pattern] = {"file_name": source.name,
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        template_copy = work / "original_template.pptx"
        shutil.copy2(TEMPLATE, template_copy)
        render_dir = work / "render"
        render_dir.mkdir()
        subprocess.run(
            ["libreoffice", f"-env:UserInstallation={(work / 'profile').as_uri()}",
             "--headless", "--convert-to", "pdf", "--outdir", str(render_dir),
             str(pptx_path), str(template_copy)], env=env,
            check=True, timeout=120, capture_output=True, text=True,
        )
        pdf = render_dir / pptx_path.with_suffix(".pdf").name
        original = fitz.open(render_dir / "original_template.pdf")
        final = fitz.open(pdf)
        checks = {}
        for label, low, high in [("header", 0, 3.1), ("footer", 45.45, 46.8104167)]:
            rect = fitz.Rect(0, low * 72, final[0].rect.width, high * 72)
            a = original[0].get_pixmap(matrix=fitz.Matrix(1, 1), clip=rect, alpha=False)
            b = final[0].get_pixmap(matrix=fitz.Matrix(1, 1), clip=rect, alpha=False)
            checks[label] = a.samples == b.samples
            if not checks[label]:
                raise ValueError(f"The rendered template {label} changed")
        shutil.copy2(pdf, pptx_path.with_suffix(".pdf"))
        return {"font_files": matches, "template_pixel_match": checks,
                "renderer": subprocess.check_output(["libreoffice", "--version"],
                                                     text=True).strip()}


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
    substituted_fonts = []
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
        for span in own:
            if span["font"] not in {"ArialMT", "Arial-BoldMT", "Arial", "Arial-Bold"}:
                substituted_fonts.append({"text_box": t["name"], "font": span["font"]})
        if any(s["bbox"][0] < x - 3 or s["bbox"][1] < y - 3 or
               s["bbox"][2] > x + w + 3 or s["bbox"][3] > y + h + 3 for s in own):
            overflow.append(t["name"])
    page.get_pixmap(matrix=fitz.Matrix(2400 / page.rect.width, 2400 / page.rect.width),
                    alpha=False).save(pdf_path.with_name("poster_preview.png"))
    result = {"page_count": len(doc), "page_size_pt": list(page.rect)[2:],
              "text_boxes": len(poster.textboxes), "missing_text": missing,
              "potential_text_overflow": overflow, "substituted_body_fonts": substituted_fonts}
    if missing or overflow or substituted_fonts:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise ValueError("Inspect the PDF: missing or overflowing text")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--second-affiliation", default="Affiliation to be confirmed (Junqi Wang)")
    parser.add_argument("--email")
    parser.add_argument("--draft-version", type=int, default=2)
    parser.add_argument("--template-font", type=Path, default=TEMPLATE_FONT,
                        help="Arial Black file used by the untouched template")
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
    template_preservation = preserve_template_parts(path)
    sources = [TEMPLATE, *FIGURES.values(), SUBJECT_TABLE, REGULAR_FONT, BOLD_FONT,
               Path(__file__).resolve(),
               ROOT / "CategoryLearning_paper/learning_paper_v1/learning_paper_v1.tex"]
    manifest = {
        "draft_date": "2026-10-30", "draft_version": args.draft_version,
        "page_size_in": [poster.width, poster.height],
        "python": platform.python_version(), "python_pptx": pptx.__version__,
        "pymupdf": fitz.__version__, "requested_font": FONT,
        "template_preservation": template_preservation,
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "second_affiliation": args.second_affiliation, "email": args.email,
        "verified_source_counts": counts,
        "panels": poster.panels, "text_boxes": poster.textboxes,
        "scientific_scope": "Existing Model 0826 development figures; no new analysis or fitting",
    }
    if not args.pptx_only:
        manifest["pdf_export"] = export_pdf(path, args.template_font.resolve())
        manifest["validation"] = verify_render(poster, path.with_suffix(".pdf"))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
