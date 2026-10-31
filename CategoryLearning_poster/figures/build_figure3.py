"""Draw the Model 1023 learning loop as individually editable PowerPoint objects.

Figure contract: one normalized rule-belief distribution connects choice,
feedback learning, local search and capacity replacement. This is a mechanism
schematic, not an empirical result. Icons and bar lengths are illustrative.
Python generates native text, shapes and connectors on a 15.6 x 6.85 inch canvas.
The standalone figure and poster call the same drawing function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from zipfile import ZipFile

import fitz
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.slide import Slide
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 15.6, 6.85
SPEC = ROOT / "src/Bayesian_state/docs/model_architecture/model_1023.tex"
SOURCES = [SPEC, *[
    ROOT / "src/Bayesian_state/model/modules/hypothesis_transition" / name
    for name in ("compact_belief.py", "belief_outcome_retention.py",
                 "belief_error_entry.py", "entry_precision.py")
]]
INK, MUTED = "23333D", "586770"
BLUE, LIGHT_BLUE, TEAL = "306A91", "EDF4F8", "267E88"
ORANGE, LIGHT_ORANGE, LINE = "AC6A2D", "FBF3EA", "CAD5DD"
RULE_COLORS = ["306A91", "76A7BA", "BDD2DD"]


class NativeDrawing:
    """Draw in figure coordinates; retain each native object's audit bounds."""

    def __init__(self, slide: Slide, x: float, y: float, width: float):
        self.slide, self.x, self.y = slide, x, y
        self.scale = width / WIDTH
        self.elements: list[dict] = []
        self.textboxes: list[dict] = []

    def register(self, shape, label: str) -> None:
        for effect in shape._element.xpath(".//a:effectRef"):
            effect.set("idx", "0")
        shape.name = f"Figure3 {len(self.elements) + 1:03d}: {label}"
        self.elements.append({"name": shape.name, "kind": str(shape.shape_type),
                              "bounds_in": [v / 914400 for v in
                                            (shape.left, shape.top, shape.width, shape.height)]})

    def rect(self, x: float, y: float, w: float, h: float, *,
             fill: str, line: str | None = None, label: str = "rectangle",
             oval: bool = False):
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.OVAL if oval else MSO_SHAPE.RECTANGLE,
            Inches(self.x + x * self.scale), Inches(self.y + y * self.scale),
            Inches(w * self.scale), Inches(h * self.scale))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        if line:
            shape.line.color.rgb = RGBColor.from_string(line)
            shape.line.width = Pt(.8 * self.scale)
        else:
            shape.line.fill.background()
        # The default Office shape theme can add rasterized shadow effects.
        # An explicit empty effect list keeps the PDF as clean vector geometry.
        shape._element.spPr.append(OxmlElement("a:effectLst"))
        self.register(shape, label)
        return shape

    def text(self, value: str, x: float, y: float, w: float, h: float, *,
             size: float = 18, bold: bool = False, color: str = INK,
             center: bool = False) -> None:
        shape = self.slide.shapes.add_textbox(
            Inches(self.x + x * self.scale), Inches(self.y + y * self.scale),
            Inches(w * self.scale), Inches(h * self.scale))
        frame = shape.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_top = frame.margin_bottom = 0
        frame.margin_left = frame.margin_right = 0
        frame.vertical_anchor = MSO_ANCHOR.TOP
        for i, line in enumerate(value.split("\n")):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.text = line
            para.font.name = "Arial"
            para.font.size = Pt(size * self.scale)
            para.font.bold = bold
            para.font.color.rgb = RGBColor.from_string(color)
            para.space_before = para.space_after = Pt(0)
            para.line_spacing = 1.08
            para.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        self.register(shape, value[:55])
        self.textboxes.append({"name": shape.name, "text": value,
                               "font_pt": size * self.scale,
                               "bounds_in": self.elements[-1]["bounds_in"]})

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             color: str = MUTED, width: float = 1.6, arrow: bool = False,
             label: str = "connector") -> None:
        shape = self.slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(self.x + x1 * self.scale),
            Inches(self.y + y1 * self.scale), Inches(self.x + x2 * self.scale),
            Inches(self.y + y2 * self.scale))
        shape.line.color.rgb = RGBColor.from_string(color)
        shape.line.width = Pt(width * self.scale)
        if arrow:
            tip = OxmlElement("a:tailEnd")
            tip.set("type", "triangle")
            tip.set("w", "med")
            tip.set("len", "med")
            shape._element.spPr.get_or_add_ln().append(tip)
        self.register(shape, label)

    def card(self, x: float, y: float, w: float, h: float, title: str,
             learning: bool = False) -> None:
        accent = ORANGE if learning else BLUE
        self.rect(x, y, w, h, fill=LIGHT_ORANGE if learning else LIGHT_BLUE,
                  label=f"{title} background")
        self.rect(x, y, .045, h, fill=accent, label=f"{title} accent")
        self.text(title, x + .15, y + .12, w - .28, .42, size=21,
                  bold=True, color=accent)


def draw_figure3(slide: Slide, x: float = 0, y: float = 0,
                 width: float = WIDTH) -> NativeDrawing:
    """Add ungrouped native shapes; return metadata for slide/PDF validation."""
    d = NativeDrawing(slide, x, y, width)
    d.text("Model 1023: belief-guided learning", .2, .04, 12.0, .5,
           size=27, bold=True, color=BLUE)
    d.text("Mechanism schematic", 12.2, .15, 3.2, .33, size=16,
           color=MUTED, center=True)
    d.text("DECISION · current trial", .2, .62, 6.0, .34,
           size=18, bold=True, color=BLUE)
    d.text("Predictions use category distance and learned precision.",
           6.1, .65, 9.3, .32, size=16, color=MUTED)

    d.card(.2, 1.05, 2.55, 2.1, "Perception")
    d.card(3.1, 1.05, 4.35, 2.1, "Active rules")
    d.card(7.8, 1.05, 3.0, 2.1, "Choice")
    d.card(11.15, 1.05, 4.25, 2.1, "Feedback")
    for start, end in [(2.79, 3.05), (7.49, 7.75), (10.84, 11.10)]:
        d.line(start, 2.13, end, 2.13, arrow=True, color=BLUE,
               label="Decision flow")

    d.text("Four features", .35, 1.65, 2.25, .37, size=17.5)
    # Paired filled bars and open markers depict physical and perceived values.
    # Their lengths are illustrative, with no correspondence to participant data.
    for i, (height, error) in enumerate(zip((.38, .60, .46, .27), (.05, -.04, .07, -.03))):
        bx, base = .63 + .49 * i, 2.67
        d.rect(bx, base - height, .13, height, fill=RULE_COLORS[1],
               label=f"Feature {i + 1}: stimulus")
        d.rect(bx - .025, base - height - error - .04, .18, .08,
               fill="FFFFFF", line=BLUE, oval=True,
               label=f"Feature {i + 1}: internal percept")
    d.line(.46, 2.69, 2.5, 2.69, width=.8, color=LINE, label="Feature baseline")
    d.text("Shared noisy percept", .35, 2.79, 2.25, .34, size=17,
           center=True)

    d.text("Up to K", 6.11, 1.2, 1.18, .35, size=17, color=MUTED, center=True)
    d.text("Rule", 3.25, 1.66, .82, .33, size=16, bold=True)
    d.text("Belief", 4.4, 1.66, 1.35, .33, size=16, bold=True)
    d.text("Predict", 6.04, 1.66, 1.15, .33, size=16, bold=True)
    weights = (.60, .25, .15)
    probabilities = ((.70, .15, .10, .05), (.15, .40, .25, .20), (.10, .20, .25, .45))
    category_colors = ("306A91", "76A7BA", "A6C0CF", "D5E0E6")
    for i, (weight, prediction, color) in enumerate(zip(weights, probabilities, RULE_COLORS)):
        row = 2.05 + .33 * i
        # Three compact rule icons: threshold, comparison and conditional split.
        d.rect(3.27, row, .36, .25, fill="FFFFFF", line=LINE,
               label=f"Illustrative rule {i + 1}")
        if i == 0:
            d.line(3.45, row + .02, 3.45, row + .23, color=BLUE, width=1.3)
        elif i == 1:
            d.line(3.29, row + .23, 3.61, row + .02, color=BLUE, width=1.3)
        else:
            d.line(3.45, row + .02, 3.45, row + .23, color=BLUE, width=1.3)
            d.line(3.45, row + .13, 3.61, row + .13, color=BLUE, width=1.3)
        d.text(f"h{i + 1}", 3.71, row - .01, .55, .29, size=16)
        d.rect(4.4, row + .035, 1.35, .18, fill="DFE7EC", label="Belief track")
        d.rect(4.4, row + .035, 1.35 * weight, .18, fill=color,
               label=f"Rule {i + 1} belief")
        left = 6.05
        for category, (probability, category_color) in enumerate(zip(prediction, category_colors), 1):
            d.rect(left, row + .035, probability, .18, fill=category_color,
                   label=f"Rule {i + 1}: category {category} prediction")
            left += probability

    d.text("Belief-weighted mixture", 7.95, 1.65, 2.7, .37, size=17.5)
    for j in range(4):
        bx, bottom = 8.10 + .64 * j, 2.53
        for weight, prediction, color in zip(weights, probabilities, RULE_COLORS):
            contribution = weight * prediction[j]
            d.rect(bx, bottom - contribution, .35, contribution, fill=color,
                   label=f"Category {j + 1}: weighted rule contribution")
            bottom -= contribution
        d.text(f"C{j + 1}", bx - .05, 2.56, .45, .28, size=15.5, center=True)
    d.text("Sample the response", 7.96, 2.84, 2.68, .29, size=16.5, center=True)

    for i, value in enumerate(("0", "0.5", "1")):
        fx = 11.62 + 1.12 * i
        d.rect(fx, 1.83, .65, .65, fill="FFFFFF", line=LINE, oval=True,
               label=f"Feedback {value}")
        d.text(value, fx, 1.97, .65, .38, size=21, bold=True, center=True)
    d.text("Partial credit where available", 11.34, 2.70, 3.86, .34,
           size=17, center=True)

    d.text("LEARNING · after feedback", 8.94, 3.56, 3.95, .34,
           size=18, bold=True, color=ORANGE)
    d.line(13.28, 3.19, 13.28, 3.99, arrow=True, color=ORANGE,
           label="Choice and feedback enter learning")
    d.card(.2, 4.05, 4.7, 2.3, "If search: rule entry", learning=True)
    d.card(5.45, 4.05, 4.65, 2.3, "Search for a new rule", learning=True)
    d.card(10.65, 4.05, 4.75, 2.3, "Update all active rules", learning=True)
    for start, end in [(10.59, 10.16), (5.39, 4.96)]:
        d.line(start, 5.14, end, 5.14, arrow=True, color=ORANGE,
               label="Post-feedback learning flow")

    d.text("Retain beliefs, then weight\nby feedback likelihood.",
           10.82, 4.69, 4.38, .67, size=19)
    d.text("Correct: full · Incorrect: individual rate", 10.82, 5.43, 4.38, .35,
           size=17.5, color=ORANGE)
    d.text("Learn precision from feedback.", 10.82, 5.91, 4.38, .35, size=18)

    d.text("Failure history controls search.", 5.62, 4.69, 4.3, .37, size=17.5)
    d.text("Local", 5.66, 5.17, 1.85, .34, size=18.5, bold=True, color=BLUE)
    d.text("Global", 8.03, 5.17, 1.85, .34, size=18.5, bold=True, color=BLUE)
    d.line(7.77, 5.2, 7.77, 6.16, width=.8, color=LINE, label="Search branches")
    d.text("Edit a rule chosen\nby updated belief", 5.66, 5.58, 2.0, .65, size=17)
    d.text("Sample from\nthe rule catalogue", 8.03, 5.58, 2.0, .65, size=17)

    d.text("Add a rule; at capacity K,\nreplace the lowest-belief rule.",
           .37, 4.69, 4.35, .67, size=18.5)
    d.text("Feedback sets entry belief at capacity.", .37, 5.43, 4.35, .35, size=17.5)
    d.text("Inherit precision + pairing knowledge.", .37, 5.91, 4.35, .35, size=17)
    for start, end in [((2.55, 4.0), (2.55, 3.59)), ((2.55, 3.59), (5.25, 3.59)),
                       ((5.25, 3.59), (5.25, 3.2))]:
        d.line(*start, *end, arrow=end == (5.25, 3.2), color=TEAL,
               label="Updated state becomes next trial's active rules")
    d.text("Next trial", 3.15, 3.22, 1.62, .32, size=17, color=TEAL, center=True)
    d.text("Without search, keep updated rules. New rules act next trial; pairing knowledge applies to partial feedback.",
           .2, 6.5, 15.2, .32, size=16, color=MUTED)

    for element in d.elements:
        ex, ey, ew, eh = element["bounds_in"]
        if (ex < x - 1e-5 or ey < y - 1e-5 or ex + ew > x + width + 1e-5
                or ey + eh > y + HEIGHT * d.scale + 1e-5):
            raise ValueError(f"Element outside Figure 3: {element['name']}")
    return d


def validate_editability(path: Path) -> dict:
    """Check the deliverable contains native objects and editable Arial text."""
    prs = Presentation(path)
    with ZipFile(path) as archive:
        media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
    if len(prs.slides) != 1 or media:
        raise ValueError("The standalone schematic must have one slide and no image media")
    counts: dict[str, int] = {}
    for shape in prs.slides[0].shapes:
        counts[str(shape.shape_type)] = counts.get(str(shape.shape_type), 0) + 1
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                if para.text and para.font.name != "Arial":
                    raise ValueError(f"Unexpected text font: {shape.name}")
    return {"native_object_counts": counts, "embedded_images": len(media),
            "grouped_objects": counts.get("GROUP (6)", 0), "editable_text_font": "Arial"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("CategoryLearning_poster/figures"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any((output / name).exists() for name in ("figure3.pptx", "figure3.png", "figure3.manifest.json")):
        parser.error("Use a new output directory; existing figures are preserved")
    output.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(WIDTH), Inches(HEIGHT)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    drawing = draw_figure3(slide)
    slide.notes_slide.notes_text_frame.text = (
        "Model 1023 mechanism schematic. The normalized rule beliefs weight category "
        "predictions, local-search parent selection and capacity replacement. Decision "
        "predictions use region distance and learned precision. All active rules learn "
        "from choice-time predictions. Feedback modulates belief retention and, at full "
        "capacity, the newcomer's belief share. Precision and pairing knowledge are "
        "inherited from the feedback-updated state before replacement. Local search "
        "falls back to functional similarity when no structural neighbour is available. "
        "Half credit gives intermediate retention and entry shares. Pairing beliefs "
        "are specific to the partial-feedback condition. Bar lengths are illustrative; "
        "no participant observations or new model runs are represented. Each visible "
        "element is an ungrouped native PowerPoint object."
    )
    # Reuse the poster's exact-font conversion and PDF text-boundary checks.
    from ..build_poster import TEMPLATE_FONT, export_pdf, verify_render
    with tempfile.TemporaryDirectory(prefix="categorylearning-figure3-") as temporary:
        temp = Path(temporary)
        path = temp / "figure3.pptx"
        prs.save(path)
        editable = validate_editability(path)
        font_checks = export_pdf(path, TEMPLATE_FONT, compare_template=False)
        render_checks = verify_render(SimpleNamespace(textboxes=drawing.textboxes), path.with_suffix(".pdf"))
        page = fitz.open(path.with_suffix(".pdf"))[0]
        if page.get_images():
            raise ValueError("Rendered Figure 3 unexpectedly contains raster images")
        page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(output / "figure3.png")
        (output / "figure3.pptx").write_bytes(path.read_bytes())
    sources = [Path(__file__).resolve(), ROOT / "CategoryLearning_poster/build_poster.py",
               *SOURCES, ROOT / "src/Arial.ttf", ROOT / "src/Arial Bold.ttf"]
    manifest = {"model_version": "1023", "date": "2026-10-31",
                "archetype": "mechanism schematic", "canvas_inches": [WIDTH, HEIGHT],
                "interpretation": "Belief-weighted mixture choice; feedback-dependent learning and rule management",
                "empirical_data": False, "new_fits_or_simulations": 0,
                "native_editability": editable, "font_checks": font_checks,
                "pdf_validation": render_checks, "raster_images_in_pdf": 0,
                "elements": drawing.elements, "text_boxes": drawing.textboxes,
                "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources}}
    (output / "figure3.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "figure3.pptx")


if __name__ == "__main__":
    main()
