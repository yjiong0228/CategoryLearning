"""Render the saved Fig1–3 panels with the poster's supplied Arial files.

This is a typography-only reuse of the established Python plotting functions.
All participants, curves, masks, statistics and source tables keep their existing
definitions. Outputs are PNG and are stored in new per-figure directories.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text
import pandas as pd

from ..fig1 import build_journal_figure as fig1
from . import build_journal_figure as fig2
from ..fig3 import build_journal_figure as fig3

ROOT = Path(__file__).resolve().parents[3]
REGULAR = ROOT / "src/Arial.ttf"
BOLD = ROOT / "src/Arial Bold.ttf"
BASE = ROOT / "CategoryLearning_codes/figures/outputs"


@contextmanager
def supplied_fonts(audits: list[dict]):
    for path in (REGULAR, BOLD):
        font_manager.fontManager.addfont(path)
    rc = {"font.family": "sans-serif", "font.sans-serif": ["Arial"],
          "mathtext.fontset": "custom", "mathtext.rm": "Arial",
          "mathtext.it": "Arial", "mathtext.bf": "Arial:weight=bold",
          "mathtext.bfit": "Arial:weight=bold", "mathtext.sf": "Arial",
          "mathtext.tt": "Arial", "mathtext.cal": "Arial", "mathtext.fallback": None}
    savefig = Figure.savefig

    def save_with_fonts(figure, filename, *args, **kwargs):
        with mpl.rc_context(rc):
            figure.canvas.draw()
            counts = {"regular": 0, "bold": 0}
            for artist in figure.findobj(match=Text):
                weight = artist.get_fontweight()
                heavy = weight in {"bold", "heavy", "semibold", "demibold", "black"}
                heavy |= isinstance(weight, (int, float)) and weight >= 600
                prop = artist.get_fontproperties().copy()
                prop.set_family("Arial")
                prop.set_file(BOLD if heavy else REGULAR)
                artist.set_fontproperties(prop)
                if artist.get_text():
                    counts["bold" if heavy else "regular"] += 1
            result = savefig(figure, filename, *args, **kwargs)
            audits.append({"image": Path(filename).name, "text_artists": counts,
                           "mathtext": "Arial regular / bold; fallback disabled"})
            return result

    with mpl.rc_context(rc), patch.object(Figure, "savefig", save_with_fonts):
        yield


def build(version: str) -> None:
    outputs = {key: BASE / key / version for key in ("fig1", "fig2", "fig3")}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Use a new version; existing figure outputs are preserved")
    audits: list[dict] = []
    with supplied_fonts(audits):
        # Reuse archived tables so this font revision reads no raw data.
        old = BASE / "fig1/journal_20260921_v2"
        people = pd.read_csv(old / "subject_source.csv")
        trials = pd.read_csv(old / "trial_source.csv")
        tasks = json.loads(Path(fig1.__file__).with_name("config.json").read_text())["tasks"]
        outputs["fig1"].mkdir(parents=True)
        fig1.render(trials, people, tasks, outputs["fig1"])
        fig2.build(ROOT / fig2.DEFAULT_STATES, ROOT / fig2.DEFAULT_VALIDATION, outputs["fig2"])
        fig3.build(ROOT / "results/model_0826/fig34_twelve_subjects_20260921_v1/analysis",
                   outputs["fig3"])
    for key, output in outputs.items():
        (output / "poster_font_audit.json").write_text(json.dumps({
            "scope": "Font-only rendering; existing scientific definitions and observations",
            "backend": "Python / matplotlib", "format": "PNG",
            "fonts_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (REGULAR, BOLD)},
            "render": audits[int(key[-1]) - 1],
            "new_model_fits": 0, "new_state_replays": 0, "new_oral_encoding": 0,
        }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if Path(args.version).name != args.version:
        parser.error("--version must be a single directory name")
    build(args.version)


if __name__ == "__main__":
    main()
