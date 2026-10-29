"""Generate independent numeric fixtures from the pinned author's source.

Run from the repository root with --source /tmp/.../sustain.py --output NEW.json.
The source is not vendored. Only the class is executed; its equations are left
unchanged. A list-returning map restores Python 2 iterator semantics. No model
code from src.SUSTAIN is imported here.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "5639bcc50a6cd195fe5d5246a8de24fa4f57cbea7113c5736a490a43b9a3ce16"
COMMIT = "21efe4df5b2ff4037fbfa5d33237feb7933fd1ac"


def generate(source: Path) -> dict:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source differs from the reviewed author revision")
    text = raw.decode()
    class_text = text[text.index("class SUSTAIN:"):text.index("# END SUSTAIN Class")]
    namespace = dict(array=np.array, resize=np.resize, float64=np.float64,
                     sum=np.sum, exp=np.exp, max=builtins.max, min=builtins.min,
                     map=lambda function, *items: list(builtins.map(function, *items)),
                     random=lambda: 0.5)
    exec(compile(class_text, "author_sustain_class", "exec"), namespace)
    problems_text = text.split("six_problems =", 1)[1].split("six_problems_env =", 1)[0].strip()
    problems = ast.literal_eval(problems_text)
    parameters = dict(r=2.0, beta=1.0, d=3.0, eta=0.2, tau=0.0)
    cases = []
    for problem in (0, 1, 5):
        model = namespace["SUSTAIN"](r=2., beta=1., d=3., threshold=0., learn=.2, initalphas=np.ones(4))
        # Distinct patterns plus later revisits exercise both recruitment and
        # movements of existing centers. The order is independent of our model.
        indices = [0, 7, 1, 6, 2, 5, 3, 4, 0, 2, 7, 5]
        records = []
        for index in indices:
            item = np.eye(2)[problems[problem][index]]
            probabilities = model.stimulate(item, ["k", "k", "k", "?"])[2][-1].copy()
            model.learn(item, ["k", "k", "k", "?"])
            clusters = np.asarray(model.clusters)
            records.append(dict(
                stimulus=item[:3].ravel().tolist(), target=int(problems[problem][index][-1]),
                probabilities=probabilities.tolist(), lambdas=model.LAMBDAS[:3].tolist(),
                clusters=clusters[:, :3].reshape(len(clusters), 6).tolist(),
                category_positions=clusters[:, 3].tolist(),
                weights=np.asarray(model.connections)[:, -2:].tolist(),
            ))
        cases.append(dict(problem=problem + 1, trials=records))
    return dict(
        source_url=f"https://raw.githubusercontent.com/NYUCCL/sustain_python/{COMMIT}/sustain.py",
        source_sha256=SOURCE_SHA256, source_commit=COMMIT,
        adaptation="Execute only the original class; Python 2 map returns lists; unused sampled response fixed at 0.5",
        comparison_scope="classification feature lambdas, feature/category positions, queried weights and pre-feedback probabilities",
        parameters=parameters, dimensions=[2, 2, 2], n_categories=2,
        recruitment="cluster_label", cases=cases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(generate(args.source), stream, indent=2, allow_nan=False)
        stream.write("\n")
