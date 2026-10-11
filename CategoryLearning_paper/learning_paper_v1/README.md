# Learning paper v1

The independent manuscript entry point is [learning_paper_v1.tex](learning_paper_v1.tex).
The readable version is [learning_paper_v1.pdf](learning_paper_v1.pdf).
[results.tex](sections/results.tex) contains the main-text model account and reserved
empirical subsections. [methods.tex](sections/methods.tex) contains the mathematical
specification and inference procedures.

## Scope and source

This draft reorganizes the current Model 0923 design for a research article.
Its primary scientific source is the unchanged
[model specification](../../src/Bayesian_state/docs/model_architecture/model_0923.tex),
with the current hypothesis-updating and choice-readout sections in the
[Markdown document](../../src/Bayesian_state/docs/model_architecture/model_0923.md).
The [previous manuscript](../sn_article/README.md) is maintained in the parallel
`sn_article/` directory. Shared journal class and bibliography styles are in
`../latex/`; `.latexmkrc` supplies their search paths.

The organization follows the supplied
[Weiss et al. (2021) article](<../references/Weiss et al_2021_Interacting with volatile environments stabilizes hidden-state inference and.pdf>),
*Interacting with volatile environments stabilizes hidden-state inference and
its brain signatures*, [Nature Communications 12, 2228](https://doi.org/10.1038/s41467-021-22396-6).
In that article, the Results model overview follows the initial behavioral
analysis and connects computational assumptions to later empirical analyses.
Methods defines the update equations and estimation procedures. The new draft
uses that division to develop an account of rule discovery and use.

## Reading structure

The opening frames how discovery, evidence accumulation and persistent rule
use can each change choices. Results first introduces the task and reserves
the behavioral findings, then develops the model in connected paragraphs.
Later Results subsections and the Discussion remain explicit placeholders.
Methods provides the definitions, trial timing, probability distributions and
parameter domains needed to reproduce the specified model.

| Term | Definition used throughout the draft |
|---|---|
| Candidate set | Rules currently available for prediction and feedback learning |
| Rule belief | Relative support for a rule within the candidate set |
| Choice weight | A rule's contribution to the current response distribution |
| Rule precision | Concentration of a rule's category predictions as a function of distance |
| Dominant rule | Candidate with the largest choice weight, recorded before choice |
| Particle weight | Researcher's weight on a possible internal learning path |

The cognitive modules are perception (P), evidence memory (M), hypothesis
updating (H), precision updating (B) and choice readout (RC). Their terminology
and notation follow the current specification.

## Evidence and completion status

| Component | Current basis or status |
|---|---|
| Title and abstract | Author-supplied text |
| Model architecture and equations | Current specified design |
| Revised candidate management and continuous choice readout | Code synchronization pending |
| Behavioral findings and model comparison | Results reserved |
| Latent trajectories and feedback-condition effects | Results reserved |
| Generative predictions and identifiability | Results reserved |
| Parameter-search settings, evaluation split and inferential statistics | To be specified for the final analyses |

The title and abstract were supplied by the author. The empirical Results and
Discussion remain to be completed from the corresponding analyses. Participant
recruitment, exclusions, ethics and the final analysis cohort also require
verified study information.

## Compile

From the repository root, enter this manuscript directory and compile into a
fresh temporary build directory. `latexmk` repeats LaTeX as needed to resolve
cross-references:

```bash
cd CategoryLearning_paper/learning_paper_v1
learning_paper_build="$(mktemp -d /tmp/learning-paper-v1-build.XXXXXXXX)"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$learning_paper_build" learning_paper_v1.tex
```

The compiled PDF and logs are in the new build directory. After review, copy
the PDF to the article's readable path. LaTeX intermediate files remain in the
temporary directory. `build/previous/` retains local compiler state from the
earlier layout; its recorded paths predate the current directory structure.
