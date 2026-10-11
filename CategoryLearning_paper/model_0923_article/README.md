# Model 0923 article draft

The independent manuscript entry point is [model_0923_article.tex](model_0923_article.tex).
The readable version is [model_0923_article.pdf](model_0923_article.pdf).
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
| Model architecture and equations | Current specified design |
| Revised candidate management and continuous choice readout | Code synchronization pending |
| Behavioral findings and model comparison | Results reserved |
| Latent trajectories and feedback-condition effects | Results reserved |
| Generative predictions and identifiability | Results reserved |
| Parameter-search settings, evaluation split and inferential statistics | To be specified for the final analyses |

The article describes the revised full model. Its empirical claims will be
written from the corresponding completed analyses. The abstract and Discussion
will be completed after those results. Participant recruitment, exclusions,
ethics and the final analysis cohort also require verified study information.

## Compile

From the repository root, enter this manuscript directory and compile into a
fresh temporary build directory. `latexmk` repeats LaTeX as needed to resolve
cross-references:

```bash
cd CategoryLearning_paper/model_0923_article
model0923_article_build="$(mktemp -d /tmp/model0923-article-build.XXXXXXXX)"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$model0923_article_build" model_0923_article.tex
```

The compiled PDF and logs are in the new build directory. After review, copy
the PDF to the article's readable path. LaTeX intermediate files remain in the
temporary directory. `build/previous/` retains local compiler state from the
earlier layout; its recorded paths predate the current directory structure.
