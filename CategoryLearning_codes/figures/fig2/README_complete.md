# Complete Fig2 draft

Current: [Fig2 v9](../outputs/fig2/fig2_complete_v9/Figure2_draft.png) (183 × 240 mm, 450 dpi, PNG).

Panel plan: a framework; b Task1 existing S101 case; c Task2 individual example; d Task3 individual example; e group behavioral prediction; f group oral–model alignment; g key model ablations. c–g contain only empty outlined rectangles with panel titles. Group panels eventually contain all three tasks; their existence is not evidence that results are available.

b uses all 320 S101 trials: trailing 32-trial observed/model accuracy; model and oral distributions over the same 29 hypotheses and common 0–1 scale; trailing 32-trial distribution overlap. H0 is the target feature-1 threshold. Full definitions and timing limitations are in README_behavior_state.md. This compact layout uses full-space overlap; the separate target-based and feature-mention panels remain available in case_sources and earlier drafts for layout alternatives. S101 is selected by availability, not claimed as a final representative participant.

Oral center sigma is now 0.05 by user decision in both the new package and legacy evaluation entry point; figure CLI defaults also use 0.05. Region temperature and behavioral model fitting are unchanged. Historical artifacts, explicit old sensitivity settings, and migration hashes remain historical records; do not mix them as if they used the current scale.

Reproduce from root with a new directory:

```bash
python -m CategoryLearning_codes.figures.fig2.build_complete --output CategoryLearning_codes/figures/outputs/fig2/fig2_complete_v10
```

Source tables, case code snapshot and input hashes are in case_sources; whole layout code and image hash are in this output. Panel a reuses the existing PNG intact, so its effective resolution/text size decreases when reduced; it should be rendered natively once the layout is settled. Current figure is for composition review, not submission certification. Viewed rendered output: no clipped panels. Oral regression suite: 18 passed; both old/new default constants and public method defaults checked at 0.05. New case generation checks trial alignment and distribution normalization.

## Full-width framework revision

Current whole figure: `../outputs/fig2/fig2_complete_v3/Figure2_draft.png`; standalone panel: `fig2a_framework_wide.png` in the same directory. Panel a uses the same horizontal limits (.075–.970 of the canvas) as the three-column result grid and is rendered natively by framework_wide.py, eliminating the previous reduced PNG. It expands finite workspace maintenance, two choice structures, search-event vs local/global proposals, belief transport, fading evidence and precision updates for all active rules. Dashed arrows mark next-trial influence; PF remains a numerical inference note. Three rule cards are schematic illustrations, not a fitted capacity. Formula and update scope checked against model_0826.tex; the schematic remains a compact abstraction of the manuscript, whose current formal implementation is binary-category.

v2 was retained; v3 fixes the stimulus heading and bottom-note clipping seen in its standalone render. Final complete figure visually inspected. Case data and missing-result placeholders unchanged; regenerated source checks passed. The former PNG-resolution limitation above applies to v1 only.

## Symbol/formula revision (v4)

Current: `../outputs/fig2/fig2_complete_v4/Figure2_draft.png` and standalone `fig2a_framework_wide.png`. Removed explanatory sentences from a, replacing them with workspace set/capacity notation, the two manuscript readout formulas, retain/local/global probabilities, fading-memory update, precision up/down indicators, failure recurrence and a short PF weight formula. E and g in the search box refer explicitly to t+1. Precision arrows summarize bounded updates rather than specify step size. Sigma_s denotes the subject's perception covariance, not oral sigma. Three candidate-rule icons remain schematic. Render inspected for clipping and equation overlap. Quantitative source regeneration retains sigma_oral=.05 and passes existing inline alignment/normalization checks.

Final symbol draft: fig2_complete_v5. Perceptual noise is written as P_s^percept because the manuscript allows measured Gaussian (including bias) or uniform noise; it is not restricted to zero-mean Gaussian. This replaces the provisional Sigma_s notation in v4.

## Mechanism demonstration (v7, 2026-09-08)

Current whole figure: `../outputs/fig2/fig2_complete_v7/Figure2_draft.png`; standalone framework: `fig2a_framework_wide.png`. The main change is a concrete illustrative episode in place of boxes of formulas:

- The same perceived stimulus appears in each rule partition. Category 1 is pale blue and category 2 white; probability bars encode category 1 in blue and category 2 in gray.
- Three candidate rules disagree. Their pre-choice beliefs determine line widths in the mixture readout. The persistent alternative uses only h1. These are alternative structures, not simultaneous stages.
- Shown choice 2 is incorrect. Outlined bars show pre-feedback beliefs, filled bars show updated beliefs. They change from [.60,.25,.15] to [.165,.606,.229] under fading-memory gamma=.8. Rule precision changes from [8,8,8] to [6.477,11.237,8]; dashed emission curves are before feedback, solid curves after. Curve x is signed category-region distance difference, y is category-2 probability. All these quantities are analytic examples, not participant results.
- Search shows active rules as filled nodes, inactive candidates as open nodes, and overlapping local neighborhoods for all active rules; opacity qualitatively follows belief. Node layout is schematic, not a computed rule-distance embedding. A local candidate h4 is circled; a separate dashed global proposal points to a distant inactive node. One possible subsequent workspace replaces h3 by h4, retaining h1 even under persistent execution. Error does not imply obligatory replacement; the retain/search branch probabilities remain visible. Transport determines newcomer beliefs in the actual model; none are invented in the next-workspace illustration.
- Lowercase h1–h4 are illustration labels rather than the H0–H28 catalog indices used in empirical panel b. The rule projections are F3 threshold, F1 threshold, F1+F3 threshold and F1/F3 comparison. The 2-D card axes correspond to F1 horizontally and F3 vertically; other features are irrelevant to these four examples.

Illustrative values and their calculation are saved in `schematic_episode.json` and `framework_wide.py`. Source preservation checked: six empirical CSVs are byte-identical to v5; both renderer snapshots match current code. Probability normalization and direction of belief/precision changes asserted. Final whole image and standalone render inspected. Component-only static validator does not see font/export configuration in the parent renderer; those checks apply to build_complete.py. Vector/TIFF export requirements are overridden by PNG-only repository policy. Draft resolution remains 450 dpi; no claim of submission readiness. Old versions retained.

## Explicit memory stage (v9)

Current framework and full figure are in `../outputs/fig2/fig2_complete_v9/`.
Memory is now visible as a separate pale panel: prior -> normalized prior^gamma, followed by multiplication by current feedback likelihood and normalization. These intermediate bars are an algebraic visualization of the existing fading-memory update, not an added model stage or separately stored cognitive variable. With gamma=.8 the prior [.60,.25,.15] contracts toward uniform while preserving rank; dotted outlines in the faded histogram show the original heights. The updated distribution is identical to v7; empirical source tables remain byte-identical. gamma=1 retains prior odds, gamma<1 attenuates them before incorporating current evidence. This is distinct from the controller's feedback-history variable F. The current stimulus prediction still uses the original pre-choice prior, not the faded intermediate display.

Validation: two-step normalization equals the manuscript's joint normalization; probability sums and rank preservation checked, peakedness reduced as expected; final standalone render inspected. v8 retained; v9 moves the category key clear of the memory heading and routes feedback clear of the update title.
