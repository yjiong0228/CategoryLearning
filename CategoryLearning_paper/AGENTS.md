# Journal manuscript instructions

Root AGENTS.md applies.

- This directory owns manuscript sources and references. Development code and
  draft figures belong in CategoryLearning_codes.
- Working manuscripts are parallel directories: sn_article/ contains the
  previous manuscript; model_0923_article/ contains the revised English draft.
  Each owns its main TeX, readable PDF and sections/. Keep the navigation in
  README.md current when moving manuscript files.
- latex/ holds the shared journal class, bibliography styles and template
  examples. Each manuscript's .latexmkrc loads latex/latexmkrc for shared
  search paths. Validate compilation in a temporary output directory.
- Keep existing manuscript PDFs and bibliographies when reorganizing files.
  build/previous/ retains local compiler state from the earlier layout;
  its recorded paths may predate the current directory structure.
- figures/ contains only user-confirmed figures, named FigureN; default PNG.
- Preserve references, bibliography, journal class/style files and cited material.
- Do not classify .bbl or compiled .pdf as disposable solely by extension.
  They can be needed for submission or a historical record.
- Model technical definitions are in src/Bayesian_state/docs/model_architecture/model_0826.tex at repository root;
  results must match the actual configuration and validation status.
- Do not describe preliminary single-subject results as formal group fits. Preserve
  explicit placeholders where fitting or mechanism validation remains incomplete.
