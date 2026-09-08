# v10 revision

Fold process_use and the subsequent FEATURE_NAME_TO_PART mapping directly into
Preprocessor_B.process. Remove code_feature_mentions; figure validation and the
standalone CSV refresh script independently use the existing process_use encoder.
No parser changes and no intended data-value changes; current CSV is not rewritten.

Remove the keyboard icon below Categorize (retain F/J). Use each task's existing
color for shorter F1–F4 report marks in both the main figure and oral atlases.
Swap panel d to RT left, report feature count right. All scientific definitions,
examples, inclusion rules and other panels are preserved. Update legends accordingly.

Python/matplotlib, 450 dpi PNG only, new version directory; model pipeline migration
remains deferred. See REVISION_v8.md for the non-oral exploratory analysis contract.
