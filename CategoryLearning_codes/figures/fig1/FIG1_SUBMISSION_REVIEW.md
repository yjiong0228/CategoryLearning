# Fig1 v10 review and promotion

v10 can serve as the current confirmed manuscript figure: it shows actual data,
all 96 subjects, explicit record coverage, aligned report coding, and participant
paired summaries. No unresolved data-integrity or figure-overlap blocker was found.
Copied byte-for-byte to CategoryLearning_paper/figures/Figure1.png.
SHA256: 93b7b9fd840ac2137d21a701a7ef93f05cfc378b4ec20e65cd77ec5e09372a11

Before journal submission:
- Check print-size readability of 5-pt oral labels; enlarge/separate supplementary
  examples if the target journal's layout makes them unreadable.
- Confirm actual trial timings and whether the ready/report screens are faithfully
  represented; current panel b explicitly remains a timing schematic.
- Legend must define fixed representative selection, task/condition mapping,
  feedback=1 correctness, missing heatmap vs missing report cells, disjoint 64-trial
  summaries and S105's paired-only omission. Record duration is not mastery time.
- Keep task differences descriptive until a justified participant-level inferential
  analysis is specified. Different learning duration/difficulty distributions limit
  causal task comparisons. RT uses a logarithmic axis and no long-tail trimming.
- Confirm target journal export requirements at submission; current PNG-only output
  follows repository instructions, not a claim to meet every journal format.

The redundant one-time data_preparation script/directory was removed. Its seven
integrated preprocessing regression cases were retained in tests/test_preprocess_b.py;
prior update audit/backups remain under figures/outputs/fig1/.
