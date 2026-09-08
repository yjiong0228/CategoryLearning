# v9 verification

- 32 tests passed (figure/preprocessing tests plus existing oral-alignment suite).
- All 250,880 persisted mention values independently match one direct canonical process_use call followed by FEATURE_NAME_TO_PART mapping. No custom oral parser remains; git diff confirms src/oral_coding.py equals its pre-task version.
- The update audit preserves all original 22 CSV columns, 62,720 rows, and their order. 13 nonmissing mention cells changed; missing records inherit process_use zeros. Raw hashes match v7, including corrected subject 319.
- Current manifest, input, output, and code hashes verified. All 96 subjects retained, no duplicate keys, category/feedback/task-geometry mismatch zero. No invalid RTs.
- Main v8 image visually reviewed; v9 main rendering is unchanged. Candidate sheet v9 visually reviewed after increasing bottom margin. All observed RT/feedback values lie within the displayed log-axis limits, including S117's extreme feedback-history value. Valid denominators remain exported.
- Static validator: 10 PASS / 3 WARN / 1 FAIL. Vector/TIFF export findings are explicitly overridden by PNG-only repository instructions. Runtime DPI is configured to 450; static width warning misreads the 183/25.4 conversion. These exceptions do not constitute blanket journal submission validation.
- v8 is retained as an intermediate snapshot; v9 corrects candidate-sheet footer spacing. No new scientific analysis was selected by p values; none were computed. Comparisons remain descriptive and subject to documented stimulus/history/learning-stage confounds.
