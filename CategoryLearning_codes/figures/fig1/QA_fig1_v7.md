# Fig1 v7 validation

- 12 figure/preprocessing tests passed; 18 existing oral alignment tests passed.
- CSV append audit verified exact equality of all original fields for 62,720 rows. The original hash matches the corrected S319 version. 664 missing texts remain missing in all new fields.
- Figure entry point independently recomputes all four mention columns and verifies equality before plotting.
- All subject summaries are identical to v6, including accuracy, example selection and the three oral summary measures. The change aligns display coordinates without changing these descriptive statistics.
- Manifest output hashes, code hashes and raw input hashes verified after completion. The main image is 3242×3986 pixels at approximately 450 dpi (183×225 mm). Eight PNGs generated; paper figures directory remains empty.
- Whole main image visually inspected: simplified grayscale stimulus, no quote bubble, five evenly spaced screens, aligned F1–F4 ticks, no obvious clipping. Dense consecutive mentions naturally form black bands; missing text remains gray and is not interpolated.
- Static Nature validator: 10 PASS, 3 WARN, 1 FAIL. Missing SVG/PDF/TIFF is intentional under the repository's PNG-only instruction. Dynamic DPI is verified from output metadata; the detected 4648.2-mm width is a static-parser misreading of 183/25.4. This is a reviewed draft, not a claim of blanket submission-validator compliance.
- Reviewed existing oral diagnostics. Fidelity is stimulus-description consistency, including a legacy region fallback, not categorical rule correctness. No new fidelity scores or strategy-recovery claims were added to this revision.
