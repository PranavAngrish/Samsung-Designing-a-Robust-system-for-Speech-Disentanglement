# Manifest Statistics

These are the committed smoke-manifest statistics used for CI and local wiring checks.
They are not the full production training or final Stage 7 evaluation counts.

- **train_content.csv**: 80 rows, 8 speakers, 8 keywords
- **train_speaker.csv**: 80 rows, 12 speakers, 0 keywords
- **train_gsc.csv**: 350 rows, 350 speakers, 35 keywords
- **dev_content.csv**: 24 rows, 4 speakers, 4 keywords
- **dev_speaker.csv**: 24 rows, 4 speakers, 0 keywords
- **test_kpi.csv**: 40 rows, 5 speakers, 4 keywords
- **test_fa.csv**: 6 rows, 6 speakers, 0 keywords
- **n_gsc_classes**: 35
- **n_aux_word_classes**: 8
- **n_aux_speaker_classes**: 28
- **smoke**: true

The final Stage 7 claim uses a separate full-data run and 40,000 external false-accept
trials; see `docs/release_manifest.md` and `docs/reproducibility.md`.
