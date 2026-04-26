# Data

Audio files are not committed to git. Manifests, vocab files, license notes, and smoke
fixtures are the reproducible interface between data preparation and training.

## Smoke Path

```bash
make download-data-smoke
make prepare-manifests-smoke
make pack-lmdb
```

The smoke path creates tiny deterministic audio under `data/raw/smoke/`, writes all seven
canonical manifests, and can run without network access.

## Full Path

```bash
make download-data
python -m scripts.prepare_libriphrase --segments-csv data/processed/libriphrase_segments.csv
make prepare-manifests
make pack-lmdb
```

LibriPhrase is generated from LibriSpeech after Montreal Forced Aligner 3.3.9 alignment
using the `english_us_arpa` acoustic model and dictionary. Cache TextGrids under
`data/processed/libriphrase_alignments/` and record their checksum here before training.

## Manifest Schema

Every manifest uses:

```text
file_path,start_s,end_s,duration_s,speaker_id,keyword_text,split,source_dataset,quadrant_class,profile_id,enrolled_user_id,enrolled_keyword_text,profile_path,trial_source,q3_gate_eligible,synthesis_backend,speaker_verification_score
```

After `make pack-lmdb`, `file_path` values are rewritten to
`lmdb://data/processed/audio.lmdb/<key>` and `data/processed/audio_lmdb_index.json`
records the original source path, segment offsets, duration, and checksum.

## Required Files

- `train_content.csv`
- `train_speaker.csv`
- `train_gsc.csv`
- `dev_content.csv`
- `dev_speaker.csv`
- `test_kpi.csv`
- `test_fa.csv`
- `keyword_vocab.json`
- `gsc_vocab.json`
- `speaker_vocab.json`
- `STATS.md`
- `STATS.json`

## License Notes

Save one markdown note per corpus under `data/licenses/` with source URL, license/terms
snapshot date, allowed use, attribution text, and restrictions. VoxCeleb approval terms
must be recorded before using VoxCeleb audio; otherwise use the Common Voice/LibriSpeech
fallback and document the decision.
