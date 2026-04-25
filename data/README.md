# Data

Audio files are NOT committed to git. Only manifests (CSV files listing paths + labels) are committed.

## Setup

```bash
make download-data      # downloads all datasets to data/raw/
make prepare-manifests  # generates CSVs in data/manifests/
```

## Datasets

| Dataset | License | Role | Size | Download |
|---------|---------|------|------|----------|
| Google Speech Commands v2 | CC-BY 4.0 | Stage 1 backbone pretrain | ~2.4 GB | Auto via script |
| LibriPhrase | CC-BY 4.0 (based on LibriSpeech) | Primary content training | ~30 GB LibriSpeech base | Auto via script |
| VoxCeleb 1 | CC-BY 4.0 | Speaker head training + eval | ~39 GB | **Requires academic access** |
| VoxCeleb 2 | CC-BY 4.0 | Speaker head training | ~200 GB | **Requires academic access** |
| MUSAN | CC-BY 4.0 | Noise augmentation | ~11 GB | Auto via script |
| BUT Reverb DB / OpenSLR-28 | CC-BY 4.0 / CC0 | Room impulse responses | ~8 GB | Auto via script |

### VoxCeleb Access

VoxCeleb requires an academic use agreement. Request access at:
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

Save the approval email to `data/licenses/voxceleb_approval.pdf` for Phase 2 submission proof.

## Manifests (committed)

Seven CSV files, each with columns: `file_path,duration_s,speaker_id,keyword_text,split,source_dataset,quadrant_class`

| File | Purpose | Expected Rows |
|------|---------|---------------|
| train_content.csv | Content head training | ~40,000 |
| train_speaker.csv | Speaker head training | ~1,000,000 |
| train_gsc.csv | Stage-1 backbone pretrain | ~85,000 |
| dev_content.csv | Content validation | ~5,000 |
| dev_speaker.csv | Speaker validation | ~50,000 |
| test_kpi.csv | 4-quadrant KPI eval | ~2,000 |
| test_fa.csv | 10 hours FA-rate eval | ~600 |

## License Audit

All datasets are CC-BY 4.0 or CC0, compatible with Apache-2.0 distribution of this codebase.
Parler-TTS (used at enrollment) is Apache-2.0.
g2p-en is MIT.
Silero-VAD is MIT.
