# Submission Email

To: ennovatex.io@samsung.com

Subject: AX Hackathon Phase 2 Submission | 04 | Resonant

Hello Samsung ennovateX team,

Please find enclosed our Phase 2 submission for Problem #04 - Speech Disentanglement.

Team: Resonant
Participant: Pranav Angrish (pangrish_be22@thapar.edu)
Institute: Thapar Institute of Engineering & Technology

Attachments / links:
- Final report PDF: resonant-04-solospeak-final.pdf
- GitHub repository: `https://github.com/pranavangrish/solospeak`
- Demo video: `PASTE_UNLISTED_YOUTUBE_URL_HERE`
- Trained artifacts: `https://github.com/pranavangrish/solospeak/releases/tag/v1.0.0-phase2`

Headline results, with full tables in the report:
- TA Clean: 100.0% (MIN gate 0.92, TARGET 0.96)
- TA Noisy macro: 100.0% (MIN gate 0.80, TARGET 0.88)
- FA per hour per user: 0.00
- Q2 imposter rejection: 100.0%
- Q3 phonetic-neighbor rejection: 100.0%
- Model size: 1.10M params, 1.47 MB INT8
- xRT local p95: 0.0032 (HARD gate 0.20, STRETCH 0.08)
- Disentanglement: speaker-probe-on-z_c reduced by 100.0% vs Stage 2 baseline
- All 4 quadrants (Q1/Q2/Q3/Q4) explicitly evaluated

Important caveat: current generated numbers are smoke-run numbers unless replaced by a
full-data training and evaluation run before submission.

Looking forward to Phase 3.

Best,
Pranav Angrish
