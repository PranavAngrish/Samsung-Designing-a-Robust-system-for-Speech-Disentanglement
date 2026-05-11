# Submission Email

To: ennovatex.io@samsung.com

Subject: Samsung ennovateX AX Hackathon Submission | Problem 04 | SoloSpeak

Hello Samsung ennovateX team,

Please find enclosed my submission for Problem #04, Speech Disentanglement.

Team: Resonant

Participant: Pranav Angrish (pangrish_be22@thapar.edu)

Institute: Thapar Institute of Engineering & Technology

Attachments / links:

- Final report PDF: `resonant-04-solospeak-final.pdf`
- GitHub repository: `https://github.com/pranavangrish/solospeak`
- Demo video: add the unlisted video URL before sending
- Trained artifacts: `https://github.com/pranavangrish/solospeak/releases/tag/v1.0.0-stage7-corrected`

Headline results from the corrected Stage 7 artifact:

- Final deployable: `exports/solospeak_stage7_deployable_corrected.pt`
- Final threshold: `tau_on = 0.27`
- TA clean: 93.97%
- Q2 imposter rejection: 95.17%
- Q3 wrong-word rejection: 97.83%
- Q4 background rejection: 100.00%
- Quadrant minimum: 93.97%
- External false-accept rate: 0.300% (`120 / 40,000`)
- Model size: 1.10M parameters

The final export uses joint internal plus external threshold calibration. An earlier
external-only calibration selected `tau=0.935`, which suppressed true accepts; the
submitted artifact uses the corrected `tau=0.27` threshold.

The repository contains the source-controlled migration of the successful Kaggle
training path, including Stage 4D hard-Q2 mining, Stage 5 fusion search, Stage 7
external false-accept tuning, final verification, and corrected export.

Best,
Pranav Angrish
