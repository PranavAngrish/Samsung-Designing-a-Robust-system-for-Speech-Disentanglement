# Demo Video Script

Target: 8 to 10 minutes, 1080p, 30 fps.

| Time | Segment | Notes |
|---:|---|---|
| 0:00-0:30 | Title | "SoloSpeak: a wake word that listens for both phrase and speaker." |
| 0:30-1:30 | Problem | Shared Samsung devices should not wake for every person who says the phrase. |
| 1:30-2:30 | Enrollment | Show `--enroll --user pranav --keyword "hey prism" --mic` or a prerecorded equivalent. |
| 2:30-3:30 | Q1 Accept | Enrolled user says the enrolled phrase and the system fires once. |
| 3:30-4:30 | Q2 Reject | Another speaker says the same phrase and the system rejects it. |
| 4:30-5:15 | Q3 Reject | Enrolled user says a wrong or nearby phrase and the system rejects it. |
| 5:15-6:00 | Q4 Reject | Background, music, or non-speech audio does not trigger the model. |
| 6:00-7:15 | Architecture | Explain content head, speaker head, fusion MLP, and local enrollment templates. |
| 7:15-8:15 | Final Metrics | Show Stage 7 corrected metrics: `tau=0.27`, TA clean 93.97%, Q2 95.17%, Q3 97.83%, Q4 100%, external FA 120/40000. |
| 8:15-9:15 | Calibration Fix | Explain the bad `tau=0.935` external-only export and the corrected joint calibration. |
| 9:15-10:00 | Samsung Fit | Bixby, Buds, TVs, SmartThings, release artifact, repo, and contact. |

Use a headset mic for narration and the laptop mic only for live model input. If the
room is noisy, use prerecorded clips but label them honestly. Upload the final video as
unlisted and add the URL to `docs/submission_email.md` before sending.
