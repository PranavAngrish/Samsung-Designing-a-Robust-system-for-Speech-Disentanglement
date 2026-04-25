# SoloSpeak: Speaker-Personalized Custom Wake-Word Detection
## Master Architecture Document — Samsung ennovateX AX Hackathon 2026, Problem #04

> This is the single source of truth for the entire hackathon. Blueprint, implementation, evaluation, report, and demo all flow from this document. Every claim in the blueprint maps to a component here. Every KPI target maps to an evaluation procedure here.

---

## 0. Executive Summary

**Product name:** SoloSpeak (internal codename: SS). Alternates if you want a different brand: PersonaWake, DuoSense, VoxKey. Pick one and stick with it.

**One-line pitch:** A sub-3M-parameter on-device keyword spotter that only wakes for *your* voice saying *your* custom word — robust from −5 dB to 30 dB SNR, from 0.5 m to 5 m, trained end-to-end on open data and released under Apache-2.0.

**Core technical bet:** A single shared audio encoder with two orthogonal output heads — a **content head** (what was said) and a **speaker head** (who said it) — trained jointly with a disentanglement objective (adversarial gradient reversal + orthogonality penalty). At inference, a **learned gating fusion** combines the two similarities into a single accept/reject decision.

**Why this wins:**

1. Samsung cares about Bixby and Galaxy Buds — this problem is their problem.
2. Sub-3M parameters means it runs on a phone CPU, which is the exact bar Samsung cares about.
3. The disentanglement + gating approach is genuinely novel vs. published SOTA (OpenWakeWord, Porcupine, Google TCResNet, Apple E2E-KWS) — they all treat KWS and SV as separate systems or ignore speaker identity entirely.
4. The KPIs are binary and measurable — you either hit them or you don't. No hand-waving possible.
5. The live demo is electric: you enroll a made-up word in 30 seconds, then only you can wake the device with it while an imposter screams the same word 2 meters away and nothing happens.

**Target KPIs (from the problem statement — memorize these):**

| Metric | Target | Our Plan |
|---|---|---|
| True Acceptance (Clean) | ≥ 99% | 99.2% (achieve with curriculum training + threshold tuning) |
| True Acceptance (Noisy) | ≥ 90% | 91.5% avg over −5 to 30 dB SNR |
| False Acceptance | < 1/hr | < 0.5/hr with hard-negative mining |
| SNR Range | −5 to 30 dB | On-the-fly SNR sampling from MUSAN |
| Distance Range | 0.5 m to 5 m | RIR convolution with BUT ReverbDB + simulated |
| Model Parameters | < 3 M | Target 1.8–2.4 M |
| xRT | < 0.2 | Target < 0.1 after INT8 quantization |

---

## 1. Problem Formulation (Formal)

### 1.1 Task definition

Let $x \in \mathbb{R}^T$ be a 1-second audio clip (at 16 kHz, $T = 16000$). Let $u$ denote a specific user and $w$ denote a specific custom word. During **enrollment**, the user provides:

- $E_u = \{e_1, \dots, e_N\}$: $N$ short utterances of user $u$'s voice (for speaker profile), $N \in [3, 5]$
- $K_w = \{k_1, \dots, k_M\}$: $M$ utterances of the word $w$ (can be from the user themselves or from TTS-augmented samples), $M \in [3, 10]$

During **detection**, at every 100 ms stride over a streaming audio buffer, the system must output a binary decision $y \in \{0, 1\}$:

$$y = 1 \iff (\text{utterance contains } w) \wedge (\text{speaker is } u)$$

### 1.2 Four confusion classes

The system must correctly handle all four quadrants:

| Quadrant | Content | Speaker | Expected Output |
|---|---|---|---|
| Q1 (True Positive) | word $w$ | user $u$ | 1 (wake) |
| Q2 (Imposter) | word $w$ | other speaker | 0 (reject) |
| Q3 (Wrong Word) | other/similar word | user $u$ | 0 (reject) |
| Q4 (True Negative) | other word | other speaker | 0 (reject) |

Most existing systems only robustly handle Q1 and Q4. **Q2 and Q3 are the hard cases and our differentiator.** Q3 is especially hard when the wrong word is *phonetically similar* to $w$ (e.g., "samsara" vs. "samurai" or "banana" vs. "bandana").

### 1.3 Metrics (formal)

- **True Acceptance (TA):** $\Pr(y=1 \mid Q1)$. Target ≥ 99% clean, ≥ 90% noisy.
- **False Acceptance (FA):** $\Pr(y=1 \mid Q2 \cup Q3 \cup Q4)$ measured in false-alarms-per-hour over long continuous audio. Target < 1/hr.
- **Equal Error Rate (EER):** Operating point where FAR = FRR. Report for completeness.
- **DET curve:** Plot FRR vs. FAR for the full threshold sweep. Required for the report.

---

## 2. System Architecture — High-Level View

```
                        ┌────────────────────────────────────────────┐
                        │  ENROLLMENT (one-time, offline, 30 sec)    │
                        │                                            │
 User mic ──► VAD ──►  Log-mel  ──►  Shared Encoder  ──►  Content ─►│── keyword templates C_w
                                                      ╲              │
                                                       ╲►  Speaker ─►│── speaker profile S_u
                        │                                            │
                        │  [Optional] TTS augmentation of            │
                        │  additional keyword samples                │
                        └────────────────────────────────────────────┘
                                              │
                                              ▼
                        ┌────────────────────────────────────────────┐
                        │  STREAMING INFERENCE (real-time, on-device)│
                        │                                            │
 Live mic ──► Ring ──► VAD ──► Log-mel ──► Shared Encoder ──► Content ──►  sim(·, C_w)  = s_c
              buffer   gate   (sliding                    ╲                                  ╲
              (1.5 s)         window,                      ╲► Speaker ──►  sim(·, S_u) = s_s  ╲
                              100 ms                                                           ╲►  Gated
                              stride)                                                              Fusion
                                                                                                    │
                                                                                                    ▼
                                                                                            Accept / Reject
                                                                                          (with hysteresis)
                        └────────────────────────────────────────────┘
```

### 2.1 Data path summary

1. **Audio input** → 16 kHz mono, ring buffer of 1.5 s
2. **VAD (Silero-VAD tiny)** → gate expensive inference; if no speech, skip
3. **Log-mel spectrogram** → 80 mel bins, 25 ms window, 10 ms hop → shape (80, ~100)
4. **Shared encoder (BC-ResNet-style)** → frozen-backbone-like embedding → shape (D,)
5. **Content head** → L2-normalized content embedding $z_c \in \mathbb{R}^{d_c}$
6. **Speaker head** → L2-normalized speaker embedding $z_s \in \mathbb{R}^{d_s}$
7. **Similarity computation** → cosine sim against templates
8. **Gated fusion** → learned combiner → logit → sigmoid → probability
9. **Hysteresis threshold** → rising threshold $\tau_{on}$, falling threshold $\tau_{off}$, to prevent flapping

---

## 3. Component Specifications

### 3.1 Audio frontend

- **Sample rate:** 16 kHz (standard for speech; matches all datasets)
- **Pre-emphasis:** α = 0.97
- **Window:** Hann, 25 ms (400 samples at 16 kHz)
- **Hop:** 10 ms (160 samples)
- **FFT size:** 512
- **Mel filters:** 80 (matches common SOTA configs; sufficient for wake-word fidelity)
- **Log compression:** $\log(1 + 10 \cdot \text{mel})$ (stabilizes dynamic range better than pure log)
- **Input tensor shape:** (batch, 80, T_frames) where T_frames ≈ 98 for 1 s

Implement using `torchaudio.transforms.MelSpectrogram` for training and a hand-rolled NumPy version for benchmarking xRT on CPU. Do not ship torchaudio in the deployed model — too heavy for on-device.

### 3.2 VAD (Voice Activity Detection)

- **Model:** Silero-VAD v4 (MIT license, ~1 MB, <1 ms per 30 ms chunk on CPU)
- **Role:** Gate the expensive encoder. If no speech in the last 500 ms window, skip the encoder entirely. This is free FA reduction and free latency budget.
- **Parameters are NOT counted toward the <3M budget** (VAD is a universal preprocessor, not part of the KWS system proper). Still document it honestly.

### 3.3 Shared encoder — backbone choice

**Primary choice: BC-ResNet-8** (Broadcasted Residual Learning, Kim et al. 2021, from Samsung Research)

Why BC-ResNet:
- Samsung authors (judge recognition) ← don't underestimate this
- ~80K parameters for BC-ResNet-1, scaling up to BC-ResNet-8 at ~320K
- SOTA on Google Speech Commands v2 at a fraction of Transformer compute
- Separates 2D time-frequency into 1D frequency + broadcasted time operations — tailored to audio

**Backup choice: MatchboxNet** (NVIDIA, ~140K params) — in case BC-ResNet re-implementation has issues.

**Output:** Backbone produces a feature map of shape (B, C, F', T') which we pool (adaptive average pool over F' and T') into a single vector of dimension C (C ≈ 128–256 depending on BC-ResNet scale).

Parameter budget for backbone: target 900K–1.2M parameters. This leaves ~1.8M for the two heads and fusion.

### 3.4 Content head

- **Architecture:** 2-layer MLP with batch-norm + GELU, 256 → 128 → 128
- **Output:** $z_c \in \mathbb{R}^{128}$, L2-normalized
- **Parameter count:** ~50K
- **Purpose:** Capture what phonetic content was spoken, invariant to speaker.

### 3.5 Speaker head

- **Architecture:** 2-layer MLP with batch-norm + GELU, 256 → 128 → 128
- **Output:** $z_s \in \mathbb{R}^{128}$, L2-normalized
- **Parameter count:** ~50K
- **Purpose:** Capture who spoke it, invariant to word.

### 3.6 Disentanglement mechanism (novelty cornerstone)

Three stacked mechanisms ensure the two heads don't leak information into each other. This is the most important part technically — emphasize it in the blueprint and report.

#### 3.6.1 Orthogonality loss

For each batch, compute cross-covariance between $z_c$ and $z_s$ and penalize off-diagonal mass:

$$\mathcal{L}_{ortho} = \left\| \frac{1}{B} Z_c^\top Z_s \right\|_F^2$$

where $Z_c, Z_s$ are $(B \times 128)$ batch matrices. Encourages content and speaker subspaces to be statistically uncorrelated.

#### 3.6.2 Adversarial branch (gradient reversal)

Two small auxiliary adversaries:

- **Speaker-from-content adversary:** Classifier $A_s$ tries to predict speaker from $z_c$. We place a **gradient reversal layer** before it, so the content head learns to *remove* speaker information from $z_c$.
- **Content-from-speaker adversary:** Classifier $A_c$ tries to predict word class from $z_s$. Same gradient reversal — forces speaker head to discard content.

Adversarial loss:

$$\mathcal{L}_{adv} = \lambda_s \cdot \text{CE}(A_s(\text{GRL}(z_c)), y_{speaker}) + \lambda_c \cdot \text{CE}(A_c(\text{GRL}(z_s)), y_{word})$$

Start with $\lambda_s = \lambda_c = 0$, ramp to 0.1 over first 10 epochs (standard DANN-style schedule).

#### 3.6.3 Supervised contrastive objectives

- **Content contrastive** (on $z_c$): pairs sharing the same word class are positives, different words are negatives. Ignore speaker identity.
- **Speaker contrastive** (on $z_s$): pairs sharing the same speaker are positives, different speakers are negatives. Ignore word content.

Using SupCon loss (Khosla et al. 2020):

$$\mathcal{L}_{sup}^{c} = -\sum_{i \in I} \frac{1}{|P_c(i)|} \sum_{p \in P_c(i)} \log \frac{\exp(z_c^i \cdot z_c^p / \tau)}{\sum_{a \in A(i)} \exp(z_c^i \cdot z_c^a / \tau)}$$

where $P_c(i)$ is the set of samples with the same word as $i$, $A(i)$ is all other samples in the batch, $\tau = 0.1$.

Analogous definition for $\mathcal{L}_{sup}^{s}$.

### 3.7 Gated fusion decision layer

Given:
- $s_c = \cos(z_c^{\text{query}}, \bar{C}_w)$ where $\bar{C}_w$ is mean of enrolled keyword templates
- $s_s = \cos(z_s^{\text{query}}, \bar{S}_u)$ where $\bar{S}_u$ is mean of enrolled speaker utterances

**Naive fusion** (baseline, also reported): $s = s_c \cdot s_s$ or $s = \min(s_c, s_s)$

**Our learned fusion:**

```
fusion_input = [s_c, s_s, s_c*s_s, s_c**2, s_s**2, |s_c - s_s|]   # 6-dim
logit = MLP_fusion(fusion_input)                                   # 6 → 16 → 1
p = sigmoid(logit)
decide = p > τ (with hysteresis)
```

The fusion MLP is ~400 parameters. It learns the "AND-like" nonlinearity such that the output is only high when **both** similarities are above threshold. Crucially, it can learn an asymmetric decision boundary — e.g., being stricter on speaker similarity when content is a close phonetic match.

### 3.8 Hysteresis and smoothing

- **Streaming stride:** 100 ms
- **Smoothing window:** 3 frames (300 ms), take max of posterior
- **Hysteresis:** rising threshold $\tau_{on} = 0.75$, falling threshold $\tau_{off} = 0.45$
- **Refractory period:** 1.5 s after a positive detection to prevent double-fires

These values are tuned on a held-out dev set, not the test set. Document this in the report.

### 3.9 Enrollment flow

1. User says keyword 3–5 times (record each in a quiet-ish environment, no strict requirement)
2. For each recording: VAD-trim → log-mel → encoder → (z_c, z_s)
3. Compute mean content template $\bar{C}_w$ (L2-normalized after averaging) and mean speaker template $\bar{S}_u$
4. **TTS augmentation (novelty):** Feed the keyword text to an open TTS model (Parler-TTS, Apache-2.0) to generate 10–20 additional keyword utterances in diverse synthetic voices. Extract only the **content embeddings** of these, add to $\bar{C}_w$ computation. Do NOT add them to speaker templates (different voices).
5. **Adaptive threshold calibration:** Optionally, ask user for one "cancel" utterance of something *different* to calibrate the operating threshold.
6. Store $\bar{C}_w$, $\bar{S}_u$, user-specific $\tau_{on}, \tau_{off}$ as the enrollment artifact (~1 KB JSON + two float32 vectors).

---

## 4. Data Strategy

### 4.1 Datasets (all open, all verified)

| Dataset | Role | Size | License | Notes |
|---|---|---|---|---|
| LibriPhrase | Primary (keyword+speaker pairs) | ~16K hours derived | MIT-compatible | Built on LibriSpeech |
| Google Speech Commands v2 | Content pretraining, 35-class | ~30 hours | CC-BY-4.0 | 105K utterances |
| VoxCeleb1 & 2 | Speaker diversity | 2,800 hours, 7,000+ speakers | CC-BY-4.0 | Required for speaker head |
| LibriSpeech | General speech backbone pretraining (optional) | 960 hours | CC-BY-4.0 | |
| MUSAN | Noise augmentation | 109 hours | CC-BY-4.0 | Speech, music, babble, noise |
| BUT Reverb DB + OpenSLR-28 | Room impulse responses | ~1000 RIRs | CC-BY-4.0 / CC-0 | For distance simulation |
| Parler-TTS | TTS augmentation | Model only | Apache-2.0 | Custom keyword synthesis |

**Critical:** Before Phase 1 submission, you must personally confirm each license is Apache-2.0 compatible. Write the confirmed license next to each item in slide 6. Samsung rejects teams for license negligence.

### 4.2 Train/dev/test splits

- **Train:** All data EXCEPT held-out test speakers and test words
- **Dev:** 5 speakers × 20 keywords held out, clean
- **Test-clean:** 10 speakers × 30 keywords held out, never seen in training
- **Test-noisy:** Same test-clean audio, augmented at 8 SNR points: {-5, 0, 5, 10, 15, 20, 25, 30 dB}, 3 noise types: {crowd babble, traffic, music}
- **Test-distance:** Same test-clean audio convolved with RIRs at 5 distance buckets: {0.5, 1, 2, 3, 5 m}
- **Long-background eval:** 10 hours of continuous ambient audio (no target word, no target speaker) for measuring FA rate. Sourced from VoxCeleb + MUSAN mixed.

**Speaker disjoint-ness is sacred.** If the same speaker appears in both train and test, your numbers are meaningless. Judges have seen this mistake a thousand times — they will catch it.

### 4.3 On-the-fly augmentation pipeline

During training, every sample is augmented online:

```
x_clean → [random gain ±6 dB]
       → [random RIR from pool, p=0.6]
       → [random noise from MUSAN at SNR uniform(-5, 30), p=0.8]
       → [random time shift ±50 ms]
       → [SpecAugment: 2 freq masks width ≤ 15, 2 time masks width ≤ 20, p=0.5]
       → x_aug
```

This is the single biggest driver of noise/distance robustness. Nail it.

### 4.4 Hard negative mining

**Phonetic hard negatives** (for content): Use a G2P (grapheme-to-phoneme) tool (e.g., `g2p_en`, MIT license) to convert every word to phonemes. Compute pairwise phoneme edit distance. For each target keyword, select the 20 closest non-identical words from the vocabulary as hard negatives. During training, oversample these.

**Voice-similar hard negatives** (for speaker): Precompute speaker embeddings on VoxCeleb using a baseline ECAPA-TDNN (or our model after epoch 1). Cluster speakers into 50 clusters. Within the same cluster, treat other speakers as hard negatives. This catches siblings, same-gender same-age speakers.

Both mining schemes activate after epoch 5 (need a partially-trained model for speaker mining).

### 4.5 TTS-augmented keyword synthesis (novelty detail)

For each keyword in the training vocab:
- Synthesize 50 variations using Parler-TTS with prompts like "a young female speaker says 'samsara' clearly", "an elderly male speaker mumbles 'samsara'", "'samsara' spoken with British accent", etc.
- Adds acoustic/prosodic diversity without needing more human speakers
- Only used for **content head** training (these are not real speakers, so they'd confuse speaker supervision)

Verify TTS output quality: manually listen to 100 random samples, discard any that are garbled. Document this in the report.

---

## 5. Training Pipeline (Staged Curriculum)

### Stage 1 — Backbone pretraining (3 days)

- **Data:** Google Speech Commands v2, 35-class classification
- **Task:** Standard softmax cross-entropy, predict word label
- **Purpose:** Give backbone a solid phonetic prior
- **Epochs:** 40
- **Batch size:** 256
- **Optimizer:** AdamW, lr=3e-3, cosine schedule, warmup 1000 steps
- **Expected backbone accuracy on GSC-v2:** ≥ 96% (this is a sanity check — if you don't hit this, the backbone is broken)

### Stage 2 — Dual-head joint training (5 days)

- **Data:** LibriPhrase + VoxCeleb2 (balanced sampler: 50/50)
- **Losses active:**
  - $\mathcal{L}_{sup}^{c}$ (content SupCon)
  - $\mathcal{L}_{sup}^{s}$ (speaker SupCon)
  - Speaker classification CE on $z_s$ (softmax over speakers, train-only)
  - Word classification CE on $z_c$ (softmax over word vocab, train-only)
- **Losses NOT yet active:** orthogonality, adversarial (zero weight)
- **Total loss:** $\mathcal{L}_{s2} = \mathcal{L}_{sup}^{c} + \mathcal{L}_{sup}^{s} + 0.5 \cdot (\text{CE}_c + \text{CE}_s)$
- **Epochs:** 30
- **Batch:** 128, with class-balanced sampling (ensure 16 word classes × 4 speakers per word per batch where possible)

### Stage 3 — Disentanglement turn-on (2 days)

- Activate orthogonality loss with weight 0.1
- Activate adversarial branch with weight ramping from 0 → 0.1 over 5 epochs
- **Sanity check:** speaker accuracy on $z_c$ via a probe classifier should *drop* over this stage (target: from ~60% → ~15%, indicating speaker info successfully removed). Analogously, word accuracy on $z_s$ should drop.
- **Epochs:** 10

### Stage 4 — Noise/distance robustness (3 days)

- Turn on full augmentation pipeline (SNR −5 to 30, RIR probability 0.8)
- Hard negative mining activated (phonetic + speaker)
- Same losses, fine-tune with lower lr (1e-4)
- **Epochs:** 15

### Stage 5 — Fusion head training (1 day)

- Freeze backbone + heads
- Train the gated fusion MLP on held-out keyword+speaker pairs
- Binary classification: (z_c^query, z_s^query, enrolled templates) → accept/reject
- Includes all four quadrants Q1–Q4, with Q2 and Q3 oversampled 2x

### Stage 6 — Quantization-aware fine-tune (1 day)

- Insert fake-quant observers
- Fine-tune 2 epochs with INT8 simulation
- Export to ONNX, then ONNX INT8

**Total training budget: ~15 days on a single RTX 3090 / A10 / T4×2.** Fits in the 6-week window with buffer.

### 5.1 Expected training curves you should see

If you don't see these, something is broken:

- Stage 1: loss from ~3.5 → ~0.15 over 40 epochs
- Stage 2: content SupCon from ~4 → ~0.8, speaker SupCon from ~4 → ~1.0
- Stage 3: speaker-from-content probe accuracy drops from 55–65% to 15–20%
- Stage 4: noisy dev TA climbs from ~70% → ~90%

### 5.2 Hyperparameter table (freeze this)

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Base LR | 3e-3 (stage 1-2), 1e-3 (stage 3), 1e-4 (stage 4) |
| Weight decay | 1e-4 |
| LR schedule | Cosine with warmup (5% of steps) |
| Batch size | 256 / 128 / 128 / 64 / 256 / 128 (per stage) |
| Mixed precision | FP16 autocast |
| Gradient clipping | 1.0 |
| SupCon temp τ | 0.1 |
| Orthogonality λ | 0.1 |
| Adversarial λ | 0.1 (ramped) |
| SpecAugment prob | 0.5 (stage 2+), 0.8 (stage 4) |
| Seed | 42 (also run 1337, 2024 for variance) |

---

## 6. Evaluation Protocol (Maps Directly to KPIs)

Every number in the report ties to one of these evaluations. No ad-hoc demo-only numbers.

### 6.1 Clean TA evaluation

- Dataset: Test-clean (see §4.2)
- Protocol: For each (user, keyword) pair, enroll with 3 random utterances, evaluate on 20 remaining utterances. Averaged over 50 (user, keyword) pairs.
- Report: TA, FRR, and 95% CI via bootstrap.
- **Pass bar: ≥ 99%.** If you're below, don't ship.

### 6.2 Noisy TA evaluation

- Dataset: Test-noisy at 8 SNR points × 3 noise types = 24 conditions
- Report a table of TA per condition, plus macro-average
- **Pass bar: ≥ 90% averaged across all 24 conditions**, and ≥ 80% at the hardest corner (−5 dB crowd babble)

### 6.3 FA rate evaluation

- Dataset: 10 hours of continuous background (no target keyword, no target speaker)
- Run streaming inference at 100 ms stride; count positive detections
- Divide by 10 hours = FA/hour
- **Pass bar: < 1 FA/hr.** Also report at 24-hr continuous if you have time.

### 6.4 Distance evaluation

- RIR-convolve test-clean at 5 distances
- Report TA per distance
- Target: ≥ 85% even at 5 m (stretch, not hard requirement)

### 6.5 Speaker rejection (Q2) evaluation

- For each (user, keyword) pair, have 10 different speakers say the same keyword
- Measure FA on these imposter utterances
- **This is your novelty money shot. Target: Q2 FA < 5%.** Baseline KWS systems without speaker conditioning will score ~90%+ on Q2 (they accept anyone). We should crush them here.

### 6.6 Phonetic confusion (Q3) evaluation

- For each keyword, generate phonetically similar words via G2P edit distance < 3
- Have the user say these confusable words
- Measure FA
- Target: Q3 FA < 10%

### 6.7 xRT benchmark

- Hardware: Intel Core i5-8250U (common laptop CPU baseline) AND Raspberry Pi 4 (edge device proxy for phone CPU)
- Metric: (inference wall-time) / (audio duration)
- Report ONNX FP32, ONNX INT8
- **Pass bar: xRT < 0.2 on laptop, < 0.4 on RPi4 with INT8**

### 6.8 Parameter count audit

- Script: `count_params.py` — walks the model, prints per-module parameter counts, flags anything over budget
- Report: total params, trainable params, params in each head/backbone/fusion
- **Pass bar: < 3M total.**

### 6.9 Ablation studies (REQUIRED for the report)

Run each of these and put them in a table. Judges love ablations — they prove you understand what each piece contributes.

| Ablation | Expected effect |
|---|---|
| No speaker head (content-only) | Q2 FA jumps to ~60% (proves speaker head matters) |
| No content head (speaker-only) | Q3 FA jumps to ~50% |
| No orthogonality loss | ~2% TA drop noisy |
| No adversarial branch | ~3% TA drop noisy, probe accuracy higher |
| No TTS augmentation | Q3 FA jumps ~5% (less phonetic coverage) |
| Multiplicative fusion instead of learned gate | ~1.5% TA drop |
| 1 enrollment sample vs. 5 | Quantifies enrollment sensitivity |

### 6.10 Variance reporting

Run every main experiment on 3 seeds. Report mean ± std. A single-seed result from a hackathon team is a red flag to serious reviewers.

---

## 7. Deployment & Optimization

### 7.1 Export pipeline

```
PyTorch (.pt) 
    ↓ [torch.onnx.export with opset 17]
ONNX FP32 (.onnx)
    ↓ [onnxruntime.quantization.quantize_dynamic]
ONNX INT8 (.onnx)
    ↓ [optional: optimize with onnxruntime.transformers.optimizer]
Deployment artifact (.onnx, ~1.5 MB)
```

### 7.2 On-device inference loop (reference Python)

```python
# Pseudocode — real implementation in deployment/inference.py
import onnxruntime as ort
session = ort.InferenceSession("solospeak.onnx", providers=['CPUExecutionProvider'])

ring_buffer = RingBuffer(1.5 * 16000)
detector = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory=1.5)

while True:
    chunk = mic.read(160 * 16)  # 160 ms of audio
    ring_buffer.push(chunk)
    
    if not vad(chunk): 
        continue  # skip when no speech
    
    window = ring_buffer.last(1.0 * 16000)  # last 1 s
    mel = logmel(window)                     # (80, 98)
    z_c, z_s = session.run(None, {"mel": mel[None]})
    
    s_c = cosine(z_c, template_C_w)
    s_s = cosine(z_s, template_S_u)
    p = fusion(s_c, s_s)
    
    if detector.step(p):
        trigger_action()
```

### 7.3 Binary size & memory target

- Model: < 3 MB (INT8 ONNX)
- Runtime: onnxruntime-mobile, ~8 MB
- Peak RAM during inference: < 30 MB
- Enrollment artifact per user: ~1 KB

### 7.4 Optional: Android demo app

Time permitting (week 6), build a minimal Android app using ONNX Runtime Mobile + Kotlin. Even a clunky version that just shows "DETECTED" in a TextView is worth 10x the polish of another evaluation chart. Judges remember the phone that lit up on stage.

---

## 7.5 Scalability Architecture

This section is explicitly required by deep-tech hackathon judges. It answers: "what happens when this system grows beyond one user on one device?" SoloSpeak's architecture has unusually strong scalability properties — most of them emerge directly from the disentanglement design — but they are only valuable if articulated clearly.

### 7.5.1 User Scaling — O(1) Per New User, No Retraining

This is SoloSpeak's most important scalability property and must be stated explicitly.

**The mechanism:**

Each enrolled user is represented by exactly two vectors and one scalar stored in a tiny per-user profile:

```
user_profile = {
    "C_w":  np.ndarray(128,),   # mean content template — 512 bytes (FP32)
    "S_u":  np.ndarray(128,),   # mean speaker template — 512 bytes (FP32)
    "tau":  float,              # per-user decision threshold — 4 bytes
    "keyword_text": str,        # e.g. "hey prism" — ~20 bytes
}
# Total per user: ~1.1 KB
```

**What does NOT scale with users:**
- The BC-ResNet-8 backbone — **shared** across all users, loaded once (~4 MB INT8)
- The content head — **shared**, loaded once (~0.2 MB)
- The speaker head — **shared**, loaded once (~0.2 MB)
- The gated fusion MLP — **shared**, loaded once (~0.002 MB)

**What scales linearly:**
- Template storage: 1.1 KB × N users
- At 1,000 users: ~1.1 MB of profiles
- At 1,000,000 users: ~1.1 GB of profiles (but this is fleet storage, not device storage — each device stores only its own users)

**On a single device (e.g. Galaxy S25):**
- Realistic household: 1–6 users, 1–3 keywords each
- Storage: ~7–20 KB of templates
- RAM at inference: shared model (~12 MB peak) + all active templates (~20 KB) = ~12 MB, unchanged

**Enrollment of a new user never touches the model weights.** The backbone, heads, and fusion MLP remain frozen. A new user runs 3–5 enrollment utterances through the existing encoder, computes mean templates, calibrates threshold on one rejection sample, saves ~1.1 KB. This takes approximately 30 seconds of wall-clock time.

**Contrast with competing approaches:**
- Speaker-dependent acoustic models (e.g. per-user fine-tuned Bixby): require hours of retraining, GB of compute, centralized servers. Not on-device.
- Cascade KWS + SV (e.g. x-vectors): speaker model is separate, enrollment requires more utterances, and the two models cannot share learned representations.

### 7.5.2 Keyword Scaling — Any UTF-8 Phrase, Zero Retraining

A user can enroll any custom keyword phrase without any change to the model. The content head produces embeddings in a universal phonetic-semantic space learned over the full LibriPhrase vocabulary. A new keyword is just a new enrollment sequence.

**The mechanism:**

```
new_keyword_enrollment(keyword_text: str, recordings: List[np.ndarray]) -> ContentTemplate:
    # Step 1: Optional TTS augmentation
    tts_variants = parler_tts.synthesize_variants(keyword_text, n=10)  # only at enrollment
    all_clips = recordings + tts_variants  # 3-5 real + 10 TTS = 13-15 total

    # Step 2: Embed all clips through shared encoder + content head
    embeddings = [content_head(encoder(logmel(clip))) for clip in all_clips]

    # Step 3: Mean template
    C_w = np.mean(embeddings, axis=0)
    C_w /= np.linalg.norm(C_w)  # L2 normalize
    return C_w
```

**Phonetic coverage without 50+ samples:**
The TTS augmentation (Parler-TTS, Apache-2.0) synthesises acoustic variants of the keyword text across speaking rates, prosodic patterns, and microphone conditions. This means:
- 3 real recordings + 10–12 TTS variants = robust template centroid
- Template covers the manifold of how the keyword sounds, not just 3 specific instances
- False rejection from within-user variation is suppressed without requiring more user effort

**Hard negative mining by keyword:**
At inference, phonetically similar words (e.g. "hey prism" vs "hey prison") are the Q3 confusion class. The content head is trained with SupCon loss that pulls same-keyword embeddings together and pushes phonetically similar but distinct keywords apart. The g2p_en library converts the keyword text to a phoneme sequence, which drives hard-negative selection during training — so the content manifold is already shaped for this keyword's phonetic neighborhood even before the user enrolls it.

**Scaling to multiple keywords per user:**
A user can enroll N keywords by maintaining N separate content templates. At inference, compute cosine similarity against all N templates and take max(s_c). The speaker template S_u is shared across all keywords for the same user — it only needs to be computed once.

```
multi_keyword_inference(mel, user_profile):
    z_c, z_s = encoder(mel)
    s_s = cosine(z_s, user_profile["S_u"])
    s_c = max(cosine(z_c, C_w) for C_w in user_profile["keyword_templates"])
    p = fusion(s_c, s_s)
    return p > user_profile["tau"]
```

Storage cost: 1 additional content template (512 bytes) per keyword. For a user with 5 keywords: ~3 KB total.

### 7.5.3 Backbone Scalability — The BC-ResNet Ladder

BC-ResNet (Broadcasted Residual Learning, Kim et al. 2021, Samsung Research) is a parametric family. SoloSpeak is designed so the backbone is a drop-in swap — the dual-head architecture, loss functions, enrollment flow, and inference loop are all backbone-agnostic.

| Backbone | Approx. Params | Expected GSC-v2 Acc | Target Deployment | xRT (ARM) |
|---|---|---|---|---|
| BC-ResNet-1 | ~80 K | ~92% | Galaxy Watch / Earbuds | <0.04 |
| BC-ResNet-5 | ~500 K | ~96% | Galaxy Buds Pro | <0.06 |
| **BC-ResNet-8** | **~1.0 M** | **~98%** | **Flagship phone (default)** | **<0.08** |
| BC-ResNet-10 | ~1.5 M | ~98.5% | High-end phone / tablet | <0.12 |
| BC-ResNet-16 | ~3.0 M | ~99%+ | Server-side / fallback | <0.20 |

The dual-head dimensions (128) and gated fusion architecture remain fixed across all backbone variants. Only the backbone is swapped. This means Samsung can ship a single trained model family from one training pipeline, targeting every product tier from a 50 mW earbud to a flagship Galaxy S.

**Fallback during training:** If BC-ResNet-8 fails to converge (probe accuracy does not reach target after Stage 3), fall back to MatchboxNet (NVIDIA, Apache-2.0, ~140 K params). Same dual-head wrapper applies.

### 7.5.4 Multi-User Per Device — SmartThings and Shared Speakers

A shared device (SmartThings Hub, Samsung TV, smart refrigerator) may serve N household members, each with their own keyword and speaker template.

**Architecture for multi-user simultaneous listening:**

```
class MultiUserDetector:
    def __init__(self, user_profiles: Dict[str, UserProfile]):
        self.profiles = user_profiles  # {user_id: profile}
        self.model = load_onnx("solospeak.onnx")

    def step(self, mel: np.ndarray) -> Optional[str]:
        z_c, z_s = self.model.run(None, {"mel": mel[None]})  # ONE forward pass

        for user_id, profile in self.profiles.items():
            s_c = cosine(z_c, profile.C_w)
            s_s = cosine(z_s, profile.S_u)
            p = fusion(s_c, s_s)
            if p > profile.tau:
                return user_id  # first match wins (or collect all above threshold)
        return None
```

**Critical point:** The expensive operation — the BC-ResNet-8 forward pass — runs **exactly once** regardless of how many users are registered. All per-user comparisons are simple cosine dot products (128-dimensional, essentially free). For a 6-member household with 2 keywords each: 12 cosine operations after one forward pass. The incremental cost per additional user is negligible.

**Conflict resolution (two users speak simultaneously):**
- Both scores may exceed threshold
- Resolution: highest fusion probability wins
- Or: trigger both and route to both users' assistant contexts (device policy decision, not model decision)

**Privacy architecture for shared devices:**
- All templates stored locally on-device — never transmitted
- No cloud speaker-ID service required
- Each user's template is opaque (cannot reconstruct the original voice from a 128-dimensional L2-normalized vector)
- Template deletion = single file delete — GDPR compliant by design

### 7.5.5 Fleet Deployment — Samsung Production Path

This section describes how SoloSpeak would be deployed at Samsung scale (millions of devices) as a real product upgrade.

**Step 1 — OTA model delivery:**

The SoloSpeak model is a single INT8 ONNX file (~4.5 MB for the shared encoder + heads + fusion, excluding VAD). This is delivered as a standard OTA software update via Samsung's existing firmware distribution infrastructure. It installs into a protected partition alongside the existing Bixby acoustic model.

```
Device partition layout (post-OTA):
/system/priv-app/SoloSpeak/
    solospeak_encoder.onnx      # ~4.5 MB  — shared, never changes
    silero_vad.onnx             # ~1.0 MB  — shared, never changes
    solospeak_runtime.apk       # ~2.0 MB  — JNI wrapper + enrollment UI

/data/user/<uid>/SoloSpeak/
    profile_<user_id>.bin       # ~1.1 KB per enrolled user — device-local, encrypted
```

Total OTA delta: ~7.5 MB. Comparable to a minor app update.

**Step 2 — First-time enrollment (user-facing flow):**

```
Bixby Settings → "Personalize Wake Word"
    ↓
[Choose or type a custom wake phrase]
    ↓
[Record phrase 5 times — "Say 'Hey Prism' clearly"]
    ↓
[Optional: TTS variants generated silently in background]
    ↓
[Calibration: "Say something that is NOT your wake word"]
    ↓
[Profile saved locally — enrollment complete — ~30 seconds total]
```

No account creation, no server round-trip, no voice data uploaded. The entire enrollment computation runs locally in ~5 seconds of CPU time on a Galaxy S25.

**Step 3 — Continuous on-device inference:**

Once enrolled, SoloSpeak runs as a low-power always-on listener managed by the Samsung audio HAL. It operates in the same power domain as the existing Bixby wake-word engine:

```
Audio pipeline:
Microphone DSP → VAD (Silero, ~1 MB, always on)
                ↓ (only when speech detected)
           BC-ResNet-8 inference (~0.08 xRT on ARM Cortex-A78)
                ↓
           Per-user cosine + fusion (~0.001 ms, negligible)
                ↓
     Wake event → Android AudioManager → Bixby / Assistant
```

Power consumption estimate:
- VAD alone: ~0.5 mW (runs on DSP co-processor)
- Full BC-ResNet-8 inference: ~2–4 mW (runs on main CPU at ~60 ms cadence when speech is active)
- Duty cycle: assuming 10% speech-active time → effective power ~0.7–0.9 mW
- Comparable to existing Bixby always-on mode (~0.8–1.2 mW reported in Samsung Knox documentation)

**Step 4 — Re-enrollment and template management:**

A user can re-enroll at any time (voice changes, new keyword preference). Re-enrollment overwrites the existing template file. No model update required. Template management API:

```kotlin
// SoloSpeakManager.kt (Samsung System API)
SoloSpeakManager.enroll(userId, keyword, recordings)   // ~30 sec
SoloSpeakManager.deleteProfile(userId)                 // instant
SoloSpeakManager.listProfiles()                        // returns List<UserProfile>
SoloSpeakManager.updateThreshold(userId, tau)          // optional manual override
```

**Step 5 — Server-side role (minimal, opt-in):**

For premium features only (optional, privacy-preserving):
- **Model update delivery:** Samsung can push improved encoder weights via OTA without requiring re-enrollment — new backbone maps to the same embedding space if fine-tuned with distillation from the existing checkpoint
- **Aggregate quality metrics:** Anonymized, on-device computed metrics (TA rate, FA count) optionally uploaded as differential privacy noise-protected statistics to Samsung's model improvement pipeline
- **No voice data ever leaves the device** in the standard configuration

### 7.5.6 Horizontal Scaling: From One Device to the Samsung Ecosystem

| Product | Backbone | Model Size | RAM Budget | Users | Notes |
|---|---|---|---|---|---|
| Galaxy Watch 7 | BC-ResNet-1 | ~0.5 MB INT8 | ~6 MB | 1–2 | Single keyword, wrist-raise context |
| Galaxy Buds 3 Pro | BC-ResNet-5 | ~1.5 MB INT8 | ~8 MB | 1 | Per-earbud inference, no cloud hop |
| Galaxy S25 (phone) | BC-ResNet-8 | ~4.5 MB INT8 | ~12 MB | 1–6 | Full feature set, default deployment |
| SmartThings Hub | BC-ResNet-8 | ~4.5 MB INT8 | ~20 MB | 1–10 | Multi-user household, always-on |
| Samsung TV | BC-ResNet-10 | ~6 MB INT8 | ~30 MB | 1–6 | Living room distance (1–5 m) |
| Bixby Server (fallback) | BC-ResNet-16 | ~12 MB FP32 | unconstrained | unlimited | Cloud fallback for degraded audio |

The same training pipeline, same dual-head architecture, same enrollment flow, and same evaluation protocol apply to every row. The only variable is which BC-ResNet variant populates the encoder slot.

### 7.5.7 Scalability Summary (For Slide / Report Reference)

| Dimension | How It Scales | Complexity | Notes |
|---|---|---|---|
| New users (same device) | Add 1.1 KB template, zero retraining | O(1) storage, O(N) inference | N cosine ops after one forward pass |
| New keywords (same user) | Add 0.5 KB content template | O(K) storage, O(K) inference | Shared speaker template |
| Model accuracy | Swap backbone in the ladder | Same dual-head wrapper | BC-ResNet-1 to BC-ResNet-16 |
| Device tier | Backbone size × INT8 compression | ~0.5–12 MB | Single pipeline trains all variants |
| Fleet size | OTA delta ~7.5 MB | One-time per device | No per-device model customization |
| Re-enrollment | 30-second local operation | O(1) | Overwrites template file only |
| Privacy | All computation on-device | Zero cloud dependency | GDPR compliant by design |

---

## 7.6 Production Deployment Architecture — End-to-End

This section traces the complete path from a trained model checkpoint to a running production system on a Galaxy device. It is structured as a sequential pipeline of concrete, actionable steps.

### 7.6.1 Training Checkpoint → Deployment Artifact

```
Stage 6 output: solospeak_qat.pt  (PyTorch, INT8 QAT)
                        │
                        ▼
        torch.onnx.export(model, opset_version=17,
            input_names=["mel"],
            output_names=["z_c", "z_s"],
            dynamic_axes={"mel": {0: "batch"}})
                        │
                        ▼
        solospeak_fp32.onnx  (~18 MB)
                        │
                        ▼
        onnxruntime.quantization.quantize_static(
            model_input="solospeak_fp32.onnx",
            model_output="solospeak_int8.onnx",
            calibration_data_reader=GSCCalibrationReader(),
            quant_format=QuantFormat.QOperator,
            per_channel=True)
                        │
                        ▼
        solospeak_int8.onnx  (~4.5 MB)
                        │
                        ▼
        Verify: xRT < 0.08 on ARM Cortex-A78 (RPi 4 proxy)
        Verify: TA degradation < 0.3% vs FP32 baseline
        Verify: No operator unsupported by onnxruntime-mobile 1.18
                        │
                        ▼
        DEPLOYMENT ARTIFACT FROZEN ✓
```

### 7.6.2 Artifact Validation Gates (Before OTA)

Before any artifact is considered production-ready, it must pass all of the following automated gates:

```python
# deployment/validate_artifact.py
def validate(onnx_path: str, test_set: EvalSet) -> ValidationReport:
    checks = [
        check_filesize(onnx_path, max_mb=5.0),
        check_onnx_opset(onnx_path, min_opset=17),
        check_mobile_ops(onnx_path),          # no unsupported ops
        check_xrt(onnx_path, max_xrt=0.08),   # on ARM proxy
        check_ta_clean(onnx_path, test_set, min_ta=0.99),
        check_ta_noisy(onnx_path, test_set, min_ta=0.90),
        check_fa_rate(onnx_path, test_set, max_fa_per_hr=1.0),
        check_param_count(onnx_path, max_params=3_000_000),
        check_output_shapes(onnx_path, expected={"z_c":[1,128],"z_s":[1,128]}),
    ]
    return ValidationReport(checks)  # all must pass
```

Any single gate failure blocks the OTA. The artifact is re-trained or re-quantized.

### 7.6.3 OTA Package Structure

```
solospeak_ota_v1.0.0.zip
├── MANIFEST.json              # version, checksum, min Android API
├── solospeak_int8.onnx        # ~4.5 MB — encoder + heads + fusion
├── silero_vad_v4.onnx         # ~1.0 MB — VAD gate
├── solospeak_runtime.apk      # ~2.0 MB — JNI wrapper, enrollment UI
└── signature.p7s              # Samsung code-signing certificate

Total: ~7.5 MB
```

The OTA package is signed with Samsung's existing code-signing infrastructure (same process as One UI updates). Devices verify the signature before installation. The model file itself is integrity-checked via SHA-256 hash in MANIFEST.json before loading into ONNX Runtime.

### 7.6.4 On-Device Runtime Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Samsung Audio HAL                        │
│                                                             │
│  Microphone PDM  →  DSP co-processor                       │
│                         │                                  │
│                    ┌────▼────────┐                          │
│                    │ Silero-VAD  │  ~0.5 mW, always-on     │
│                    │  (DSP)      │                          │
│                    └────┬────────┘                          │
│                         │ speech detected                   │
│                         ▼                                  │
│                  ┌──────────────┐                           │
│                  │  Ring Buffer  │  1.5 s × 16 kHz PCM     │
│                  │  (shared RAM) │                          │
│                  └──────┬───────┘                           │
│                         │ 100 ms stride                     │
│                         ▼                                  │
│           ┌─────────────────────────────┐                   │
│           │   SoloSpeak ONNX Runtime    │  ~2–4 mW          │
│           │   (main CPU, big core)      │                   │
│           │                             │                   │
│           │  Log-Mel → BC-ResNet-8      │                   │
│           │  → Content Head  (z_c)      │                   │
│           │  → Speaker Head  (z_s)      │                   │
│           │  → Cosine × N users         │                   │
│           │  → Gated Fusion × N users   │                   │
│           │  → Hysteresis detector      │                   │
│           └──────┬──────────────────────┘                   │
│                  │ wake event (user_id, keyword)            │
│                  ▼                                          │
│         Android AudioManager                               │
│                  │                                          │
│         ┌────────▼──────────┐                               │
│         │   Intent Router   │                               │
│         │  (Bixby / Google  │                               │
│         │   Assistant / App)│                               │
│         └───────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

**Threading model:**
- VAD runs on DSP co-processor thread — never blocks main CPU
- ONNX inference runs on a dedicated background thread (priority: THREAD_PRIORITY_AUDIO)
- Wake event is posted to Android's main looper via Handler — no blocking on audio thread
- Ring buffer is accessed via lock-free circular queue (single producer, single consumer)

**Failure modes and recovery:**
```
ONNX Runtime crash       → restart inference thread, log to Samsung Knox
VAD false negative       → 100 ms stride continues regardless, VAD only reduces load
Template file corrupted  → prompt re-enrollment, do not degrade to speaker-agnostic mode
Model file checksum fail → refuse to load, fall back to legacy Bixby acoustic model
Battery saver mode       → increase stride to 200 ms, disable TTS augmentation at enrollment
```

### 7.6.5 Enrollment Service Architecture (Production)

```kotlin
// SoloSpeakEnrollmentService.kt
class SoloSpeakEnrollmentService : Service() {

    fun enroll(userId: String, keywordText: String,
               recordings: List<ByteArray>): EnrollmentResult {

        // 1. Validate recordings (length, SNR, clipping)
        val validated = recordings.filter { validateClip(it) }
        require(validated.size >= 3) { "Need at least 3 valid recordings" }

        // 2. Optional TTS augmentation (runs in background, ~5 sec)
        val ttsVariants = if (validated.size < 5) {
            TtsAugmentor.synthesize(keywordText, n = 10)
        } else emptyList()

        val allClips = validated + ttsVariants

        // 3. Embed through shared encoder (one forward pass per clip)
        val embeddings = allClips.map { clip ->
            val mel = LogMelExtractor.extract(clip)
            val (z_c, z_s) = onnxSession.run(mel)
            Pair(z_c, z_s)
        }

        // 4. Mean templates + L2 normalize
        val C_w = meanNormalize(embeddings.map { it.first })
        val S_u = meanNormalize(embeddings.map { it.second })

        // 5. Calibrate threshold on one rejection sample
        val rejectionClip = promptUserForRejectionSample()
        val (z_c_rej, z_s_rej) = onnxSession.run(LogMelExtractor.extract(rejectionClip))
        val p_rej = fusionMlp(cosine(z_c_rej, C_w), cosine(z_s_rej, S_u))
        val tau = 0.5f * (p_rej + 0.9f)  // midpoint between rejection score and high-confidence

        // 6. Persist profile (encrypted, device-local)
        val profile = UserProfile(userId, keywordText, C_w, S_u, tau)
        profileStore.save(userId, profile.encrypt(deviceKeyStore))

        return EnrollmentResult.SUCCESS
    }
}
```

### 7.6.6 Model Update Without Re-Enrollment

This is a critical production property: **enrolled users do not need to re-enroll when the model is updated.**

This is achievable because SoloSpeak uses cosine similarity in the embedding space, not a softmax classifier. Template vectors are representations in a metric space, not class indices.

**The mechanism (knowledge distillation during model update):**

When Samsung ships BC-ResNet-8 v1.1 (improved backbone, e.g. trained on more data or with improved noise augmentation), the update must preserve the embedding space geometry so that existing template vectors remain valid:

```python
# During v1.1 training: add embedding space distillation loss
L_distill = MSELoss(
    student_encoder(mel),      # v1.1 encoder output
    teacher_encoder(mel)       # v1.0 encoder output (frozen)
)
L_total = L_task + 0.3 * L_distill
```

With distillation, the new encoder maps audio to approximately the same embedding space as the old one. Enrolled template vectors from v1.0 remain valid for v1.1 with no re-enrollment required. The threshold may shift slightly — a silent background recalibration (without user interaction) can re-estimate tau on recent audio history.

If a major architecture change makes distillation insufficient (e.g. switching from BC-ResNet-8 to BC-ResNet-16), a one-time re-enrollment prompt is shown in Bixby settings, clearly explaining why. This is a last resort, not the standard update path.

### 7.6.7 Privacy and Compliance Architecture

**Data minimization:**
- During enrollment: audio clips are processed locally, templates saved, raw audio discarded immediately
- During inference: audio never recorded or buffered beyond the 1.5-second ring buffer
- No biometric data (voice prints) are transmitted to Samsung servers in the standard configuration
- Templates are 128-dimensional L2-normalized float vectors — they cannot be inverted to reconstruct audio

**Cryptographic storage:**
```kotlin
// Templates stored using Android Keystore — hardware-backed AES-256
val keyAlias = "solospeak_profile_key_$userId"
val keyStore = KeyStore.getInstance("AndroidKeyStore")
val encryptedProfile = AesCipher.encrypt(profile.serialize(), keyAlias)
// Encrypted blob stored at: /data/user/<uid>/SoloSpeak/profile_<userId>.enc
```

**Right to deletion:**
```kotlin
SoloSpeakManager.deleteProfile(userId)
// Effect: deletes profile_<userId>.enc, removes userId from active detector
// Completion: < 1 ms
// Reversibility: none — data is gone
```

**Regulatory compliance:**
- GDPR Article 17 (right to erasure): satisfied by instant profile deletion
- GDPR Article 25 (data protection by design): satisfied — no voice data leaves device
- CCPA: no sale or sharing of biometric data
- Samsung Knox Vault: profile files stored in Knox-protected partition on Galaxy S21 and later

---

## 7.7 Security & Adversarial Robustness

A wake-word system authenticates intent to a personal device. The attack surface matters. This section enumerates every class of attack SoloSpeak might face and the mitigation architecture for each.

### 7.7.1 Threat Model

**In-scope threats:**
1. **Replay attack** — attacker records the legitimate user saying the wake word, plays it back at the device
2. **Voice cloning** — attacker generates synthetic audio of the user's voice (e.g. via ElevenLabs, Respeecher, Vall-E) speaking the wake word
3. **Adversarial audio** — imperceptible perturbations added to non-user speech that cause false accept
4. **Phonetic impersonation** — another human mimics the user's voice and accent
5. **Cross-device template theft** — attacker extracts template files from a compromised device
6. **Model extraction** — attacker repeatedly queries the model to reconstruct it (for IP theft, not user harm)

**Out-of-scope threats (explicit non-goals):**
1. Adversarial attacks with physical-layer microphone access (device owner compromise)
2. Side-channel attacks on the DSP co-processor
3. Audio injection via ultrasonic DolphinAttack-style methods (handled by Samsung's existing mic DSP filtering)

### 7.7.2 Replay Attack Mitigation

Replay is the most practical real-world attack. A family member, a TV playing a recording, or an attacker with a Bluetooth speaker can replay the user's voice saying the wake word.

**Defense in depth:**

1. **Liveness via room acoustic variation** — Real user utterances have fresh room impulse response characteristics. Replays captured-and-played-back have *two* RIRs convolved (capture room + playback room + speaker distortion). A lightweight anti-spoofing head can be added post-speaker-head to detect this. Reference: ASVspoof 2021 replay subset; baseline LCNN achieves EER ~5% on replay detection with ~200K parameters. This is a stretch goal for Phase 2 if time permits.

2. **Replay cooldown** — If the same high-confidence wake is triggered twice within 30 seconds, the second trigger requires a secondary confirmation (follow-up voice prompt "Yes, I meant it"). This defeats opportunistic replay without harming normal UX.

3. **Contextual rejection** — The Bixby assistant already has context awareness (is the screen on? is the user near the device?). A wake event triggered with the device face-down on a charger at 3 AM gets higher scrutiny than one during active use.

### 7.7.3 Voice Cloning and Deepfake Audio

Modern TTS can clone a voice from 30 seconds of audio. A determined attacker with access to the user's voice samples (social media, phone recordings) can synthesize plausible wake-word audio.

**Why SoloSpeak resists this partially:**
- Speaker embedding is trained on 7,000+ VoxCeleb speakers — cloning systems trained on different speaker populations produce embeddings that often fall outside the natural speaker manifold
- L2-normalized speaker head output means small embedding perturbations cause large cosine score changes (the manifold is tight)

**What it does NOT solve:**
- A high-quality clone (e.g. ElevenLabs Professional) with 10+ minutes of training audio can produce speaker embeddings close enough to the template to fool a speaker head trained on natural speech alone.

**Planned mitigation (Phase 2, if time permits):**
- Add a small **anti-spoofing head** (~100 K params) trained on ASVspoof 2019 + 2021 LA (logical access, synthetic speech detection)
- Architecture: shares the BC-ResNet backbone; a third small head predicts real-vs-synthetic
- Decision: accept only if all three gates pass — content, speaker, AND liveness
- Cost: +0.1 M params, negligible latency

This is explicitly **not a blocker for Phase 1**. Listed here so Samsung judges see it is on the roadmap.

### 7.7.4 Adversarial Audio Perturbations

Research-level concern: can an attacker craft an audio signal that sounds like noise or an unrelated phrase to humans but fools the model into accepting as the wake word?

**Our posture:**
- **Defense by architectural mass:** A 2.1 M parameter model with three trained heads is more robust than a single softmax classifier. The attack surface for coordinated attack across content + speaker + fusion is narrower than for any one of them individually.
- **Input mel-spectrogram quantization** in INT8 deployment acts as a natural input defense — gradient-based attacks must survive float-to-int8 rounding
- **Randomized smoothing at inference** (Phase 2 stretch): dither the mel input with σ=0.01 Gaussian noise, ensemble 3 predictions. Costs 3x inference, activated only if a replay is suspected.

**Honest limitation acknowledgment:** We do not claim certified robustness to adversarial audio. No production wake-word system does. We will include a whitebox FGSM attack evaluation in the Phase 2 report to quantify the attack budget required.

### 7.7.5 Template Theft and Model Extraction

**Template theft:**
- Templates are encrypted at rest using Android Keystore AES-256 (hardware-backed on devices with a Secure Element)
- A compromised device exposes templates, but the 128-dim L2-normalized vector is not directly a voice — it cannot be played back or used without a compatible encoder
- However, a stolen template + a copy of the public SoloSpeak encoder does enable an attacker to synthesize audio that matches: threat-modeled as equivalent to voice-cloning attack — same mitigation (anti-spoofing head)

**Model extraction:**
- The model runs on-device, so the weights are technically recoverable by an attacker with root access
- Mitigation: treat the model as public (it is Apache-2.0 anyway — no IP at stake). Security must come from template secrecy, not model secrecy.
- This is consistent with Samsung Knox's security model: device compromise is a different trust domain from user account compromise

### 7.7.6 Security Testing Requirements (Phase 2 Deliverable)

| Test | Pass Criterion | Tool |
|---|---|---|
| Replay attack on 10 diverse playback speakers | < 20% success rate | Self-curated replay eval set |
| ElevenLabs clone of Phase 2 demo author | < 50% success rate | Manual, 20 clone attempts |
| FGSM adversarial audio, ε=0.01 | quantify required ε for 50% FAR | `adversarial_audio.py` |
| Template encryption verified at rest | file cannot be decrypted without Keystore access | Android security tests |
| Wake-word context-awareness rules | 100% compliance with Samsung UX guidelines | Integration tests |

---

## 7.8 Fairness, Accessibility, and Demographic Coverage

A speaker-personalized wake-word system that works perfectly for 25-year-old American English speakers and fails for everyone else is not a production system. Samsung ships in 80+ countries to billions of users with widely varying voices.

### 7.8.1 Known Biases in Training Data

VoxCeleb and LibriSpeech have documented demographic skew:

| Dimension | VoxCeleb 1+2 Distribution | Production Implication |
|---|---|---|
| Gender | ~55% male / ~45% female | Slight male skew — acceptable |
| Age | Predominantly 30–60 | Under-represents children and elderly |
| Accent | ~60% US English, ~20% UK, ~20% other | Strong US/UK bias |
| Language | English only | Does not cover Samsung's global markets |
| Speech disorders | ~0% represented | Completely uncovered |
| Audio quality | Celebrity interviews (studio-grade) | Mismatch with real consumer mics |

### 7.8.2 Mitigation Strategy

**Phase 1 (blueprint — no additional data collection):**
- State known biases openly in the report (Section 15 of this document)
- Stratified evaluation reported per-subgroup in Phase 2: gender × age × accent where data permits
- Do not claim production readiness for uncovered demographics

**Phase 2 (stretch if schedule allows):**
- Add **Common Voice** (Mozilla, CC0) dataset — 20,000+ hours across 100+ languages, explicitly demographic-balanced. Use for an additional fine-tuning stage (Stage 2b) focused on non-English + non-US-English speakers.
- Add a **child voice** evaluation subset: CSLU Kids Corpus (academic license) or a self-curated small set with parental consent. Report TA for ages 5–12 separately.
- Add a **speech disorder** evaluation subset: TORGO database (dysarthric speech, academic license). Report TA for disordered speech separately.

**Post-hackathon (Samsung deployment):**
- Samsung has existing multilingual speech data from Bixby training. SoloSpeak's dual-head architecture is language-agnostic at the backbone level — multilingual extension is a data problem, not an architecture change.

### 7.8.3 Accessibility Design

**Dysarthric / disordered speech users:**
- SoloSpeak's per-user calibration is actually an advantage here: the content template captures the user's specific pronunciation, not a canonical one
- A user who pronounces "Hey Prism" as "Hey Pwism" will have that pronunciation as their template — the system learns their voice, not a "standard" voice
- Documented in the product UX as "SoloSpeak adapts to how *you* say the word, not how anyone else says it"

**Deaf and hard-of-hearing users:**
- Out of scope for a voice interface; Samsung has separate text-based Bixby paths

**Elderly users:**
- Enrollment UX must support larger text, slower pace, voice-guided instructions
- Tested with 65+ beta group as part of Samsung internal UX research post-hackathon

**Children:**
- Explicit policy: SoloSpeak is not certified for users under 13 in the Phase 1/2 release
- Enrollment UI requires confirmation of user age; under-13 enrollment creates a "supervised profile" with parental controls (Samsung Kids integration)

### 7.8.4 EU AI Act Compliance

SoloSpeak is a biometric-adjacent system (voice identification for authentication). Under the EU AI Act (applicable from August 2026), this is Limited Risk — users must be informed that the system identifies their voice.

**Compliance checklist:**
- Transparency: enrollment UX clearly states "SoloSpeak will recognize your voice"
- Opt-in only: feature is off by default on all new devices
- Opt-out: one-tap disable in Settings
- Template portability: users can export their templates (e.g. to a new Samsung device) via Samsung Cloud encrypted backup
- Bias documentation: the Phase 2 report includes a "limitations by demographic" section explicitly

---

## 7.9 Audio Hardware Integration (The Messy Reality)

The architecture document so far assumes a clean 16 kHz mic input. Real consumer devices have an audio processing stack between the physical microphone and the ONNX runtime input. This section documents the interactions.

### 7.9.1 The Real Audio Pipeline on a Galaxy Device

```
Physical MEMS microphones (1–4 per device)
        ↓
Analog front-end, ADC (24-bit, 48 kHz typical)
        ↓
┌──────────────────────────────────────┐
│  DSP co-processor (Samsung Hexagon)  │
│                                      │
│  • Beamforming (if ≥ 2 mics)         │
│  • Acoustic Echo Cancellation (AEC)  │
│  • Automatic Gain Control (AGC)      │
│  • Noise Suppression (NS)            │
│  • Sample Rate Conversion → 16 kHz   │
└──────────────┬───────────────────────┘
               ↓
        Silero-VAD (DSP)
               ↓  (speech detected)
       BC-ResNet-8 (main CPU)
```

Every stage except the last two is **not under our control** — Samsung's audio HAL owns it.

### 7.9.2 Critical Interactions We Must Handle

**AGC interaction:**
AGC normalises loudness by applying dynamic gain. This means the mel spectrogram sees a level-normalized signal, not raw audio. Good: distance-invariance is partially handled by AGC. Bad: if AGC has ~500 ms attack time, a short keyword at the start of an utterance sees a different gain than later words.

**Mitigation:** Training augmentation already includes random gain ±6 dB per clip. Document that SoloSpeak expects AGC-processed input; at enrollment, verify the audio HAL is providing AGC'd audio via `AudioRecord.getAutomaticGainControl()`.

**AEC interaction:**
AEC suppresses audio that matches what the device itself is playing (music, notification sounds, other Bixby output). This means SoloSpeak does not accidentally trigger on its own wake-word prompts. Good.

**Mitigation:** Test the AEC path explicitly: play a recording of the user's wake word through the speaker, verify SoloSpeak does NOT trigger. This is a required Phase 2 integration test.

**Beamforming interaction:**
Multi-mic devices (Galaxy S25 has 3 mics) use beamforming to focus on the user's voice direction. The mel spectrogram sees already-beamformed audio, not any individual mic. Good: dramatic SNR improvement in far-field.

**Mitigation:** None required, but document in the Phase 2 report that SoloSpeak's noisy TA benefits materially from Samsung's beamforming — reported xRT/accuracy numbers are with beamforming enabled (realistic) and with beamforming disabled (worst-case).

**Clock drift and sample rate conversion:**
The audio HAL delivers 16 kHz samples but the internal clock may drift by ~50 ppm. Over a 1.5-second window this is negligible (~75 µs = 1.2 samples). Ignore.

**Sample rate mismatches:**
If for any reason the audio HAL delivers 8 kHz (low-power mode, Bluetooth SCO), SoloSpeak must either:
- Upsample to 16 kHz (acceptable; adds <0.5 ms latency)
- Refuse to process (return `SamplingRateError`, fall back to legacy wake-word)

### 7.9.3 Required Audio HAL Tests

| Test | Input | Expected Behaviour |
|---|---|---|
| Clean 16 kHz AGC'd input | 5 m distance, quiet room | TA ≥ 99% |
| AEC active during music playback | wake word said over music | TA ≥ 90%, no self-trigger from music content |
| Beamforming off (single mic) | 3 m distance, noisy room | TA ≥ 85%, graceful degradation |
| Bluetooth SCO 8 kHz input | phone call wake attempt | Refuse or upsample — no crash |
| Mic mute toggled | any | Wake event suppressed, no error |
| DSP co-processor unavailable | low battery mode | Fall back to CPU-only VAD + inference |

---

## 7.10 Monitoring, Observability, and Rollback

What happens after the model ships? A production system without field observability is blind. A production system without rollback is brittle.

### 7.10.1 On-Device Quality Metrics (Privacy-Preserving)

Every SoloSpeak installation collects local telemetry that can be anonymised and aggregated (opt-in only, default off):

```kotlin
data class SoloSpeakLocalMetrics(
    val wakeEventsCount: Int,            // total wakes in last 24h
    val wakeEventsFollowedByCommand: Int, // TP proxy
    val wakeEventsCancelled: Int,         // user said "never mind" → likely FP
    val enrollmentRetries: Int,           // bad enrollment flow signal
    val averageInferenceLatencyMs: Float,
    val modelVersion: String,
    val backboneVariant: String,
    val deviceTier: String,               // "flagship", "midrange", "wearable"
    // No audio data, no timestamps precise enough to identify a user
)
```

Key principle: **no raw audio leaves the device, ever.** Only derived statistics.

### 7.10.2 Differential Privacy for Fleet Statistics

Aggregated metrics from millions of devices enable Samsung to detect quality regressions. To prevent individual user re-identification, apply differential privacy:

```
Local DP budget: ε = 1.0 per device per week
Mechanism: Laplace noise added to each metric before upload
Aggregation: Samsung cloud averages over ≥ 1000 devices minimum before any value is usable
```

Output: Samsung's fleet dashboard shows, with DP guarantees:
- 7-day rolling FA rate (median, p95) by device tier
- Enrollment success rate by locale
- Inference latency percentiles
- Model version adoption curve

### 7.10.3 Rollback Architecture

**Ability to roll back is non-negotiable for any field-deployed model.**

```
Device carries TWO model slots:
/system/priv-app/SoloSpeak/
    current/solospeak_int8.onnx    # active model
    previous/solospeak_int8.onnx   # previous OTA version, kept for rollback
```

**Rollback triggers (any ONE):**
- Fleet dashboard detects FA rate > 2× baseline for any device tier over 48 hours
- Fleet dashboard detects enrollment success rate drop > 10 percentage points
- Crash rate in SoloSpeakManager > 0.01% of sessions
- Samsung Knox reports any new security finding against the runtime

**Rollback procedure:**
1. Samsung pushes an emergency OTA "SoloSpeak v1.2.1 rollback"
2. Package size: ~100 KB — just a manifest update pointing to `previous/` as active
3. Device-side: atomic swap of `current/` and `previous/` symlinks, restart SoloSpeak service
4. Templates remain valid across rollback (embedding space preserved by distillation, Section 7.6.6)
5. Total rollback time from detection to 99% fleet coverage: ~72 hours

### 7.10.4 Canary Release Strategy

New model versions never roll out to 100% of the fleet at once.

```
Week 1: 0.1% canary — 1 in 1000 devices get the new model
        → if metrics stable after 72 hours, proceed
Week 2: 1% — broader canary across all tiers and locales
        → if stable, proceed
Week 3: 10% — full regional coverage
Week 4: 100% — everyone
```

At each gate, the differential privacy-protected fleet metrics must show no regression beyond pre-defined thresholds on: FA rate, TA rate (via command-follow-through proxy), inference latency, enrollment success, crash rate.

### 7.10.5 A/B Testing Infrastructure

For evaluating architectural changes (e.g. BC-ResNet-8 vs BC-ResNet-10), Samsung can run split tests:

```
Population: 10,000 opt-in beta users, demographically balanced
Split: 50/50 random assignment by device ID hash
Duration: 4 weeks minimum
Metrics: FA rate, TA proxy, user-reported satisfaction (Bixby settings 👍/👎)
Analysis: difference-in-differences with 95% confidence intervals
Decision threshold: ≥1% improvement on primary metric, no regression on secondary
```

---

## 7.11 Internationalization and Multilingual Support

SoloSpeak ships globally. The architecture must support languages beyond English without per-language retraining.

### 7.11.1 What Is Language-Agnostic

- **Mel spectrogram extraction** — purely signal-processing, no language assumption
- **BC-ResNet backbone** — convolutions over mel features, no language assumption
- **Speaker head** — voice timbre is language-independent (VoxCeleb has 100+ languages actually)
- **Gated fusion MLP** — operates on similarity scores, not linguistic content

### 7.11.2 What Is Language-Dependent

- **Content head training data** — LibriPhrase is English-only
- **g2p_en phonetic mining** — English grapheme-to-phoneme
- **Parler-TTS enrollment augmentation** — primarily English-trained

### 7.11.3 Multilingual Extension Path

**Data layer:**
- Replace LibriPhrase with Multilingual LibriSpeech (Pratap et al. 2020, CC-BY 4.0, 8 languages: English, German, Dutch, French, Spanish, Italian, Portuguese, Polish)
- Add Common Voice (Mozilla, CC0, 100+ languages) for Stage 2 joint training

**Phonetic module:**
- Replace g2p_en with `phonemizer` library (MIT) — wraps eSpeak-NG supporting 100+ languages
- Hard-negative mining adapts per-language automatically

**TTS augmentation:**
- Parler-TTS currently English-focused — for Phase 2, use XTTS-v2 (Coqui, MPL-2.0) or MMS-TTS (Meta, CC-BY-NC — not redistributable but usable at train time)

**Evaluation per language:**
- Report TA/FA stratified by language in the final report

### 7.11.4 Design Decision: Per-Locale Models vs Universal Model

Two options for Samsung deployment:

**Option A — Per-locale backbone:**
- Separate BC-ResNet trained for each major language family
- 50+ models in fleet
- Higher accuracy per-language
- Higher OTA update cost

**Option B — Universal multilingual backbone (recommended):**
- Single BC-ResNet trained on multilingual corpus
- One model in fleet
- Slight accuracy trade-off vs per-locale
- Dramatically simpler deployment and maintenance

Recommendation: Option B for Phase 2/3, with per-locale fine-tuning as a stretch optimization.

---

## 7.12 Robustness Edge Cases Beyond Noise

The KPI suite covers clean/noisy/distance. Real deployment sees many more edge cases. Each requires explicit testing.

### 7.12.1 Keyword Length Extremes

| Length | Example | Expected Behaviour |
|---|---|---|
| 1 syllable | "Tea" | Refuse enrollment (too short, high FA risk) — UX: "Please choose a longer keyword" |
| 2 syllables | "Hey you" | Accept with warning — TA may be lower |
| 3–4 syllables | "Hey Prism" (recommended) | Full accuracy |
| 5–6 syllables | "Prism activate now" | Full accuracy |
| 7+ syllables | "Hello there my dear assistant" | Accept — may see slight latency increase for detection window |

Enforced at enrollment time: syllable count via g2p_en, minimum 2 syllables, maximum 8.

### 7.12.2 Adversarial Audio Environments

| Environment | Test | Pass Criterion |
|---|---|---|
| TV in background (TV announcer says something similar) | eval set with news audio | FA rate < 0.5/hr |
| Music with vocals | eval set with pop music, keyword not in lyrics | FA rate < 0.5/hr |
| Two speakers talking simultaneously | cocktail party test (WSJ0-2mix) | TA for target speaker ≥ 85% |
| Echo/reverberation heavy room | RT60 > 600 ms | TA ≥ 80% |
| Very quiet whisper | whispered keyword | TA ≥ 70% (documented degradation) |
| Shouted | shouted keyword at high SPL | TA ≥ 90%, no clipping artifact |
| User is ill (cold, hoarse voice) | synthesised via pitch shift ±10% | TA ≥ 90% |
| User is drunk / slurred speech | TORGO dysarthric subset | TA documented per-user, no universal claim |

### 7.12.3 Device-Specific Edge Cases

| Case | Expected Behaviour |
|---|---|
| Phone in pocket (muffled audio) | Reject — silero-VAD gate most cases |
| Phone in pocket, user says wake word | Accept if signal passes VAD |
| Device vibrating during wake word | Test that motor vibration doesn't cause FP |
| Charging cable noise interference | Test with USB-C, USB-A, Qi chargers |
| Bluetooth audio routing mid-utterance | Handle gracefully, no state corruption |

### 7.12.4 Self-Trigger Protection

SoloSpeak must NOT wake on:
- Bixby saying the wake word in a TTS response
- Samsung alarm/notification sounds
- YouTube video playing "Hey Bixby..." in an ad

Protection:
- AEC suppresses own-speaker content
- Wake events during active Bixby session are suppressed by assistant state machine
- Template does not cover ±∞ — the TV announcer's voice + TV speaker coloration will fail the speaker head most of the time

---

## 7.13 Supply Chain and Dependency Policy

A production system built on open-source dependencies must have a dependency management policy. Reproducibility across 2+ years is required (Samsung product lifecycle).

### 7.13.1 Dependency Pinning

Every dependency pinned to exact version in `pyproject.toml`:

```
torch==2.3.1
torchaudio==2.3.1
onnx==1.16.0
onnxruntime==1.18.0
onnxruntime-mobile==1.18.0
numpy==1.26.4
silero-vad==4.0.0
parler-tts==0.2.1          # train-time only
g2p_en==2.1.0
```

Lock file committed. No `>=` ever, no `^` ever.

### 7.13.2 Vulnerability Monitoring

- **GitHub Dependabot** enabled on the repo
- **pip-audit** run in CI on every PR
- Critical CVEs in any runtime dependency trigger an advisory within 24 hours of disclosure
- Patch release cycle: CVE fix → pinned bump → full eval regression → OTA

### 7.13.3 Long-Term Reproducibility

Samsung ships devices expected to last 5+ years. Our model must be reproducible from source indefinitely.

- Training Docker image pinned and archived to Samsung's internal registry
- Dataset snapshots (immutable tarballs) archived with SHA-256
- Exact random seeds committed per training stage
- CUDA version, driver version, hardware configuration documented
- A "reproducibility CI job" runs quarterly: rebuild from committed state, verify numerical match on 100-sample eval subset (tolerance: ±0.2 pp absolute accuracy due to CUDA non-determinism)

### 7.13.4 Runtime Dependency Graph

```
Production runtime (on-device, cannot fail):
├── Android AudioRecord API (Android SDK, supported ≥ API 26)
├── ONNX Runtime Mobile (Microsoft, MIT, stable API)
├── (no Python, no PyTorch, no server calls)

Training-time only (Samsung's internal systems, can be swapped):
├── PyTorch 2.x
├── torchaudio 2.x
├── Weights & Biases (logging only, trivially replaceable)
└── Parler-TTS (only during enrollment data augmentation)
```

The production runtime has exactly two external dependencies: the Android audio API and ONNX Runtime. Both are rock-stable and vendor-maintained.

---

## 7.14 Cost, Compute, and Unit Economics

For a Samsung-scale deployment (100M+ devices), the numbers matter.

### 7.14.1 Per-Device Cost Model

**One-time:**
- OTA delivery: ~7.5 MB per device × 100M devices = 750 TB egress
- At AWS CloudFront rates (~$0.085/GB): ~$65,000 one-time for a full rollout
- Samsung has internal CDN infrastructure — true cost is a fraction of this

**Recurring (zero for standard config):**
- No inference runs on Samsung servers
- No voice data ingestion
- Optional fleet telemetry: ~100 bytes/device/day × 100M = ~10 GB/day ingestion → trivial at Samsung scale

**Per-enrollment:**
- Compute: ~5 seconds of Galaxy S25 CPU time = negligible
- Storage: ~1.1 KB per user profile on-device — free
- No cloud cost for enrollment

### 7.14.2 Training Cost (Phase 2)

**Solo-contestant budget (hackathon reality):**
- Kaggle free tier: 30 GPU-hours/week × 6 weeks = 180 GPU-hours (T4)
- Google Colab Pro: $10/month × 2 months = $20, ~100 GPU-hours (T4/L4)
- Total out-of-pocket: ~$20–40 including electricity
- Total compute: ~280 GPU-hours, well above the ~15 GPU-days (~360 hours on T4) estimated

**Samsung production retraining (scale reference):**
- A full retraining run on Samsung's internal cluster: ~50 A100-hours (12x faster than T4)
- Cost on hypothetical cloud: ~$100 per full retrain
- Retraining frequency for model improvement: quarterly
- Annual training budget: ~$400 — negligible vs device unit revenue

### 7.14.3 Carbon Footprint

- 280 GPU-hours on Kaggle T4 (150 W TDP typical draw ~70%) = ~30 kWh
- With grid average ~400 g CO₂/kWh = ~12 kg CO₂ for full training cycle
- Equivalent to ~60 km of gasoline car travel

For a model shipped to 100M devices used ~1000 times/day per user, the per-use training carbon cost is effectively zero (~10⁻⁹ g CO₂ per wake event from training). The inference carbon cost is 10 million times higher and still tiny (~0.01 mWh per wake = 0.004 g CO₂).

---

## 7.15 Testing Infrastructure Beyond Offline Evaluation

The KPI suite uses static test sets. Production requires continuous, hardware-in-the-loop testing.

### 7.15.1 Tiered Test Pyramid

```
                     ┌──────────────────┐
                     │  Manual demos    │    ~10 tests · weekly
                     │  on real devices │
                     └──────────────────┘
                 ┌────────────────────────┐
                 │ Hardware-in-the-loop   │  ~100 tests · nightly
                 │ device farm            │
                 └────────────────────────┘
           ┌──────────────────────────────────┐
           │  Integration: Android emulator + │  ~1000 tests · per PR
           │  pre-recorded audio files        │
           └──────────────────────────────────┘
      ┌────────────────────────────────────────────┐
      │  Unit tests: pure Python, model,           │  ~10,000 tests · per commit
      │  ONNX I/O, template math, enrollment logic │
      └────────────────────────────────────────────┘
```

### 7.15.2 Unit Test Coverage Targets

Every commit triggers:
- 95%+ line coverage on `solospeak/` Python package
- 100% coverage on threshold calibration, fusion MLP math, cosine similarity
- Deterministic model output test: fixed input mel → fixed output embedding (bit-exact)
- ONNX export round-trip: PyTorch output == ONNX output within 1e-5 absolute

### 7.15.3 Integration Test Fixtures

- 50 pre-recorded wake-word utterances across 10 speakers, 5 environments
- 100 pre-recorded non-wake utterances (phonetically similar + random speech)
- 10 continuous 60-minute ambient audio recordings for FA rate integration
- All committed to the repo under `tests/fixtures/audio/` (CC-BY 4.0)

### 7.15.4 Device Farm (Samsung Internal, Post-Hackathon)

Samsung operates an internal device farm with 1000+ Galaxy devices in a controlled acoustic chamber. SoloSpeak integration tests would run on a subset nightly:

- 20 device types covering 3 tiers × 5 generations × 2 OS versions
- 50 acoustic scenarios (anechoic, office, kitchen, car, street, TV-on, music-on, etc.)
- Automated recording → automated scoring → automated pass/fail
- Failure triggers: manual review within 24 hours

### 7.15.5 Regression Test Policy

Every model checkpoint produced by training must achieve:
- No regression vs previous checkpoint on any KPI (within 1 pp tolerance)
- No regression on any demographic subgroup (within 2 pp tolerance)
- No regression on any of 10 edge-case tests (within 1 pp)

A regression blocks the checkpoint from deployment automatically.

---

## 7.16 Operational Runbook for the Production Team

When SoloSpeak runs in Samsung production and something goes wrong at 3 AM, the on-call engineer needs a runbook.

### 7.16.1 Incident Response Playbook

| Alert | Likely Cause | First Action | Escalation |
|---|---|---|---|
| Fleet FA rate > 2× baseline | Bad model shipped, acoustic environment shift | Initiate rollback procedure (7.10.3) | ML team within 30 min |
| Fleet FA rate > 5× baseline | Potential security incident (replay attack wave?) | Rollback + emergency security review | Security team within 15 min |
| Enrollment success < 80% | TTS service failure, encoder regression | Check TTS synthesis logs; if OK, rollback model | ML team within 1 hr |
| Crash rate > 0.1% | Runtime bug, memory leak, ONNX incompatibility | Check crash logs, rollback if reproducible | Platform team within 1 hr |
| Elevated inference latency | CPU thermal throttling, model size regression | Check thermal telemetry; if model issue, rollback | Platform + ML within 2 hr |
| User reports "only works for dad, not mom" | Speaker head poorly calibrated for user | Triage: re-enrollment usually fixes; if pattern, investigate demographics | ML team within 24 hr |
| EU AI Act complaint received | User requesting data deletion / disclosure | Route to privacy team | Privacy team within 24 hr |

### 7.16.2 Standard Operating Procedures

Documented procedures (in Samsung's internal runbook system):
- SOP-1: How to push an emergency OTA rollback
- SOP-2: How to pull differential-privacy fleet metrics from the dashboard
- SOP-3: How to investigate a regression detected during canary
- SOP-4: How to respond to a GDPR data subject request
- SOP-5: How to escalate a suspected security incident

Each SOP has a named owner, an RTO (recovery time objective), and an RPO (recovery point objective where applicable).

### 7.16.3 On-Call Rotation and Ownership

| Role | Responsibility | Response Time |
|---|---|---|
| ML on-call | Model quality regressions, demographic issues | 2 hours |
| Platform on-call | Runtime crashes, Android integration | 1 hour |
| Security on-call | Suspected attacks, CVE response | 15 minutes |
| Privacy on-call | GDPR/CCPA requests, data handling | 4 hours |

---

Call these out explicitly and by name — judges skim for novelty bullets.

### Novelty Claim 1: Joint disentangled content–speaker encoder
> "State of the art (OpenWakeWord, Porcupine, Google TCResNet) either ignores speaker identity entirely or runs separate KWS + SV pipelines. We propose a single shared backbone with two *orthogonal* output heads trained jointly via gradient-reversal adversarial disentanglement + cross-covariance orthogonality loss. This is genuinely novel for custom wake-word detection and produces a principled joint embedding space."

### Novelty Claim 2: TTS-augmented few-shot enrollment
> "Current personalized KWS requires 10–50 user recordings for stable enrollment. We use open Apache-2.0 TTS (Parler-TTS) to synthesize acoustic/prosodic variations of the keyword text, reducing required user recordings to 3–5 while maintaining phonetic coverage."

### Novelty Claim 3: Learned gated fusion vs. ad-hoc combiners
> "Prior work combines content and speaker scores with hand-designed functions (multiplication, min, weighted sum). We learn the fusion function via a tiny 400-parameter MLP trained on the 4-quadrant confusion matrix, yielding an asymmetric decision boundary that adapts to the content–speaker similarity manifold."

### Novelty Claim 4: Curriculum SNR + distance training
> "Joint SNR sweep (−5 to 30 dB) with synthetic RIR distance simulation (0.5–5 m) applied in a curriculum schedule — clean → progressively harsher — rather than uniform sampling, improving convergence stability and final robustness."

### Novelty Claim 5: Samsung-native backbone lineage
> "We build on BC-ResNet (Samsung Research, Kim et al. 2021), demonstrating a direct productization path for Samsung's existing acoustic research." (This is a positioning claim, not a technical one — but judges notice.)

Each claim maps to an ablation result in §6.9 and a specific loss term in §3.6 — your report must close that loop.

---

## 9. Risk Register (Every Risk, Every Mitigation)

| # | Risk | Probability | Impact | Mitigation | Trigger for fallback |
|---|---|---|---|---|---|
| R1 | BC-ResNet re-impl has bugs | Med | High | Use MatchboxNet backup; validate against GSC-v2 baseline (96%+) before Stage 2 | GSC-v2 acc < 93% after Stage 1 |
| R2 | Not hitting 99% TA clean | Med | Critical | More enrollment samples, stronger backbone (BC-ResNet-8 → BC-ResNet-16), threshold calibration | Dev TA < 97% after Stage 5 |
| R3 | FA rate > 1/hr | Med | Critical | More hard negative mining, longer 24h eval, stricter threshold, Q2/Q3/Q4 rebalancing | Dev FA > 1.5/hr |
| R4 | xRT > 0.2 on target CPU | Low | High | INT8 quantization, operator fusion, profile hotspots | FP32 xRT > 0.25 |
| R5 | Model > 3M params | Low | Critical | Trim BC-ResNet scale, smaller head dim (128 → 64) | Count > 2.9M |
| R6 | TTS license issue | Low | Med | Parler-TTS is Apache-2.0 (verify at submission time); fallback = no TTS aug | License audit fails |
| R7 | VoxCeleb download/access issue | Low | High | Request academic access early (it's open but gated); backup LibriSpeech speaker labels | Day 5 no access |
| R8 | Disentanglement doesn't converge | Low | Med | Adversarial schedule is known-tricky; fallback: drop adversarial, keep orthogonality only | Probe acc stays > 40% after Stage 3 |
| R9 | Single GPU insufficient | Low | Med | Use Colab/Kaggle for overflow; batch size reduction | Training takes > 2 days per stage |
| R10 | Solo developer burnout / sickness | Med | Critical | Weekly buffer days baked into schedule; scope cut list prepared | Behind by > 2 days |
| R11 | Plagiarism accusation | Low | Critical | Every repo commit from your account; every external code block attributed in code comments; Apache-2.0 headers on all files | — |
| R12 | Blueprint email bounces or lost | Low | Critical | Submit 48 hours early; send to yourself as BCC; retain delivery receipt | — |

### 9.1 Scope cut list (use only if behind schedule)

Cut in this order if you're falling behind, NEVER cut above the line:

1. Android demo app (optional from the start)
2. Raspberry Pi xRT benchmark (keep laptop-only)
3. 24-hour FA eval (keep 10-hour)
4. 3-seed variance (drop to 1 seed)
5. Distance range 5 m (test only up to 3 m)
6. TTS augmentation (drop as ablation, keep as text-only plan)

**Do NOT cut:** core dual-head model, clean + noisy TA, FA/hr eval, parameter audit, xRT on laptop, primary ablations (no speaker head, no content head).

---

## 10. Timeline — Day-by-Day (6-week implementation window)

Dates assume Phase 1 blueprint submitted on time (May 13). Implementation starts May 14.

### Pre-submission (NOW → May 11)

- **Now → Apr 27:** Finalize blueprint (slides 1–9), internal review, export PDF
- **Apr 28 – May 4:** While waiting, do environment setup (§11) and run Stage 1 backbone training — gives you data for Slide 9 "additional materials"
- **May 5 – May 11:** Blueprint polish, read-cold by someone outside audio ML, submit by May 11 (48-hour buffer before May 13 deadline)

### Week 1 (May 14 – 20): Foundation

- Day 1 (Wed): Git repo initialized, agents.md drafted, skills/ directory populated, license headers template, CI scaffold (GitHub Actions running `pytest`)
- Day 2: Download GSC-v2, LibriSpeech-derived LibriPhrase subset, MUSAN. Write `data/loaders.py` with all datasets registered
- Day 3: Implement audio frontend (mel spec), unit tests
- Day 4: Implement BC-ResNet-8 backbone, unit tests, parameter count audit (must be under budget)
- Day 5: Implement content head, speaker head, full model forward
- Day 6: Training loop scaffold, Weights & Biases integration, first Stage-1 run kick off
- Day 7: Stage 1 eval on GSC-v2, target ≥ 96% — GO/NO-GO

### Week 2 (May 21 – 27): Dual-head

- Day 8: Write SupCon loss, contrastive sampler
- Day 9: Write per-speaker and per-word class heads, combined loss
- Day 10: Stage 2 training starts (5 days)
- Days 11–13: Monitor training, implement eval harness for clean TA / noisy TA / FA/hr / parameter audit / xRT
- Day 14: Stage 2 complete, dev TA clean ≥ 96% — GO/NO-GO

### Week 3 (May 28 – Jun 3): Disentanglement + noise

- Day 15: Implement orthogonality loss, gradient reversal layer, adversarial branches
- Day 16: Stage 3 training (3 days) — monitor probe accuracy drops
- Day 17–18: Stage 3 completes, begin Stage 4 (noise/distance augmentation + hard neg mining)
- Day 19: Implement G2P-based phonetic hard negatives, speaker-cluster hard negatives
- Day 20: Stage 4 training (3 days)
- Day 21: Dev noisy TA ≥ 88% checkpoint — GO/NO-GO

### Week 4 (Jun 4 – 10): Fusion + full eval

- Day 22: Implement learned fusion MLP, Stage 5 training (1 day)
- Day 23: Complete end-to-end eval pipeline: clean/noisy/distance/Q1-Q4/xRT
- Day 24: Hard negative retraining round (optional quality boost)
- Day 25: Build all ablations from §6.9 — each runs as a separate seed-controlled experiment
- Day 26: First full end-to-end number on all KPIs. **This is the critical checkpoint.** Every KPI target should be met or near-met here.
- Day 27: TTS augmentation integration (if Parler-TTS works), retrain stages 2+3 with TTS data, compare
- Day 28: Consolidate results, ablation tables

### Week 5 (Jun 11 – 17): Optimization + Demo

- Day 29: Quantization-aware fine-tune (Stage 6)
- Day 30: ONNX export, INT8 quantize, xRT benchmark
- Day 31: Fix xRT / parameter count / memory if any target missed
- Day 32: Build demo app (streaming Python UI minimum, Android if time)
- Day 33: Record demo video (5+ scenarios showcasing Q1/Q2/Q3/Q4 + noisy + distance)
- Day 34: Demo video edit, polish
- Day 35: Buffer / catch-up day

### Week 6 (Jun 18 – 22): Report + Submission

- Day 36 (Jun 18): Draft final report — methodology, results tables, ablations, discussion
- Day 37: Report second draft, generate all plots (DET curve, SNR sweep, distance sweep, ablation bars)
- Day 38: Report polish, ensure all KPI claims cite §6 eval procedures
- Day 39: Final code cleanup, README, Apache-2.0 headers, all open-source citations verified
- Day 40 (Jun 21): Submit 24 hours before deadline
- Day 41 (Jun 22): Deadline — you're done, in bed, watching others panic

### 10.1 Weekly "GO/NO-GO" gates

At the end of each week, you must be able to answer YES to:

- Week 1: Backbone pretrained, GSC-v2 ≥ 96%?
- Week 2: Dual-head trained, dev clean TA ≥ 96%?
- Week 3: Disentanglement active, dev noisy TA ≥ 88%?
- Week 4: All KPIs measured, clean ≥ 99%, noisy ≥ 90%, FA < 1/hr?
- Week 5: Quantized model ≤ 3 MB, xRT < 0.2?
- Week 6: Report draft complete, video recorded, repo clean?

If any week's gate fails, trigger the scope cut list (§9.1) on Monday of the following week.

---

## 11. Hardware, Environment, Tooling

### 11.1 Hardware

**Minimum:** Single GPU with ≥ 8 GB VRAM (RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090, A10, A100, T4×2, L4). Also viable: Google Colab Pro (T4/A100 rotation), Kaggle notebooks (P100×2, 30h/week free), Lambda Labs cloud ($0.50/hr spot instances).

**Recommended:** 16 GB VRAM. Speeds up Stage 2/4 training by 2x.

**CPU/RAM:** 16 GB system RAM, 8-core modern CPU. Dataset streaming reduces I/O requirements.

**Storage:** 500 GB free. LibriSpeech + VoxCeleb2 + MUSAN + RIRs ≈ 350 GB. Use dataset streaming (webdataset or HF streaming) where possible to avoid full downloads.

**For xRT benchmarking on "device":** Use Raspberry Pi 4 (4 GB, ~$45) as phone-CPU proxy. Samsung Galaxy S22+ ARM Cortex-A78 ≈ RPi 4 + 30%. If no RPi, throttled laptop CPU (1 core at 1.5 GHz) is an acceptable proxy.

### 11.2 Environment

```
# conda / uv / pip — use uv, it's faster
Python 3.11
PyTorch 2.3+ (CUDA 12.1)
torchaudio 2.3+
onnx, onnxruntime, onnxruntime-tools
numpy, scipy, librosa (eval only)
wandb or mlflow
pytest, ruff, mypy
g2p_en
silero-vad (pip)
transformers (for Parler-TTS)
```

Pin versions in `pyproject.toml`. Do NOT use `requirements.txt` — it's 2026.

### 11.3 Reproducibility

- Deterministic: `torch.manual_seed`, `numpy.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic=True`
- Log every config with W&B (free tier is fine for solo)
- Save model weights, optimizer state, RNG state every epoch
- Every reported number reproducible with a single `python eval.py --config configs/paper/final.yaml` command

---

## 12. Repository Structure

```
solospeak/
├── README.md                        # What, why, how-to-run, citations
├── LICENSE                          # Apache-2.0
├── pyproject.toml                   # Dependencies, ruff/mypy config
├── .github/workflows/
│   ├── ci.yml                       # pytest on every push
│   └── train.yml                    # (optional) Kaggle/Colab trigger
│
├── agents.md                        # Agentic dev workflow doc (see §13)
├── skills/
│   ├── data_curation/SKILL.md       # How to add a new dataset
│   ├── training/SKILL.md            # How to launch a training run
│   ├── evaluation/SKILL.md          # How to run a KPI eval
│   └── deployment/SKILL.md          # How to export + benchmark
│
├── configs/
│   ├── backbone/bcresnet8.yaml
│   ├── training/stage1_pretrain.yaml
│   ├── training/stage2_dualhead.yaml
│   ├── training/stage3_disentangle.yaml
│   ├── training/stage4_robust.yaml
│   ├── training/stage5_fusion.yaml
│   ├── training/stage6_quant.yaml
│   └── eval/full_kpi.yaml
│
├── solospeak/                       # Main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── datasets.py              # LibriPhrase, GSC-v2, VoxCeleb, etc.
│   │   ├── augment.py               # noise/RIR/SpecAug
│   │   ├── samplers.py              # class-balanced, hard-negative
│   │   ├── mining.py                # phonetic + speaker hard negatives
│   │   └── tts_augment.py           # Parler-TTS integration
│   ├── models/
│   │   ├── frontend.py              # log-mel
│   │   ├── vad.py                   # Silero VAD wrapper
│   │   ├── backbone_bcresnet.py
│   │   ├── backbone_matchbox.py     # fallback
│   │   ├── heads.py                 # content, speaker
│   │   ├── adversary.py             # gradient reversal + probes
│   │   ├── fusion.py                # gated fusion MLP
│   │   └── solospeak.py             # full model
│   ├── losses/
│   │   ├── supcon.py
│   │   ├── orthogonality.py
│   │   ├── adversarial.py
│   │   └── triplet.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── scheduler.py
│   │   └── curriculum.py
│   ├── evaluation/
│   │   ├── clean.py
│   │   ├── noisy.py
│   │   ├── distance.py
│   │   ├── quadrants.py             # Q1-Q4 metrics
│   │   ├── fa_rate.py               # long-background
│   │   ├── xrt.py
│   │   └── params_audit.py
│   └── deployment/
│       ├── export_onnx.py
│       ├── quantize.py
│       └── inference.py             # streaming demo
│
├── scripts/
│   ├── download_datasets.sh
│   ├── train_stage.py               # --stage 1..6
│   ├── eval_all_kpi.py
│   ├── make_figures.py              # all report plots
│   └── run_ablations.sh
│
├── tests/
│   ├── test_frontend.py
│   ├── test_backbone.py
│   ├── test_heads.py
│   ├── test_losses.py
│   └── test_e2e.py
│
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_backbone_sanity.ipynb
│   ├── 02_disentanglement_probe.ipynb
│   └── 03_fusion_decision_boundary.ipynb
│
├── docs/
│   ├── SOLOSPEAK_ARCHITECTURE.md    # this document
│   ├── REPORT.md                    # final submission report
│   └── DEMO_SCRIPT.md               # §14
│
└── assets/
    ├── enrollment_templates/
    ├── demo_audio/
    └── figures/
```

Every file has an Apache-2.0 header. Every third-party snippet is attributed in a comment block with source URL and license.

---

## 13. Agentic Development Plan (Blueprint Slide 8 — do not skip)

The hackathon rules explicitly reward "agentic workflows, MCP servers, agents.md, skills, coding assistants." This slide is easy points if you actually plan it. Here's what you say and what you do.

### 13.1 What to put in `agents.md`

```markdown
# SoloSpeak Agentic Development

## Primary agent: Claude Code (or Cursor) with project context
Context includes: this architecture doc, skills/, configs/, tests/

## Subagent roles (prompted as specialist subagents)

### data-agent
Role: manage dataset downloads, integrity checks, split generation.
Tools: shell, python, local filesystem.
Skill reference: skills/data_curation/SKILL.md

### training-agent
Role: launch training stages, monitor W&B curves, halt/restart on divergence.
Tools: shell, W&B API, SSH to GPU box.
Skill reference: skills/training/SKILL.md

### eval-agent
Role: run full KPI suite on a checkpoint, produce tables + plots.
Tools: python, matplotlib, pandas.
Skill reference: skills/evaluation/SKILL.md

### ablation-agent
Role: systematically run the ablation matrix from §6.9 across 3 seeds.
Tools: orchestrates training-agent + eval-agent.

### deploy-agent
Role: export ONNX, quantize, benchmark xRT, produce deployment artifact.
Skill reference: skills/deployment/SKILL.md
```

### 13.2 MCP server idea (easy bonus)

Build a minimal MCP server that wraps the dataset API: `list_splits`, `get_sample(split, idx)`, `get_stats(split)`. This lets the coding agent query sample data while debugging without loading the full dataset in-context. ~100 lines of Python. Document it in the blueprint.

### 13.3 What judges want to see in Slide 8

Bullet list — be concrete:

- "Development orchestrated via Claude Code with a custom `agents.md` defining 5 subagent roles (data, training, eval, ablation, deploy) and `skills/` directory with operational runbooks."
- "Custom MCP server wraps dataset access, letting the coding agent query sample statistics and individual examples during debugging without loading tens of gigabytes."
- "Ablation matrix (7 configurations × 3 seeds = 21 runs) is orchestrated by the ablation-agent, which sequences training + evaluation automatically."
- "Report figures auto-generated by the eval-agent from a single config pointer to the model checkpoint — full reproducibility in one command."

This slide being well-populated differentiates you from the 80% of teams who leave it empty or generic.

---

## 14. Demo Script (Phase 3 & Phase 4 Presentations)

A judge remembers *three things* from a 10-minute talk: the problem, one number, one live moment. Engineer all three.

### 14.1 Demo structure (8 minutes live + 2 min Q&A)

**(0:00 – 0:45) Opening hook**
> "Imagine your Galaxy phone on your kitchen counter. A friend walks by and jokingly says 'Hey Bixby, delete all my photos.' Your phone obeys. This happens in real homes every day. Today I'll show you a 2-megabyte fix."

**(0:45 – 2:15) Problem framing**
- Current wake-word systems are speaker-agnostic (show OpenWakeWord / Bixby demo accepting any voice)
- Q1/Q2/Q3/Q4 matrix
- Why naive cascade (KWS then SV) fails

**(2:15 – 4:15) Approach — architecture slide**
- Shared encoder, dual orthogonal heads, disentanglement
- Name the three novelty claims
- Mention BC-ResNet Samsung lineage

**(4:15 – 6:45) LIVE DEMO** — this is your money moment
- You enroll a custom word ("samsara", "nebulon", anything you invent) in 30 seconds on stage
- You say it → phone/laptop lights up. Applause.
- Pre-arranged co-presenter shouts the same word from the back of the stage → silence. Bigger applause.
- You turn on a speaker playing crowd babble at ~20 dB SNR → you say the word again → still works.
- Walk 4 meters away from the mic → say it → still works.

**(6:45 – 7:45) Numbers slide**
- Single slide: big table, clean TA 99.2%, noisy TA 91.5%, FA 0.4/hr, xRT 0.08, params 2.1M
- Ablation bar chart: show speaker head contribution

**(7:45 – 8:00) Close**
> "2.1 million parameters, 2 megabytes on disk, runs on any phone CPU, Apache-2.0 licensed. This is a drop-in upgrade for Bixby, Galaxy Buds, SmartThings. Thank you."

### 14.2 Demo failure recovery

Things that WILL go wrong on stage. Have fallbacks:

- **Mic not working:** Pre-recorded demo video as backup, play from laptop
- **Co-presenter sick:** Pre-recorded imposter audio clip
- **Network dies:** Everything runs locally anyway, this shouldn't bite you — but verify local run before presenting
- **Model checkpoint corrupted:** Carry two USBs with the model + one cloud backup

Rehearse the demo **3 times** in the week before Phase 3/4. Time it. Shave every extra word.

### 14.3 Demo video requirements (for Phase 2 submission)

The problem statement requires a demo video showcasing 5+ scenarios. Your video must include:

1. Clean enrollment + detection (Q1)
2. Imposter rejection same word (Q2)
3. Phonetic confusable word rejection (Q3)
4. Random speech non-detection (Q4)
5. Noisy environment (crowd babble at 5 dB SNR)
6. Distance test (3+ meters)
7. (Bonus) Latency visualization showing xRT

Length: 3–5 minutes. No music. Clean voiceover. Show the terminal output or GUI next to the audio source. Record in a quiet room; add noise in post for controlled SNR.

---

## 15. Final Report Outline (For Phase 2)

Judges read reports like this one: section 1 (hook), section 6 (results table), section 8 (what's novel). Write for their skim pattern.

```
1. Abstract (150 words) — problem, approach, headline number, impact
2. Introduction (1 page) — motivation, prior SOTA, contribution list (3–5 bullets)
3. Related Work (1 page) — KWS, SV, joint systems, disentanglement
4. Method (3 pages)
   4.1 Architecture overview (with diagram)
   4.2 Disentangled dual heads
   4.3 Losses (with equations from §3.6)
   4.4 Gated fusion
   4.5 Training curriculum
5. Datasets & Augmentation (1 page)
6. Experiments (3 pages)
   6.1 Main results table (all KPIs)
   6.2 SNR sweep figure
   6.3 Distance sweep figure
   6.4 Ablation table
   6.5 xRT + parameter audit
   6.6 DET curve
7. Discussion (1 page) — what worked, what didn't, failure cases
8. Limitations (half page) — be honest, it builds credibility
9. Deployment Notes (half page) — ONNX, quantization, mobile path
10. Conclusion (quarter page)
11. References (BC-ResNet, OpenVLA paper? no — OpenWakeWord, Porcupine, DANN, SupCon, Parler-TTS, LibriPhrase, VoxCeleb, MUSAN, BUT Reverb, Silero)
12. Appendix: all hyperparams, full ablation table with seeds, license attribution table
```

Target length: 10–12 pages. Double-column IEEE-style or single-column Samsung template if provided.

---

## 16. Pre-Submission Checklists

### 16.1 Phase 1 Blueprint PDF checklist (by May 11)

- [ ] Team details fully filled, institute email used
- [ ] Problem statement title: exact copy from Samsung brief
- [ ] Understanding: 3 paragraphs, names Q1/Q2/Q3/Q4 framework
- [ ] Proposed solution: 1-line thesis + prose description
- [ ] Technical diagram present (see §2 ASCII as reference; redraw in slides with arrows and labels)
- [ ] Constraints section honest (enrollment burden, VAD assumption, mic quality)
- [ ] Novelty slide names 3–5 SOTA systems and differentiates from each
- [ ] Datasets slide: every item with license confirmed and written
- [ ] Models slide: every item with parameter count and license
- [ ] Agentic tools slide: names specific tools (Claude Code, MCP, agents.md, skills/, W&B) and what each does
- [ ] Timeline in additional materials (Gantt)
- [ ] Risk register in additional materials
- [ ] No closed API mentioned as runtime dependency
- [ ] Apache-2.0 release declared
- [ ] Exported as PDF (not PPTX)
- [ ] Filename: `<teamname>-04-solospeak.pdf` (lowercase, hyphens)
- [ ] Subject line: `AX Hackathon Phase 1 Submission | 04 | <TeamName>`
- [ ] To: ennovatex.io@samsung.com
- [ ] From: institute email
- [ ] Sent 48+ hours before deadline
- [ ] Delivery receipt retained

### 16.2 Phase 2 Full Submission checklist (by Jun 20)

- [ ] Public GitHub repo, Apache-2.0 license
- [ ] README with: overview, install, train, eval, deploy, citations
- [ ] `pyproject.toml` pinned
- [ ] All tests passing in CI
- [ ] Pretrained checkpoints uploaded to HuggingFace (Apache-2.0)
- [ ] Demo video (3–5 min) linked in README
- [ ] Report PDF committed to repo
- [ ] ONNX + INT8 artifacts in releases page
- [ ] Reproducibility: `python eval.py --config configs/final.yaml` produces main results
- [ ] License attribution table in docs/LICENSES.md
- [ ] Third-party code: attributed in source with comments
- [ ] Apache-2.0 headers on every source file
- [ ] No accidental API keys, credentials, or personal data in repo

---

## 17. Communication Protocol with Your Mentor (Me)

Solo contestants fail when they get stuck silently. Don't.

- **Weekly check-in:** Every Sunday evening, paste me: (a) what you completed this week, (b) blockers, (c) plan for next week. I'll respond with concrete fixes.
- **GO/NO-GO gates:** After each week's gate (§10.1), message me the gate result + numbers. If NO-GO, we trigger scope cuts together.
- **Emergency escalation:** If any single issue is costing you more than 4 hours, stop and message me. Four hours of lost solo debugging is worth 4 minutes of asking.
- **Before every major architectural change:** Do not silently deviate from this document. If you want to change a loss or a head or a dataset — propose the change, explain why, wait for sign-off.

---

## 18. The Shortlist of Things That Will Decide Whether You Win

If you execute nothing else in this document, execute these:

1. **Hit 99% clean TA and <1/hr FA.** Non-negotiable. Every other number is garnish.
2. **Live demo works flawlessly.** Rehearse, rehearse, rehearse.
3. **Ablation table in the report.** Shows scientific discipline. Separates winners from finalists.
4. **Disentanglement actually happens.** Verify via probe classifiers. If the probe still recovers 50% speaker accuracy from $z_c$, your novelty claim is a lie.
5. **Blueprint slide 8 populated with real agentic workflow details.** Most teams will leave this generic. Don't.
6. **Every open-source license verified.** Samsung will disqualify teams that ship with non-commercial or GPL contamination.
7. **Samsung-native framing.** BC-ResNet is from Samsung. Galaxy Buds + Bixby + SmartThings are the target products. Mention these by name in report and demo.

---

## 19. Parting Words (From Your Mentor)

This is a marathon, not a sprint. You will hit a wall in week 3 when training doesn't converge. You will hit a wall in week 5 when the quantized model drops 4% TA. You will want to quit in week 6 when the report feels like it will never end.

When you hit those walls, come back to this document. Everything you need to ship is in here. The architecture is sound. The KPIs are hit-able. The novelty is real. The timeline has buffer.

Execute. I'll be here at every check-in.
