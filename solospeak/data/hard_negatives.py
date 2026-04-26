"""Phonetic and speaker hard-negative mining.

Negatives are refreshed once per epoch (not per batch) to keep the training loop fast.
"""

from __future__ import annotations

import numpy as np

from solospeak.utils.types import FloatArray


def phone_edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    """Levenshtein distance between two phoneme sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def phonetic_hard_negatives(keyword: str, vocab: list[str], k: int = 10) -> list[str]:
    """Find k phonetically closest words to keyword using g2p_en phoneme edit distance.

    Call once per epoch (not per batch) — G2p() loads a language model on each
    instantiation; per-batch use would add ~200 ms overhead per call.
    """
    from g2p_en import G2p

    g2p = G2p()
    target_phones = g2p(keyword)
    scored = [
        (word, phone_edit_distance(target_phones, g2p(word)))
        for word in vocab
        if word != keyword
    ]
    scored.sort(key=lambda x: x[1])
    return [w for w, _ in scored[:k]]


def speaker_hard_negatives(
    anchor_speaker_id: str,
    speaker_embeddings: FloatArray,
    speaker_ids: list[str],
    k: int = 10,
) -> list[str]:
    """Find k speakers whose embedding is closest to the anchor (hardest negatives).

    speaker_embeddings: (N, D) L2-normalized embedding matrix, row i = speaker_ids[i].
    """
    anchor_idx = speaker_ids.index(anchor_speaker_id)
    anchor_vec = speaker_embeddings[anchor_idx]
    scores = speaker_embeddings @ anchor_vec  # cosine similarity (embeddings are L2-normalized)
    ranked = np.argsort(-scores)
    # Exclude the anchor itself
    hard_neg_idx = [i for i in ranked if i != anchor_idx][:k]
    return [speaker_ids[i] for i in hard_neg_idx]
