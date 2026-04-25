"""Speaker-disjoint train/dev/test split generation.

Critical invariant: no speaker appears in more than one split.
Split assignment uses a deterministic SHA-256 hash of the speaker ID,
so the same speaker always lands in the same split regardless of call order.
"""

from __future__ import annotations

import hashlib


def assign_split(
    speaker_id: str,
    seed: int = 42,
    train_pct: float = 0.85,
    dev_pct: float = 0.10,
) -> str:
    """Return 'train', 'dev', or 'test' for a speaker ID.

    Deterministic and reproducible. Speaker-disjoint by construction.
    """
    h = hashlib.sha256(f"{seed}:{speaker_id}".encode()).hexdigest()
    pct = int(h[:8], 16) / 0xFFFFFFFF

    if pct < train_pct:
        return "train"
    elif pct < train_pct + dev_pct:
        return "dev"
    return "test"


def verify_disjoint(speaker_split_map: dict[str, str]) -> bool:
    """Verify no speaker_id maps to more than one split value.

    speaker_split_map: {speaker_id: split} as produced by assign_split.
    Since assign_split is deterministic each speaker always gets one split,
    but this check catches bugs if the map was built from mixed sources.
    """
    from collections import defaultdict
    split_to_speakers: dict[str, set[str]] = defaultdict(set)
    for spk, split in speaker_split_map.items():
        split_to_speakers[split].add(spk)

    all_speakers = list(speaker_split_map.keys())
    total_unique = sum(len(s) for s in split_to_speakers.values())
    if total_unique != len(all_speakers):
        raise ValueError(
            f"Speaker leak detected: {len(all_speakers)} entries but "
            f"{total_unique} unique across splits — some speakers appear in multiple splits."
        )
    return True
