"""Training-only wrapper around the deployable SoloSpeak model."""

from __future__ import annotations

import torch
from torch import nn

from solospeak.losses.adversarial import AdversarialProbeHead, grad_reverse
from solospeak.models.heads import AuxiliaryHeads
from solospeak.models.solospeak import SoloSpeakModel
from solospeak.utils.config import SoloSpeakConfig


class TrainingWrapper(nn.Module):
    """Adds auxiliary and adversarial heads used only during training."""

    def __init__(self, config: SoloSpeakConfig) -> None:
        super().__init__()
        if config.training.n_aux_word_classes is None:
            raise ValueError("training.n_aux_word_classes must be injected from STATS.json")
        if config.training.n_aux_speaker_classes is None:
            raise ValueError("training.n_aux_speaker_classes must be injected from STATS.json")
        self.model = SoloSpeakModel(config)
        embed_dim = config.heads.content_dim
        n_words = config.training.n_aux_word_classes
        n_speakers = config.training.n_aux_speaker_classes
        self.aux_heads = AuxiliaryHeads(embed_dim, n_words, n_speakers)
        self.content_adversary = AdversarialProbeHead(embed_dim, n_speakers)
        self.speaker_adversary = AdversarialProbeHead(embed_dim, n_words)

    def forward(self, mel: torch.Tensor, adv_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        z_c, z_s = self.model(mel)
        word_logits, speaker_logits = self.aux_heads(z_c, z_s)
        return {
            "z_c": z_c,
            "z_s": z_s,
            "word_logits": word_logits,
            "speaker_logits": speaker_logits,
            "speaker_from_content_logits": self.content_adversary(grad_reverse(z_c, adv_lambda)),
            "word_from_speaker_logits": self.speaker_adversary(grad_reverse(z_s, adv_lambda)),
        }

    def deployable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only clean SoloSpeakModel weights for Stage 5/6/export."""
        return self.model.state_dict()
