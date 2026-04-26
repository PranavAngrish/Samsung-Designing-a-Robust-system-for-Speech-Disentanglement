"""Import smoke tests for the assembled model module."""

from __future__ import annotations

from solospeak.models.solospeak import SoloSpeakModel


def test_model_class_imports() -> None:
    assert SoloSpeakModel.__name__ == "SoloSpeakModel"
