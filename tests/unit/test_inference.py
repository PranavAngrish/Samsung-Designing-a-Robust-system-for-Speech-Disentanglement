"""Unit tests for streaming inference utilities."""

from __future__ import annotations


from solospeak.inference.hysteresis import HysteresisDetector


def test_fires_on_high_score() -> None:
    det = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory_s=1.5, stride_s=0.1)
    assert det.step(0.9) is True


def test_no_fire_below_tau_on() -> None:
    det = HysteresisDetector(tau_on=0.75)
    for _ in range(20):
        assert det.step(0.5) is False


def test_refractory_prevents_double_fire() -> None:
    det = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory_s=1.5, stride_s=0.1)
    assert det.step(0.9) is True
    for _ in range(14):  # 14 frames < 15 refractory frames
        assert det.step(0.9) is False


def test_fires_again_after_refractory() -> None:
    det = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory_s=1.5, stride_s=0.1)
    det.step(0.9)
    for _ in range(15):  # exhaust refractory
        det.step(0.0)
    assert det.step(0.9) is True


def test_hysteresis_requires_drop_below_tau_off() -> None:
    det = HysteresisDetector(tau_on=0.75, tau_off=0.45, refractory_s=0.0, stride_s=0.1)
    det.step(0.9)           # fires, active=True
    det._refractory_countdown = 0   # clear refractory manually for this test
    det.step(0.6)           # above tau_off but below tau_on — stays active, no re-fire
    assert det._active is True
    det.step(0.3)           # below tau_off — deactivate
    assert det._active is False
    assert det.step(0.9) is True    # can fire again now


def test_reset_clears_state() -> None:
    det = HysteresisDetector()
    det.step(0.9)
    det.reset()
    assert det._active is False
    assert det._refractory_countdown == 0
