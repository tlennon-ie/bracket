import math
import pytest

from bracket.ema import EMASmoother


def test_first_update_seeds_with_raw_value():
    s = EMASmoother(alpha=0.1)
    assert s.value is None
    out = s.update(5.0)
    assert out == 5.0
    assert s.value == 5.0


def test_subsequent_updates_blend_correctly():
    s = EMASmoother(alpha=0.5)
    s.update(0.0)
    out = s.update(10.0)
    assert out == pytest.approx(5.0)
    out = s.update(10.0)
    assert out == pytest.approx(7.5)


def test_alpha_one_tracks_input_exactly():
    s = EMASmoother(alpha=1.0)
    for x in [1.0, 2.0, 3.0, -5.0]:
        assert s.update(x) == x


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        EMASmoother(alpha=0.0)
    with pytest.raises(ValueError):
        EMASmoother(alpha=1.5)
    with pytest.raises(ValueError):
        EMASmoother(alpha=-0.1)


def test_smoothed_oscillating_signal_converges_to_mean():
    """The user's actual problem: loss bouncing 0.30..0.37. EMA with small
    alpha should converge to the mean and have low variance."""
    s = EMASmoother(alpha=0.02)
    for i in range(2000):
        # alternating high/low — same shape as the real chart
        x = 0.305 if i % 2 == 0 else 0.365
        s.update(x)
    # mean is 0.335; smoothed should be close
    assert math.isclose(s.value, 0.335, abs_tol=0.005)


def test_reset_clears_state():
    s = EMASmoother(alpha=0.5)
    s.update(10.0)
    s.update(20.0)
    s.reset()
    assert s.value is None
    assert s.update(7.0) == 7.0
