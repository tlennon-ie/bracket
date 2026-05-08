import random
import pytest

from bracket.search.mutator import perturb, sample_random
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)


def _space():
    return SearchSpace(
        name="t",
        knobs={
            "lr": FloatKnob(low=1e-6, high=1e-2, log=True),
            "warmup": IntKnob(low=0, high=200),
            "opt": CategoricalKnob(choices=("AdamW", "AdamW8bit", "Lion")),
            "rank": FixedKnob(value=32),
        },
    )


def test_sample_random_obeys_space():
    space = _space()
    rng = random.Random(0)
    for _ in range(20):
        s = sample_random(space, rng)
        space.validate(s)


def test_perturb_keeps_in_range_and_changes_some_keys():
    space = _space()
    rng = random.Random(1)
    base = {"lr": 1e-4, "warmup": 50, "opt": "AdamW8bit", "rank": 32}
    out = perturb(base, space, rng, sigma=0.5, fraction_changed=1.0)
    space.validate(out)
    assert out["rank"] == 32  # fixed knob never moves
    differences = [k for k in base if base[k] != out.get(k)]
    assert differences, "perturb with fraction=1.0 should change at least one knob"


def test_perturb_log_float_stays_positive_and_in_range():
    space = SearchSpace(name="t", knobs={"lr": FloatKnob(low=1e-7, high=1e-2, log=True)})
    rng = random.Random(0)
    out = perturb({"lr": 1e-4}, space, rng, sigma=2.0, fraction_changed=1.0)
    assert 1e-7 <= out["lr"] <= 1e-2


def test_perturb_validates_input_base():
    space = _space()
    rng = random.Random(0)
    with pytest.raises(ValueError):
        perturb({"lr": 1e-4}, space, rng)  # missing keys
