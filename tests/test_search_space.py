import math
import random
import pytest

from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchOverrides,
    SearchSpace,
    apply_search_overrides,
)


def test_float_knob_uniform_within_range():
    rng = random.Random(0)
    k = FloatKnob(low=0.5, high=2.0)
    samples = [k.sample(rng) for _ in range(500)]
    assert all(0.5 <= s <= 2.0 for s in samples)
    assert min(samples) < 0.7 and max(samples) > 1.8  # spans the range


def test_float_knob_log_uniform_spans_orders_of_magnitude():
    rng = random.Random(0)
    k = FloatKnob(low=1e-6, high=1e-3, log=True)
    samples = [k.sample(rng) for _ in range(500)]
    # Log-uniform sampling spans 3 orders of magnitude; we should hit at
    # least 3 distinct integer log10 buckets in the [-6, -3] range.
    log_buckets = {math.floor(math.log10(s)) for s in samples}
    assert all(-6 <= b <= -3 for b in log_buckets), f"buckets out of range: {sorted(log_buckets)}"
    assert len(log_buckets) >= 3, f"only {len(log_buckets)} distinct buckets: {sorted(log_buckets)}"


def test_float_knob_log_requires_positive_low():
    with pytest.raises(ValueError):
        FloatKnob(low=0.0, high=1.0, log=True)


def test_float_knob_validate_rejects_out_of_range():
    k = FloatKnob(low=0.0, high=1.0)
    k.validate(0.5)
    with pytest.raises(ValueError):
        k.validate(1.5)


def test_int_knob_inclusive_endpoints():
    rng = random.Random(0)
    k = IntKnob(low=0, high=2)
    samples = [k.sample(rng) for _ in range(200)]
    assert set(samples) == {0, 1, 2}


def test_categorical_knob_returns_only_choices():
    rng = random.Random(0)
    k = CategoricalKnob(choices=("a", "b", "c"))
    samples = [k.sample(rng) for _ in range(100)]
    assert set(samples) <= {"a", "b", "c"}
    assert set(samples) == {"a", "b", "c"}  # all hit eventually


def test_fixed_knob_always_returns_value():
    rng = random.Random(0)
    k = FixedKnob(value=7)
    assert all(k.sample(rng) == 7 for _ in range(20))
    k.validate(7)
    with pytest.raises(ValueError):
        k.validate(8)


def test_search_space_sample_and_validate_roundtrip():
    space = SearchSpace(
        name="t",
        knobs={
            "lr": FloatKnob(low=1e-6, high=1e-3, log=True),
            "rank": IntKnob(low=4, high=64),
            "opt": CategoricalKnob(choices=("AdamW", "Lion")),
            "fixed": FixedKnob(value=1),
        },
    )
    rng = random.Random(0)
    s = space.sample(rng)
    assert set(s) == {"lr", "rank", "opt", "fixed"}
    space.validate(s)


def test_search_space_validate_rejects_missing_or_extra():
    space = SearchSpace(name="t", knobs={"a": FixedKnob(1)})
    with pytest.raises(ValueError):
        space.validate({})
    with pytest.raises(ValueError):
        space.validate({"a": 1, "b": 2})

def _trainer_like_space():
    return SearchSpace(
        name='sdxl-lora-v0.3-mid',
        knobs={
            'learning_rate': FloatKnob(low=1e-6, high=5e-4, log=True),
            'train_batch_size': CategoricalKnob(choices=(1, 2, 4, 8)),
            'gradient_checkpointing': CategoricalKnob(choices=(True, False)),
        },
    )


def test_apply_search_overrides_empty_is_noop():
    space = _trainer_like_space()
    out = apply_search_overrides(space, SearchOverrides())
    assert out is space


def test_apply_search_overrides_none_is_noop():
    space = _trainer_like_space()
    assert apply_search_overrides(space, None) is space


def test_lr_min_and_max_replace_float_knob():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(lr_min=1e-5, lr_max=1e-4),
    )
    lr = out.knobs['learning_rate']
    assert isinstance(lr, FloatKnob)
    assert lr.low == 1e-5 and lr.high == 1e-4 and lr.log is True
    assert out.name.endswith('+overrides')


def test_lr_min_only_keeps_trainer_max():
    space = _trainer_like_space()
    out = apply_search_overrides(space, SearchOverrides(lr_min=1e-5))
    lr = out.knobs['learning_rate']
    assert lr.low == 1e-5 and lr.high == 5e-4


def test_lr_min_above_max_is_normalised_not_thrown():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(lr_min=1e-4, lr_max=1e-5),
    )
    lr = out.knobs['learning_rate']
    assert lr.low == 1e-5 and lr.high == 1e-4


def test_batch_size_filters_categorical_choices():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(batch_size_min=2, batch_size_max=4),
    )
    bs = out.knobs['train_batch_size']
    assert isinstance(bs, CategoricalKnob)
    assert bs.choices == (2, 4)


def test_batch_size_falls_back_to_int_knob_when_no_choice_fits():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(batch_size_min=16, batch_size_max=32),
    )
    bs = out.knobs['train_batch_size']
    assert isinstance(bs, IntKnob)
    assert bs.low == 16 and bs.high == 32


def test_gradient_checkpointing_on_forces_fixed_true():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(gradient_checkpointing_mode='on'),
    )
    gc = out.knobs['gradient_checkpointing']
    assert isinstance(gc, FixedKnob) and gc.value is True


def test_gradient_checkpointing_off_forces_fixed_false():
    space = _trainer_like_space()
    out = apply_search_overrides(
        space, SearchOverrides(gradient_checkpointing_mode='off'),
    )
    gc = out.knobs['gradient_checkpointing']
    assert isinstance(gc, FixedKnob) and gc.value is False


def test_gradient_checkpointing_search_replaces_fixed_with_categorical():
    space = SearchSpace(
        name='tier-low',
        knobs={'gradient_checkpointing': FixedKnob(value=True)},
    )
    out = apply_search_overrides(
        space, SearchOverrides(gradient_checkpointing_mode='search'),
    )
    gc = out.knobs['gradient_checkpointing']
    assert isinstance(gc, CategoricalKnob) and set(gc.choices) == {True, False}


def test_overrides_for_missing_knobs_are_silently_ignored():
    space = SearchSpace(name='minimal', knobs={'learning_rate': FloatKnob(1e-6, 1e-4, log=True)})
    out = apply_search_overrides(
        space,
        SearchOverrides(batch_size_min=1, batch_size_max=2, gradient_checkpointing_mode='off'),
    )
    assert set(out.knobs) == {'learning_rate'}

