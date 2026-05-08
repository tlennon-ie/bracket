from bracket.search.controller import LedgerEntry, RandomSearch
from bracket.search.space import FloatKnob, IntKnob, SearchSpace


def _space():
    return SearchSpace(
        name="t",
        knobs={"lr": FloatKnob(low=1e-5, high=1e-3, log=True), "warmup": IntKnob(0, 100)},
    )


def test_random_search_is_deterministic_given_seed():
    a = RandomSearch(seed=42)
    b = RandomSearch(seed=42)
    space = _space()
    for _ in range(5):
        sa = a.next_config(space, [])
        sb = b.next_config(space, [])
        assert sa == sb


def test_random_search_stop_at_budget():
    c = RandomSearch(seed=0)
    history: list[LedgerEntry] = []
    for i in range(3):
        assert not c.should_stop(history, budget_runs=3)
        history.append(LedgerEntry(
            run_id=f"r{i}", config={}, score=0.1, error=None,
            duration_s=1.0, n_steps=10, timestamp=0.0,
        ))
    assert c.should_stop(history, budget_runs=3)
