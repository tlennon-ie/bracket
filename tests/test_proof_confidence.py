"""Confidence-stat tests for the proof report — Welch's t-test, CI computation,
and the multi-seed render path."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from bracket.orchestrator.ledger import Ledger
from bracket.orchestrator.loop import config_id
from bracket.proof.report import (
    generate_report,
    welch_ci_diff,
    welch_t_test,
)


def test_welch_t_clearly_separated_groups_low_p():
    a = [0.10, 0.11, 0.10, 0.09]
    b = [0.30, 0.31, 0.29, 0.30]
    p = welch_t_test(a, b)
    assert 0.0 <= p < 0.01, f"expected p<0.01 for clearly different groups; got {p}"


def test_welch_t_overlapping_groups_high_p():
    a = [0.30, 0.32, 0.31]
    b = [0.31, 0.30, 0.32]
    p = welch_t_test(a, b)
    assert 0.5 < p <= 1.0, f"expected p>0.5 for indistinguishable groups; got {p}"


def test_welch_t_handles_single_sample_returns_nan():
    p = welch_t_test([0.1], [0.2, 0.3])
    assert math.isnan(p)


def test_welch_ci_contains_zero_for_identical_groups():
    a = [0.3, 0.3, 0.3]
    b = [0.3, 0.3, 0.3]
    lo, hi = welch_ci_diff(a, b)
    # Both variances zero → CI is degenerate; just check bounds bracket zero
    assert lo <= 0 <= hi or (lo == 0 and hi == 0)


def test_welch_ci_separated_groups_excludes_zero():
    a = [0.10, 0.11, 0.10]
    b = [0.30, 0.31, 0.29]
    lo, hi = welch_ci_diff(a, b)
    assert hi < 0  # mean(a) - mean(b) is decisively negative


def _seed_multi_seed_ledger(path: Path) -> None:
    ledger = Ledger(path)
    base_cfg = {"learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "lr_warmup_steps": 100,
                "network_dim": 32, "network_alpha": 16.0, "noise_offset": 0.0,
                "lr_scheduler": "cosine", "train_batch_size": 4,
                "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
                "max_grad_norm": 1.0}
    cand_cfg = {**base_cfg, "learning_rate": 5e-5, "optimizer_type": "Lion", "noise_offset": 0.05}
    base_id = config_id(base_cfg)
    cand_id = config_id(cand_cfg)
    # Baseline: 3 seeds, mean=0.20, low variance
    for i, score in enumerate([0.20, 0.21, 0.19]):
        ledger.append({
            "run_id": f"baseline-000-s{i}-{i}", "role": "baseline",
            "config_id": base_id, "config": base_cfg, "seed": 42 + i, "seed_idx": i,
            "score": score, "score_components": {"final_smoothed": score, "slope": 0.0},
            "n_steps": 100, "duration_s": 60.0, "timestamp": float(i),
        })
    # Candidate: 3 seeds, mean=0.10, low variance — clearly better
    for i, score in enumerate([0.10, 0.11, 0.09]):
        ledger.append({
            "run_id": f"cand-000-s{i}-{10+i}", "role": "candidate",
            "config_id": cand_id, "config": cand_cfg, "seed": 142 + i, "seed_idx": i,
            "score": score, "score_components": {"final_smoothed": score, "slope": 0.0},
            "n_steps": 100, "duration_s": 60.0, "timestamp": float(10 + i),
        })


def test_proof_report_includes_confidence_when_multiseed(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_multi_seed_ledger(ledger_path)
    out = generate_report(ledger_path, tmp_path / "report.md")
    body = out.read_text(encoding="utf-8")
    assert "Multi-seed" in body
    assert "Welch's t-test" in body
    assert "CI for" in body
    # Clearly-separated means should land at high confidence
    assert "high confidence" in body
    assert "Orchestrator beat baseline" in body


def test_proof_report_skips_confidence_when_single_seed(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    base_cfg = {"learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "lr_warmup_steps": 100,
                "network_dim": 32, "network_alpha": 16.0, "noise_offset": 0.0,
                "lr_scheduler": "cosine", "train_batch_size": 1,
                "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
                "max_grad_norm": 1.0}
    ledger.append({
        "run_id": "baseline-000-s0-1", "role": "baseline",
        "config_id": config_id(base_cfg), "config": base_cfg,
        "seed": 42, "seed_idx": 0,
        "score": 0.5, "score_components": {}, "n_steps": 50, "duration_s": 5.0, "timestamp": 1.0,
    })
    out = generate_report(ledger_path, tmp_path / "r.md")
    body = out.read_text(encoding="utf-8")
    assert "Single-seed" in body
    assert "Welch's t-test" not in body
