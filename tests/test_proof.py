from pathlib import Path

from bracket.orchestrator.ledger import Ledger
from bracket.proof.report import generate_report


def _seed_ledger(path: Path) -> Ledger:
    ledger = Ledger(path)
    ledger.append({
        "run_id": "baseline-000-1", "role": "baseline",
        "config": {"learning_rate": 1e-4, "optimizer_type": "AdamW8bit", "lr_warmup_steps": 100,
                   "network_dim": 32, "network_alpha": 16.0, "noise_offset": 0.0,
                   "lr_scheduler": "cosine", "train_batch_size": 1,
                   "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
                   "max_grad_norm": 1.0},
        "score": 0.50, "score_components": {"final_smoothed": 0.48, "slope": -0.02},
        "n_steps": 100, "duration_s": 60.0, "timestamp": 1.0,
    })
    ledger.append({
        "run_id": "cand-000-2", "role": "candidate",
        "config": {"learning_rate": 5e-5, "optimizer_type": "Lion", "lr_warmup_steps": 50,
                   "network_dim": 16, "network_alpha": 8.0, "noise_offset": 0.05,
                   "lr_scheduler": "constant", "train_batch_size": 1,
                   "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
                   "max_grad_norm": 1.0},
        "score": 0.42, "score_components": {"final_smoothed": 0.40, "slope": -0.02},
        "n_steps": 100, "duration_s": 65.0, "timestamp": 2.0,
    })
    ledger.append({
        "run_id": "cand-001-3", "role": "candidate",
        "config": {"learning_rate": 1e-2}, "score": None,
        "score_components": {}, "n_steps": 0, "duration_s": 5.0,
        "timestamp": 3.0, "disqualified": "diverging(slope=0.5)",
    })
    return ledger


def test_proof_report_marks_orchestrator_as_winner(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger_path)
    out = generate_report(ledger_path, tmp_path / "report.md")
    body = out.read_text(encoding="utf-8")
    assert "Orchestrator beat baseline" in body
    assert "cand-000-2" in body
    assert "Top-5" in body
    # Disqualified candidate appears in disqualifications section
    assert "diverging" in body


def test_proof_report_includes_plain_language_explanations(tmp_path: Path):
    """The report must explain score/mean/stdev in plain English so users
    don't stare at a table of decimals wondering what they mean."""
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger_path)
    body = generate_report(ledger_path, tmp_path / "r.md").read_text(encoding="utf-8")
    assert "How to read this report" in body
    assert "lower = better" in body
    assert "average" in body.lower()  # mean explanation
    assert "stdev" in body.lower()
    # Headline must include a plain-English summary line
    assert "% improvement" in body or "% worse" in body


def test_proof_report_surfaces_judge_state(tmp_path: Path):
    """When no run was judged, the report must say so explicitly so users
    don't assume LMStudio scoring happened when it didn't."""
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger_path)  # seeded rows have judge_report missing
    body = generate_report(ledger_path, tmp_path / "r.md").read_text(encoding="utf-8")
    assert "judge not configured" in body or "loss only" in body.lower()


def test_proof_report_training_runs_excludes_setup_rows(tmp_path: Path):
    """The total count must report training runs only — setup rows (latent
    caching, TE caching) are infrastructure, not trials."""
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger_path)
    Ledger(ledger_path).append({
        "run_id": "setup-000", "role": "setup", "config": {},
        "score": None, "score_components": {}, "n_steps": 0,
        "duration_s": 5.0, "timestamp": 0.5, "exit_code": 0,
    })
    body = generate_report(ledger_path, tmp_path / "r.md").read_text(encoding="utf-8")
    # 3 training rows seeded; setup row should NOT inflate the count to 4.
    assert "Training runs: **3**" in body


def test_proof_report_handles_no_baseline(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append({
        "run_id": "cand-000", "role": "candidate", "config": {"lr": 1e-4},
        "score": 0.3, "score_components": {}, "n_steps": 50,
        "duration_s": 10.0, "timestamp": 1.0,
    })
    out = generate_report(ledger_path, tmp_path / "r.md")
    assert "No baseline" in out.read_text(encoding="utf-8")


def test_proof_report_empty_ledger(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    Ledger(ledger_path)
    out = generate_report(ledger_path, tmp_path / "r.md")
    assert "No runs" in out.read_text(encoding="utf-8")
