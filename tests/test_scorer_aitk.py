"""Scorer tests for the ai-toolkit (ostris) SQLite-loss path.

ai-toolkit produces no tfevents. With ``logging.use_ui_logger: true`` it writes
a SQLite ``loss_log.db`` (see ``test_sqlite_loss_reader.py`` for the schema).
The scorer must read the loss curve from ``loss_db_path`` and run the SAME
divergence / slope / NaN disqualification logic as the tfevents / stdout-log
paths.

ai-toolkit also writes validation samples to ``<save_root>/samples/`` named
``<timestamp_ms>__<step_zfill9>_<count>.<ext>`` where ``<count>`` is the
0-BASED prompt index (no sidecar ``.txt``). Pairing must map those to the
right prompt without regressing the sd-scripts / musubi / LTX-2 conventions.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

from bracket.orchestrator.scorer import (
    Scorer,
    _extract_prompt_idx,
    _pair_samples_with_prompts,
)


def _build_loss_db(
    path: Path,
    rows: Sequence[tuple[int, float, Optional[float]]],
) -> Path:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE metrics ("
            "step INTEGER, key TEXT, value_real REAL, value_text TEXT, "
            "PRIMARY KEY(step, key))"
        )
        conn.execute(
            "CREATE TABLE metric_keys ("
            "key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)"
        )
        for step, loss, lr in rows:
            conn.execute(
                "INSERT INTO steps (step, wall_time) VALUES (?, ?)",
                (step, 1000.0 + step),
            )
            conn.execute(
                "INSERT INTO metrics (step, key, value_real, value_text) "
                "VALUES (?, 'loss', ?, NULL)",
                (step, loss),
            )
            if lr is not None:
                conn.execute(
                    "INSERT INTO metrics (step, key, value_real, value_text) "
                    "VALUES (?, 'lr', ?, NULL)",
                    (step, lr),
                )
        conn.commit()
    finally:
        conn.close()
    return path


def _sample_at(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)
    return p


# --- (i) loss scoring from a SQLite db -------------------------------------


def test_score_run_from_db_converging_is_not_disqualified(tmp_path: Path):
    rows = [(20 * i, 0.5 - 0.003 * i, 1.0e-4) for i in range(1, 101)]
    db = _build_loss_db(tmp_path / "loss_log.db", rows)
    rep = Scorer(positive_slope_kill_threshold=0.1).score_run(
        tfevents_path=None, loss_db_path=db, loss_source="sqlite_db",
    )
    assert rep.disqualified is None
    assert math.isfinite(rep.score)
    assert rep.n_steps == 100
    assert rep.components["slope"] < 0  # going down
    assert "loss_only" in rep.components


def test_score_run_from_db_diverging_is_disqualified(tmp_path: Path):
    rows = [(20 * i, 0.3 + 1.0 * i, 1.0e-4) for i in range(1, 41)]
    db = _build_loss_db(tmp_path / "loss_log.db", rows)
    rep = Scorer(positive_slope_kill_threshold=0.01).score_run(
        tfevents_path=None, loss_db_path=db, loss_source="sqlite_db",
    )
    assert rep.score == math.inf
    assert rep.disqualified is not None and rep.disqualified.startswith("diverging")


def test_score_run_from_db_nan_loss_disqualifies(tmp_path: Path):
    rows = [(20, 0.5, 1.0e-4), (40, 0.4, 1.0e-4), (60, 0.3, 1.0e-4),
            (80, float("nan"), 1.0e-4)]
    db = _build_loss_db(tmp_path / "loss_log.db", rows)
    rep = Scorer().score_run(
        tfevents_path=None, loss_db_path=db, loss_source="sqlite_db",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "nan_loss"


def test_score_run_no_loss_db_disqualifies(tmp_path: Path):
    rep = Scorer().score_run(
        tfevents_path=None, loss_db_path=None, loss_source="sqlite_db",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "no_loss_db"


def test_score_run_missing_db_file_disqualifies(tmp_path: Path):
    rep = Scorer().score_run(
        tfevents_path=None,
        loss_db_path=tmp_path / "nope.db",
        loss_source="sqlite_db",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "no_loss_db"


def test_score_run_empty_loss_db_disqualifies(tmp_path: Path):
    # A valid db with steps but no loss rows.
    db = tmp_path / "loss_log.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE metrics ("
            "step INTEGER, key TEXT, value_real REAL, value_text TEXT, "
            "PRIMARY KEY(step, key))"
        )
        conn.execute("INSERT INTO steps (step, wall_time) VALUES (10, 1010.0)")
        conn.commit()
    finally:
        conn.close()
    rep = Scorer().score_run(
        tfevents_path=None, loss_db_path=db, loss_source="sqlite_db",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "empty_loss_db"


def test_score_run_tfevents_trainer_ignores_loss_db(tmp_path: Path):
    """A tfevents trainer (loss_source default) that wrote no events must keep
    the historical ``no_tfevents`` DQ — even if a loss db happens to exist."""
    db = _build_loss_db(
        tmp_path / "loss_log.db",
        [(20 * i, 0.5 - 0.003 * i, 1.0e-4) for i in range(1, 51)],
    )
    rep = Scorer().score_run(
        tfevents_path=None, loss_db_path=db,  # loss_source defaults to "tfevents"
    )
    assert rep.score == math.inf
    assert rep.disqualified == "no_tfevents"


# --- (ii) ai-toolkit 0-based filename pairing ------------------------------


def test_extract_prompt_idx_aitk_format():
    # <timestamp_ms>__<step_zfill9>_<count> with 0-based count.
    assert _extract_prompt_idx("1724085406830__000000500_0") == 0
    assert _extract_prompt_idx("1724085406830__000000500_2") == 2


def test_pair_samples_aitk_format(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "1724085406830__000000500_0.jpg")  # 0-based idx 0 → A
    _sample_at(sd, "1724085406831__000000500_1.jpg")  # 0-based idx 1 → B
    _sample_at(sd, "1724085406832__000000500_2.jpg")  # 0-based idx 2 → C
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert pairing[sd / "1724085406830__000000500_0.jpg"] == "A"
    assert pairing[sd / "1724085406831__000000500_1.jpg"] == "B"
    assert pairing[sd / "1724085406832__000000500_2.jpg"] == "C"


def test_pair_samples_aitk_png_format(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "1724085406830__000000500_0.png")
    _sample_at(sd, "1724085406831__000000500_1.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B"])
    assert pairing[sd / "1724085406830__000000500_0.png"] == "A"
    assert pairing[sd / "1724085406831__000000500_1.png"] == "B"


# --- (iii) no regression on existing conventions ---------------------------


def test_pair_samples_sdscripts_still_pairs(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "candidate_000200_0_42.png")
    _sample_at(sd, "candidate_000200_1_42.png")
    _sample_at(sd, "candidate_000200_2_42.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert len(pairing) == 3
    assert pairing[sd / "candidate_000200_0_42.png"] == "A"
    assert pairing[sd / "candidate_000200_2_42.png"] == "C"


def test_extract_prompt_idx_sdscripts_unchanged():
    assert _extract_prompt_idx("candidate_000200_0_42") == 0
    assert _extract_prompt_idx("candidate_000200_2_42") == 2


def test_extract_prompt_idx_ltx2_unchanged():
    # LTX-2 1-based step format still maps correctly.
    assert _extract_prompt_idx("step_000100_3") == 2
    assert _extract_prompt_idx("step_000020_1") == 0
    assert _extract_prompt_idx("step_000100_3_frame00") == 2


# --- (iv) animated-WebP video samples (LTX-2.5 / MiniMax-H3) ---------------
#
# ai-toolkit forces `output_ext = webp` whenever num_frames > 1, so its video
# models write an ANIMATED WebP where every other video trainer writes .mp4.
# `.webp` also matches the still-image sample regex, so without the
# is_frame_extractable() guard the judge would score the container's first
# frame *in addition to* the frames extracted from it — double-counting one
# frame and diluting the run's mean with a duplicate.

import pytest


def _write_animated_webp(path: Path, *, frames: int = 4) -> Path:
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    imgs = [Image.new("RGB", (16, 16), (i * 30 % 256, 50, 90)) for i in range(frames)]
    imgs[0].save(
        path, format="WEBP", save_all=True, append_images=imgs[1:], duration=100,
    )
    return path


def test_pair_samples_skips_animated_webp_container(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    clip = _write_animated_webp(sd / "1724085406830__000000500_0.webp")
    pairing = _pair_samples_with_prompts(sd, ["A", "B"])
    assert clip not in pairing, (
        "the animated container was judged as a still — its first frame would "
        "be scored twice once _frames/ is populated"
    )


def test_pair_samples_keeps_still_webp(tmp_path: Path):
    """A single-frame WebP is an ordinary image sample and must still pair."""
    sd = tmp_path / "samples"; sd.mkdir()
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    still = sd / "1724085406830__000000500_0.webp"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(still, format="WEBP")
    pairing = _pair_samples_with_prompts(sd, ["A", "B"])
    assert pairing[still] == "A"


def test_pair_samples_uses_frames_extracted_from_animated_webp(tmp_path: Path):
    """Extracted frames inherit the container's 0-based ai-toolkit index."""
    sd = tmp_path / "samples"; sd.mkdir()
    _write_animated_webp(sd / "1724085406831__000000500_1.webp")
    frames_dir = sd / "_frames"; frames_dir.mkdir()
    for i in range(3):
        _sample_at(frames_dir, f"1724085406831__000000500_1_frame{i:02d}.png")

    pairing = _pair_samples_with_prompts(sd, ["A", "B"])
    paired = {p.name: prompt for p in pairing for prompt in [pairing[p]]}
    assert paired == {
        "1724085406831__000000500_1_frame00.png": "B",
        "1724085406831__000000500_1_frame01.png": "B",
        "1724085406831__000000500_1_frame02.png": "B",
    }
