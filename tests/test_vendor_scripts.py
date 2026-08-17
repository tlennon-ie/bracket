"""Guard: every vendored trainer script an adapter names actually exists.

Bracket drives four external trainers by *filename* — `musubi_tuner.<module>`,
`<musubi>/…/<x>_train_network.py`, `ai-toolkit/run.py`,
`ltx-trainer/scripts/train.py`. Nothing in Python checks those names until a
run launches, so an upstream rename lands as a `FileNotFoundError` in front of
a user who just queued an eight-hour sweep. That has happened before (PR #3,
"repair adapters referencing non-existent musubi scripts").

These tests re-check every name against the **actual** pinned submodules, so a
`vendor/*` bump that renames or drops a script fails here instead. They are
still unit-fast: nothing is executed, only resolved.

Everything skips when the submodules are not checked out — `actions/checkout`
does not fetch them by default, so CI sees an empty `vendor/`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from bracket.registry import PRESETS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _REPO_ROOT / "vendor"
_TRAINER_PKG = _REPO_ROOT / "bracket" / "trainer"

# Field name → the vendored checkout it must point at. A preset naming a field
# that is not here gets a dummy value (weight paths, task selectors, …).
_VENDOR_FIELDS: dict[str, Path] = {
    "musubi_dir": _VENDOR / "musubi-tuner",
    "sd_scripts_dir": _VENDOR / "sd-scripts",
    "aitk_dir": _VENDOR / "ai-toolkit",
    "trainer_dir": _VENDOR / "ltx2" / "packages" / "ltx-trainer",
}

# ``python -m musubi_tuner.<module>`` invocations embedded in the adapters.
_MUSUBI_MODULE_RE = re.compile(r"musubi_tuner\.([a-z0-9_]+)")


def _skip_unless(path: Path) -> None:
    if not path.is_dir():
        pytest.skip(f"submodule not checked out: {path.relative_to(_REPO_ROOT)}")


def _referenced_musubi_modules() -> set[str]:
    """Every ``musubi_tuner.<module>`` named anywhere in bracket/trainer/."""
    found: set[str] = set()
    for src in _TRAINER_PKG.glob("*.py"):
        found.update(_MUSUBI_MODULE_RE.findall(src.read_text(encoding="utf-8")))
    return found


def test_referenced_musubi_modules_exist():
    """Cache + train modules invoked as ``python -m musubi_tuner.<module>``.

    These are the ones with no ctor-time existence check at all — the adapter
    hands the module name to a subprocess and finds out at runtime.
    """
    musubi = _VENDOR_FIELDS["musubi_dir"]
    _skip_unless(musubi)
    pkg = musubi / "src" / "musubi_tuner"
    _skip_unless(pkg)

    modules = _referenced_musubi_modules()
    assert modules, "found no musubi_tuner.* references — did the regex rot?"
    missing = sorted(m for m in modules if not (pkg / f"{m}.py").is_file())
    assert not missing, (
        f"adapters reference musubi modules that do not exist at the pinned "
        f"commit: {missing}. Either the submodule bump renamed them or the "
        f"adapter has a typo."
    )


def _trainer_kwargs(preset) -> dict[str, object] | None:
    """Build ctor kwargs for a preset, or None if its vendor dir is absent.

    Values are heterogeneous — every field is a path string except ``vram_gb``,
    which the adapters expect as a float.
    """
    kwargs: dict[str, object] = {"vram_gb": 24.0}
    for f in preset.fields:
        if f.target != "trainer":
            continue
        if f.name in _VENDOR_FIELDS:
            vendor_dir = _VENDOR_FIELDS[f.name]
            if not vendor_dir.is_dir():
                return None
            kwargs[f.name] = str(vendor_dir)
        elif f.name == "venv_python":
            # Adapters assert this exists; the interpreter running the tests
            # is a real file and is never launched here.
            kwargs[f.name] = sys.executable
        else:
            # Weight paths and selectors are not validated at construction.
            kwargs[f.name] = f.default or "/nonexistent/placeholder"
    return kwargs


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_preset_resolves_its_train_script_against_the_pinned_vendor(preset):
    """Constructing a preset resolves its entrypoint in the real checkout.

    Every adapter locates its train script in ``__init__`` and raises
    ``FileNotFoundError`` when it is missing, so construction alone is the
    assertion.
    """
    kwargs = _trainer_kwargs(preset)
    if kwargs is None:
        pytest.skip(f"{preset.id}: required submodule not checked out")
    trainer = preset.trainer_factory(**kwargs)
    assert trainer.name


def test_every_preset_declares_a_known_vendor():
    """No preset may depend on a checkout this guard does not know about.

    Without this, adding a fifth vendor would silently opt every one of its
    presets out of the coverage above.
    """
    for preset in PRESETS:
        names = {f.name for f in preset.fields if f.target == "trainer"}
        assert names & set(_VENDOR_FIELDS), (
            f"{preset.id} names no vendor directory field — add its checkout "
            f"to _VENDOR_FIELDS so its entrypoint stays guarded"
        )
