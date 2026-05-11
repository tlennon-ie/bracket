"""Trainer abstraction.

A Trainer wraps an existing training script (sd-scripts/sdxl_train_network.py,
musubi-tuner/flux_2_train_network.py, ...) so the orchestrator can drive it
without knowing the script's internals. Each Trainer:

  - declares its own SearchSpace (which knobs are mutated and over what ranges)
  - knows a sensible baseline_config (what the user runs by hand today)
  - converts a config + dataset + budget into a LaunchSpec (cmd + env + paths)

The orchestrator then launches the LaunchSpec, observes its tfevents, scores
the result, and feeds the score back to the search controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class LaunchSpec:
    """Everything needed to launch a candidate training run as a subprocess."""

    cmd: list[str]
    cwd: Path
    env: Mapping[str, str]
    output_dir: Path
    logging_dir: Path
    tfevents_glob: str
    sample_dir: Optional[Path] = None


def make_accelerate_launch_prefix(
    venv_python: Path,
    *,
    mixed_precision: str = "bf16",
    num_cpu_threads_per_process: int = 1,
) -> list[str]:
    """Build the `accelerate launch` prefix to wrap a training script invocation.

    Replaces a plain `[venv_python, train_script.py, ...args]` invocation with
    `[venv_python, -m accelerate.commands.launch, ..., train_script.py, ...args]`.
    This matches musubi-tuner's and sd-scripts' documented launch incantation —
    which on Blackwell + PyTorch 2.11 makes the difference between step times
    of 2000s (plain python) and 2-3s (accelerate launch).

    Without `accelerate launch`, the script's `Accelerator()` constructor
    doesn't see the env vars (`LOCAL_RANK`, `WORLD_SIZE`, `ACCELERATE_*`, etc.)
    that accelerate relies on for fast-path init.
    """
    return [
        str(venv_python), "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", str(num_cpu_threads_per_process),
        "--mixed_precision", mixed_precision,
    ]


def resolve_save_every_n_steps(
    save_every_n_steps: Optional[int], *, max_steps: int,
) -> int:
    """Pick the right ``--save_every_n_steps`` value.

    Search runs (the default) only need a single save at the end —
    ``max(1, max_steps)``. Promoted full-training runs pass an explicit
    value (e.g. every 500 steps) to get intermediate checkpoints.
    """

    if save_every_n_steps and save_every_n_steps > 0:
        return int(save_every_n_steps)
    return max(1, int(max_steps))


def make_subprocess_env(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Build a subprocess env that prevents OMP/MKL/BLAS thread storms.

    Without these caps, every PyTorch worker (the trainer + each DataLoader
    worker) spawns one OMP thread per CPU core. On a 16-core machine with 6
    DataLoader workers, that's 7 × 16 = 112 OMP threads contending on 16
    physical cores. Catastrophic context switching → step times go from
    seconds to thousands of seconds.

    Equivalent to what `accelerate launch --num_cpu_threads_per_process 1`
    sets up for you. We do it explicitly here so the launch path is
    consistent regardless of how the trainer is invoked.

    Also disables tqdm. Reason: when stdout is captured by subprocess.PIPE,
    tqdm's `\r`-overwriting progress bars never emit newlines. Our pump
    reads line-by-line, so it never drains the buffer. The OS pipe (~64KB
    on Windows) fills with `\r` spam and the trainer's next stdout write
    BLOCKS — training appears to "freeze" at 2000+ s/step. Direct terminal
    runs don't see this because `\r` works against the actual TTY. We
    don't lose meaningful info — actual training progress comes from
    tfevents, which the Monitor tab reads.
    """
    import os
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    if extra:
        env.update(extra)
    return env


class TrainerConfig:
    """Marker base class. Each Trainer defines its own dataclass subclass."""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict, is_dataclass

        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass")
        return asdict(self)


class Trainer(ABC):
    """Adapter for a training script."""

    name: str

    @abstractmethod
    def declare_search_space(self) -> "SearchSpace":  # noqa: F821
        """Return the search space for this trainer."""

    @abstractmethod
    def baseline_config(self) -> TrainerConfig:
        """A known-decent config to compare orchestrator finds against."""

    @abstractmethod
    def config_from_dict(self, knobs: Mapping[str, Any]) -> TrainerConfig:
        """Hydrate a TrainerConfig from a sample produced by the search space.

        Allows the orchestrator's mutator to work on plain dicts without
        knowing each trainer's dataclass shape.
        """

    @abstractmethod
    def prepare_run(
        self,
        *,
        run_dir: Path,
        config: TrainerConfig,
        dataset_toml: Path,
        max_steps: int,
        seed: int,
        sample_prompts: Optional[Path] = None,
        sample_every_n_steps: Optional[int] = None,
        # Promote-flow extensions. Defaults preserve the original search-run
        # behavior (save only at end, no resume, no separate state dir):
        #   - ``save_every_n_steps``: emit ``--save_every_n_steps N``;
        #     None = use ``max_steps`` (save once at end, same as today).
        #   - ``save_state``: emit ``--save_state`` so optimizer / RNG /
        #     scheduler state is saved alongside weights at each
        #     ``save_every_n_steps`` boundary (needed for resume).
        #   - ``resume_from``: path to a previously-saved state directory
        #     OR a LoRA checkpoint to warm-start from.
        save_every_n_steps: Optional[int] = None,
        save_state: bool = False,
        resume_from: Optional[Path] = None,
    ) -> LaunchSpec:
        """Materialize all on-disk inputs and return the LaunchSpec."""

    def session_setup_commands(
        self, *, dataset_toml: Path, run_dir: Path,
    ) -> list[LaunchSpec]:
        """Optional hook — return commands to run ONCE per session, before the
        first candidate. Used by trainers that need separate pre-caching steps
        (musubi family). Default: empty list."""
        return []

    def curated_configs(self) -> list[TrainerConfig]:
        """Known-good configurations to try BEFORE the search controller takes
        over. Sourced from official docs, community recipes, or the user's own
        production scripts — anything documented to actually work.

        Listed in priority order: index 0 runs first. The orchestrator runs
        each curated config (with full seeds_per_config replication) until
        either the curated list is exhausted or the budget is consumed, then
        falls through to the SearchController for the remaining slots.

        Default: empty (orchestrator falls straight through to the controller,
        preserving the v0.x behaviour). Trainer subclasses override to seed
        the warm-start.
        """
        return []
