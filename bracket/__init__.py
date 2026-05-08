"""Bracket — autonomous AutoML orchestrator for diffusion training.

Two layers:

  Telemetry (the eyes): tail tfevents written by accelerate, smooth the noisy
  per-step loss with EMA, fan out frames over WebSockets so dashboards or
  agents can subscribe.

  Orchestrator (the agent): wraps an existing trainer (sd-scripts/sdxl_train_network,
  musubi-tuner/flux_2_train_network, ...), samples configs from a SearchSpace,
  launches managed subprocess runs, scores them from their tfevents, records
  to a ledger, and at the end produces a Markdown proof report comparing the
  best find vs a hand-tuned baseline.
"""

# Telemetry layer
from bracket.frame import LossFrame
from bracket.ema import EMASmoother
from bracket.tfevents_reader import TFEventsTail, scalar_tags_in
from bracket.broadcaster import Broadcaster

# Orchestrator layer
from bracket.trainer import LaunchSpec, Trainer, TrainerConfig, SDXLTrainer, SDXLConfig
from bracket.search import (
    SearchSpace, Knob, FloatKnob, IntKnob, CategoricalKnob, FixedKnob,
    SearchController, RandomSearch, LedgerEntry,
    sample_random, perturb,
)
from bracket.dataset import DatasetSubsetSpec, build_subset
from bracket.orchestrator import (
    RunLauncher, RunResult, Scorer, ScoreReport, Ledger,
    orchestrate, OrchestrationResult,
)
from bracket.proof import generate_report

__all__ = [
    # Telemetry
    "LossFrame", "EMASmoother", "TFEventsTail", "scalar_tags_in", "Broadcaster",
    # Trainer
    "LaunchSpec", "Trainer", "TrainerConfig", "SDXLTrainer", "SDXLConfig",
    # Search
    "SearchSpace", "Knob", "FloatKnob", "IntKnob", "CategoricalKnob", "FixedKnob",
    "SearchController", "RandomSearch", "LedgerEntry", "sample_random", "perturb",
    # Dataset
    "DatasetSubsetSpec", "build_subset",
    # Orchestrator
    "RunLauncher", "RunResult", "Scorer", "ScoreReport", "Ledger",
    "orchestrate", "OrchestrationResult", "generate_report",
]

__version__ = "0.2.0"
