"""Wire format for one observation of training state.

Designed so v0.1 (tfevents-derived, no timestep info) and v0.2 (in-process hook,
timestep-conditioned) produce the same schema. Optional fields are None when the
source can't supply them; consumers shouldn't crash on missing data.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class LossFrame(NamedTuple):
    step: int
    raw_loss: float
    smoothed_loss: float          # EMA-smoothed (alpha configurable)
    lr: Optional[float]           # learning rate at this step
    wall_time: float              # seconds since epoch, from tfevents or time.time()
    timestep: Optional[int]       # diffusion timestep — None for tfevents source (v0.1)
    timestep_bucket: Optional[int]  # coarse bin of timestep (e.g. 0..9 for 10 buckets)
    grad_norm: Optional[float]    # None until v0.2 in-process hook lands

    def to_json(self) -> dict:
        return {
            "step": self.step,
            "raw_loss": self.raw_loss,
            "smoothed_loss": self.smoothed_loss,
            "lr": self.lr,
            "wall_time": self.wall_time,
            "timestep": self.timestep,
            "timestep_bucket": self.timestep_bucket,
            "grad_norm": self.grad_norm,
        }


def bucket_timestep(timestep: int, num_buckets: int = 10, max_timestep: int = 1000) -> int:
    """Bin a diffusion timestep into one of `num_buckets` coarse buckets.

    Diffusion training samples timesteps from [0, max_timestep). Loss magnitude
    is highly correlated with timestep — early/late steps train differently —
    so a bucketed view makes per-region drift visible.
    """
    if timestep < 0 or timestep >= max_timestep:
        raise ValueError(f"timestep {timestep} out of range [0, {max_timestep})")
    bucket = (timestep * num_buckets) // max_timestep
    return min(bucket, num_buckets - 1)
