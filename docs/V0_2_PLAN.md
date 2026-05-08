# v0.2 plan — in-process step-callback hook

## Why

`tfevents` is a great zero-touch ingestion path for v0.1, but the diffusion
timestep is averaged into the per-step loss *inside the training loop* before
anything is logged. The information path:

```
loss = mse(pred, target, reduction="none")   # tensor [B, ...]   ← timestep info still attached via batch axis
loss = loss * weighting                      # tensor [B, ...]
loss = loss.mean()                           # scalar           ← timestep info gone
accelerator.log({"loss/current": loss})      # → tfevents
```

By the time the value reaches tfevents the per-sample timesteps are
unrecoverable. The "smoothed timestep-conditioned loss" the brief asks for
needs the data captured one or two lines earlier.

## Where the hook goes

In `zimage_train.py` between the per-element loss computation
(line ~533) and the `loss.mean()` reduction (line ~538). The minimal patch is
~5 lines that compute the per-sample loss (mean over spatial dims, keep batch
dim) and emit `(step, timesteps_tensor, per_sample_loss_tensor)` to a queue.

## Two integration shapes

**(a) Monkey-patch / wrapper module** — `bracket.attach()` import
that replaces `ZImageTrainer.train` with a wrapper whose loss computation
forks a per-sample copy before the mean. Pros: zero edits to musubi_tuner.
Cons: brittle if the upstream loop changes.

**(b) Tiny upstream patch** — three or four lines added inside
`zimage_train.py`:

```python
loss_per_sample = (loss.mean(dim=tuple(range(1, loss.ndim))))  # [B]
bracket_callback(global_step, timesteps, loss_per_sample)    # no-op if not registered
loss = loss.mean()
```

Plus an entrypoint in our package to register the callback.

I lean toward **(b)** with the callback being a *no-op when not registered*,
so it costs nothing for users not running Bracket.

## What v0.2 exposes

Same `LossFrame` schema, with `timestep` and `timestep_bucket` populated.
Multiple frames per global_step (one per sample in the batch) *or* one frame
per global_step with a per-bucket loss dict — TBD; the schema can carry
either.

The dashboard adds a per-bucket heatmap (10 buckets × N steps) showing
which slice of the noise schedule is converging vs. struggling.

## What does NOT change

- WebSocket wire format
- EMASmoother (we EMA per-bucket independently)
- Broadcaster
- CLI surface

v0.1 dashboards keep working; the per-bucket panel is additive.

## Open questions for the user

1. Are you OK with a small patch to `zimage_train.py` (option b above), or
   do you want zero-touch (option a)?
2. Per-sample frames or per-bucket aggregates over the wire? Per-sample is
   richer but at batch=2 and 8000 steps that's 16k extra frames; per-bucket
   is 80k frames (10 buckets × 8000 steps) but each is summarised. I'd
   default to per-bucket aggregates.
3. Do you also want `grad_norm` exposed? It's already computed when
   `args.max_grad_norm != 0` — capturing is a one-line add.
