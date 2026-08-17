# Updating bundled trainers

`musubi-tuner`, `sd-scripts`, `ltx2`, and `ai-toolkit` ship as **git
submodules** under `vendor/`, pinned to specific upstream commits in
`.gitmodules`. End users get exactly the version maintainers tested — never an
unannounced upstream change.

Layout:

```
vendor/
├── musubi-tuner/        # https://github.com/kohya-ss/musubi-tuner @ pinned SHA
├── sd-scripts/          # https://github.com/kohya-ss/sd-scripts   @ pinned SHA
├── ltx2/                # https://github.com/Lightricks/LTX-2       @ pinned SHA
│   └── packages/ltx-trainer/  # native LTX-2 trainer (uv-managed, own .venv)
├── ai-toolkit/          # https://github.com/ostris/ai-toolkit     @ pinned SHA
├── ai-toolkit-venv/     # ai-toolkit venv (.gitignored — built by install.*)
└── venv/                # shared trainer venv (.gitignored — built by install.*)
```

## LTX-2 (native Lightricks `ltx-trainer`)

LTX-2 is the `vendor/ltx2` submodule (the [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
repo). Its trainer lives at `vendor/ltx2/packages/ltx-trainer` and is a
**`uv`-managed project** — unlike `musubi-tuner` / `sd-scripts`, it does *not*
share `vendor/venv`. The installer runs `uv sync` inside
`packages/ltx-trainer` to create its own `.venv`, and skips the step (with a
warning, not a failure) if `uv` is not installed. Training is launched via
`uv run python scripts/train.py <config>.yaml`.

### One-time maintainer step — create the submodule gitlink

The installer only `--init`s an *already-declared* submodule; it never adds
one. The `vendor/ltx2` gitlink must be created once by a maintainer **with
network access** (this is a slow clone that pins a commit):

```bash
git submodule add https://github.com/Lightricks/LTX-2 vendor/ltx2
git commit -m "deps: add ltx2 (native LTX-2 trainer) submodule"
```

Until that gitlink exists, the `.gitmodules` entry is harmless and the
installer's `git submodule update --init --recursive` simply no-ops for it.

## ai-toolkit (ostris/ai-toolkit)

ai-toolkit is the `vendor/ai-toolkit` submodule (the [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
repo). Unlike the `uv`-managed `ltx-trainer`, it is a **pip/venv-managed
project**: it ships a `requirements.txt` and its dependencies conflict with the
shared `vendor/venv`, so it does *not* share it. The installer creates a
dedicated venv at `vendor/ai-toolkit-venv`, installs the GPU-matched PyTorch
wheel into it, then runs `pip install -r requirements.txt`. It skips the step
(with a warning, not a failure) if the `vendor/ai-toolkit` gitlink hasn't been
created yet.

### One-time maintainer step — create the submodule gitlink

The installer only `--init`s an *already-declared* submodule; it never adds
one. The `vendor/ai-toolkit` gitlink must be created once by a maintainer
**with network access** (this is a slow clone that pins a commit):

```bash
git submodule add https://github.com/ostris/ai-toolkit vendor/ai-toolkit
git commit -m "deps: add ai-toolkit submodule"
```

Until that gitlink exists, the `.gitmodules` entry is harmless and the
installer's `git submodule update --init --recursive` simply no-ops for it.

## End-user flow (just install or pull)

Nothing manual. The installer runs `git submodule update --init --recursive`
and the bundled UI's self-updater (`update.bat` / `update.sh`) does the same
on every update. After a `git pull` outside the updater, run:

```bash
git submodule update --init --recursive
```

## Maintainer flow — bumping a trainer to a newer upstream commit

Only do this when you've smoke-tested the new commit against at least one
real training run (Z-Image full FT is the cheapest sanity check).

### Bump musubi-tuner

```bash
cd vendor/musubi-tuner
git fetch origin
git checkout <new-SHA-or-tag>      # e.g. git checkout main && git pull
cd ../..
git add vendor/musubi-tuner
git commit -m "deps: bump musubi-tuner to <SHA-or-tag>"
```

### Bump sd-scripts

```bash
cd vendor/sd-scripts
git fetch origin
git checkout <new-SHA-or-tag>
cd ../..
git add vendor/sd-scripts
git commit -m "deps: bump sd-scripts to <SHA-or-tag>"
```

### Bump ltx2

```bash
cd vendor/ltx2
git fetch origin
git checkout <new-SHA-or-tag>
cd ../..
git add vendor/ltx2
git commit -m "deps: bump ltx2 to <SHA-or-tag>"
# Re-sync the uv-managed trainer env to pick up any dependency changes:
( cd vendor/ltx2/packages/ltx-trainer && uv sync )
```

### Bump ai-toolkit

```bash
cd vendor/ai-toolkit
git fetch origin
git checkout <new-SHA-or-tag>
cd ../..
git add vendor/ai-toolkit
git commit -m "deps: bump ai-toolkit to <SHA-or-tag>"
# Re-run the installer so the ai-toolkit venv picks up any new
# requirements.txt entries:
./install.sh        # or .\install.ps1 on Windows
```

After an ai-toolkit bump, re-diff two upstream files against
[`bracket/trainer/aitk_profiles.py`](../bracket/trainer/aitk_profiles.py):

| Upstream file | What to check |
|---|---|
| `ui/src/app/jobs/new/options.tsx` | The per-arch default table our profiles transcribe. Changed `qtype`, adapter paths, frame counts, or guidance handling belong in the profile. |
| `extensions_built_in/diffusion_models/__init__.py` and `extensions_built_in/audio_models/` | The registered arch list. New entries here are candidate presets; a *removed* arch silently breaks the preset that names it. |

Profiles carry no version guard — an `arch` string ai-toolkit stopped
recognising fails at trainer launch, not at import. `pytest -q
tests/test_trainer_aitk_media.py` covers the config shape, not upstream's
acceptance of it, so a bump that touches archs deserves one real smoke run.

### Verify before pushing

```bash
# Re-run the installer so the venv picks up any new requirements.txt entries.
./install.sh        # or .\install.ps1 on Windows

# Run the unit tests.
pytest -q

# Smoke-test against a real dataset (small, fast model preferred).
bracket --help
# ... run a short orchestration session via the UI or CLI
```

If anything breaks, revert the submodule bump:

```bash
git checkout HEAD~ -- vendor/musubi-tuner
# or back to a known-good SHA
cd vendor/musubi-tuner && git checkout <old-SHA> && cd ../..
git add vendor/musubi-tuner && git commit -m "revert: musubi-tuner bump"
```

## Why submodules instead of `git clone` at install time?

The previous installer cloned `kohya-ss/musubi-tuner` and
`kohya-ss/sd-scripts` from `main` into `~/.cache/bracket/trainers/` at
install time. Two problems:

1. **Reproducibility.** Two users installing on different days got
   different upstream HEADs. A breaking upstream change between Tuesday
   and Friday meant Friday's user couldn't run jobs that worked for
   Tuesday's. With submodules, every user pulls exactly the same SHAs.
2. **Discoverability.** The trainer code lived outside the repo, so a
   `git log` couldn't tell you which trainer version a Bracket release
   was tested against. Now the pin is part of the commit history.

## Why is the trainer venv in-repo (`vendor/venv`) but ignored?

Two practical reasons:
- **Single delete to nuke and re-install.** `rm -rf vendor/venv` followed
  by `./install.sh` is the full reset path.
- **Co-located with the code it serves.** No hunting in
  `%LOCALAPPDATA%\bracket\trainers\venv` to find the right Python.

The `vendor/venv/` path is in `.gitignore` so the multi-GB CUDA wheel
install never gets committed by accident.

Legacy installs at `~/.cache/bracket/trainers/` still work — `BRACKET_*`
env vars in your existing `.env` continue to override the new defaults.
