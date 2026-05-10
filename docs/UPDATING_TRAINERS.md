# Updating bundled trainers

`musubi-tuner` and `sd-scripts` ship as **git submodules** under `vendor/`,
pinned to specific upstream commits in `.gitmodules`. End users get exactly
the version maintainers tested — never an unannounced upstream change.

Layout:

```
vendor/
├── musubi-tuner/        # https://github.com/kohya-ss/musubi-tuner @ pinned SHA
├── sd-scripts/          # https://github.com/kohya-ss/sd-scripts   @ pinned SHA
└── venv/                # trainer venv (.gitignored — built by install.*)
```

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
