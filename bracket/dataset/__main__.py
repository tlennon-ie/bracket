#!/usr/bin/env python3
"""CLI for dataset validation and repair.

Usage:
    python -m bracket.dataset.validator /path/to/dataset.toml
    python -m bracket.dataset.validator --check /path/to/dataset.toml
    python -m bracket.dataset.validator --fix /path/to/dataset.toml
    python -m bracket.dataset.validator --remove-bad /path/to/dataset.toml
"""

from __future__ import annotations

import sys

# Re-export from the validator CLI module
if __name__ == "__main__":
    from bracket.dataset.validator_cli import main
    sys.exit(main())
