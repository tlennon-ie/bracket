"""Backwards-compat shim. Real CLI now lives at bracket.cli.orchestrate."""
from __future__ import annotations

from bracket.cli.orchestrate import main


if __name__ == "__main__":
    import sys
    sys.exit(main())
