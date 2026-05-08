"""Importable wrapper around scripts/orchestrate.py so pyproject scripts can find it."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Delegate to the actual implementation in scripts/orchestrate.py
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import orchestrate as _impl  # type: ignore[import-not-found]
    return _impl.main()


if __name__ == "__main__":
    sys.exit(main())
