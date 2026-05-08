"""``bracket-serve`` CLI entry point.

Exposes the FastAPI surface in :mod:`bracket.api.server` over uvicorn.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from bracket.api.server import serve


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bracket-serve",
        description="Run the Bracket API for the React frontend.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--reload", action="store_true",
        help="Dev-mode auto-reload (uvicorn factory + watchfiles).",
    )
    parser.add_argument(
        "--cors", default=None,
        help=(
            "Override BRACKET_CORS_ORIGINS. Comma-separated origins, e.g. "
            '"http://localhost:5173,http://127.0.0.1:5173". '
            "Empty string = same-origin only."
        ),
    )
    parser.add_argument(
        "--log-level", default="info",
        help="Root logger level (debug/info/warning/error).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cors_origins: Optional[list[str]] = None
    if args.cors is not None:
        cors_origins = [o.strip() for o in args.cors.split(",") if o.strip()]
    serve(host=args.host, port=args.port, reload=args.reload, cors_origins=cors_origins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
