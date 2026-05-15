"""bracket-serve --event-path PATH [--host ...] [--port ...]"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from bracket.server import build_app
from bracket.tfevents_reader import scalar_tags_in


def serve() -> None:
    parser = argparse.ArgumentParser(prog="bracket-serve")
    parser.add_argument("--event-path", required=True, help="path to events.out.tfevents.* file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    tags = scalar_tags_in(args.event_path)
    print(f"Found scalar tags in event file: {tags}")
    from bracket.tfevents_reader import LOSS_TAG_CANDIDATES
    if not any(t in tags for t in LOSS_TAG_CANDIDATES):
        print(
            f"WARNING: none of {LOSS_TAG_CANDIDATES} present — frames will not be emitted."
        )

    app = build_app(
        event_path=args.event_path,
        ema_alpha=args.ema_alpha,
        poll_interval_s=args.poll_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    serve()
