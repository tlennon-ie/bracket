"""Dump scalar tags + first/last values from a tfevents file. Use this to
sanity-check that the file you're about to point bracket-serve at
actually contains the expected loss/current series.

    python scripts/dump_tfevents.py PATH_TO_EVENTS_FILE
"""
from __future__ import annotations

import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(path: str) -> None:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    print(f"event file: {path}")
    print(f"scalar tags ({len(tags)}): {tags}")
    print()
    for tag in tags:
        evts = ea.Scalars(tag)
        if not evts:
            print(f"  {tag!r}: empty")
            continue
        print(
            f"  {tag!r}: n={len(evts)} "
            f"steps=[{evts[0].step}..{evts[-1].step}] "
            f"first={evts[0].value:.6f} last={evts[-1].value:.6f} "
            f"min={min(e.value for e in evts):.6f} "
            f"max={max(e.value for e in evts):.6f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
