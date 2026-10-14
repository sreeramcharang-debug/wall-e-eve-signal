from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mission import run_mission
from .report import format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WALL-E searches for EVE's signal in a simulated sector."
    )
    parser.add_argument("mission", help="Path to a mission JSON file")
    parser.add_argument(
        "--json",
        dest="json_out",
        help="Optional path to write the full JSON report",
    )
    args = parser.parse_args(argv)

    report = run_mission(args.mission)
    print(format_report(report))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {out}")
    return 0
