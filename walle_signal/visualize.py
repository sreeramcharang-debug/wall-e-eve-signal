
"""ASCII visualization for a completed WALL-E mission."""

import json
import sys
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def draw_grid(mission, report):
    grid = mission["map"]
    width = grid["width"]
    height = grid["height"]

    blocked = {tuple(cell) for cell in grid.get("blocked", [])}
    start = tuple(grid["start"])

    hidden = mission.get("hidden", {})
    actual_eve = tuple(hidden["eve_location"])
    estimated = tuple(report["estimated_location"])

    clues = {
        tuple(clue["at"]): "C"
        for clue in hidden.get("clues", [])
    }

    visited = set()
    path = []

    for step in report.get("trace", []):
        pos = tuple(step["position"])
        visited.add(pos)
        if not path or path[-1] != pos:
            path.append(pos)

    print("\n=== WALL-E and EVE's Signal ===\n")
    print("Legend: S=Start, W=WALL-E final, #=Rubble")
    print("        C=Clue, *=Visited path, E=Actual EVE")
    print("        ?=Estimated EVE, B=Actual and estimated")
    print()

    for y in range(height):
        row = []

        for x in range(width):
            cell = (x, y)

            if cell == actual_eve and cell == estimated:
                symbol = "B"
            elif cell == actual_eve:
                symbol = "E"
            elif cell == estimated:
                symbol = "?"
            elif path and cell == path[-1]:
                symbol = "W"
            elif cell in blocked:
                symbol = "#"
            elif cell in clues and cell in visited:
                symbol = "C"
            elif cell in visited:
                symbol = "*"
            elif cell == start:
                symbol = "S"
            else:
                symbol = "."

            row.append(f" {symbol} ")

        print("".join(row))

    
    print("\nPath:", " -> ".join(map(str, path)))
    print("Actual EVE location:", actual_eve)
    print("Estimated EVE location:", estimated)
    print("Estimated status:", report["estimated_status"])
    print("EVE found:", report["found_eve"])
    print("Steps taken:", report["steps"])
    print("Battery remaining:", report["final_battery"])
    print("Time remaining:", report["final_time_remaining"])



def main():
    mission_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "missions/earth_sector_7.json"
    )
    report_path = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "output/report.json"
    )

    if not mission_path.exists() or not report_path.exists():
        print("Mission or report file not found.")
        print("Run the mission first to generate output/report.json.")
        raise SystemExit(1)

    mission = load_json(mission_path)
    report = load_json(report_path)
    draw_grid(mission, report)


if __name__ == "__main__":
    main()
