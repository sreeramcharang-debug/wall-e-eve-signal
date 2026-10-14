"""Simulated Earth sectors. Hidden EVE state is environment-private."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .models import (
    DELTAS,
    DIRECTIONS,
    Action,
    Observation,
    StepResult,
    chebyshev,
    compass_from_to,
    manhattan,
    signal_band_for_distance,
)


class Environment:
    """Loads a mission JSON file and answers only through observations."""

    def __init__(self, config: dict[str, Any]):
        self._config = deepcopy(config)
        self._hidden = self._config.pop("hidden", None)
        if not self._hidden:
            raise ValueError("Mission config must include a hidden block")

        self.mission_id = self._config.get("mission_id", "unnamed")
        self.width = int(self._config["map"]["width"])
        self.height = int(self._config["map"]["height"])
        start = self._config["map"]["start"]
        self.start = (int(start[0]), int(start[1]))
        self.blocked = { (int(x), int(y)) for x, y in self._config["map"].get("blocked", []) }
        self.statuses = list(self._config.get("statuses", ["operational", "signal_weak", "damaged", "dormant"]))
        self.action_costs = self._config["action_costs"]
        self.max_battery = int(self._config["resources"]["battery"])
        self.max_time = int(self._config["resources"]["time_limit"])
        self.recharge_stations = {
            (int(x), int(y)) for x, y in self._config.get("recharge_stations", [])
        }
        self.scan_range = int(self._config.get("noise", {}).get("scan_range", 1))

        eve = self._hidden["eve_location"]
        self._eve_location = (int(eve[0]), int(eve[1]))
        self._eve_status = str(self._hidden["eve_status"])
        self._clues = []
        for clue in self._hidden.get("clues", []):
            at = clue["at"]
            self._clues.append(
                {
                    "at": (int(at[0]), int(at[1])),
                    "kind": clue.get("kind", "scrap"),
                    "status_hint": clue.get("status_hint"),
                    "reliability": float(clue.get("reliability", 0.8)),
                    "note": clue.get("note", ""),
                }
            )

        self._validate_hidden()
        self.reset()

    @classmethod
    def from_json(cls, path: str | Path) -> "Environment":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def _validate_hidden(self) -> None:
        if not self.walkable(self._eve_location):
            raise ValueError("EVE location must be a walkable cell")
        if self._eve_status not in self.statuses:
            raise ValueError(f"Unknown EVE status: {self._eve_status}")
        for clue in self._clues:
            if not self.walkable(clue["at"]):
                raise ValueError(f"Clue at {clue['at']} is not walkable")

    def reset(self) -> None:
        self.position = self.start
        self.battery = self.max_battery
        self.time_remaining = self.max_time
        self.mission_over = False
        self.found_eve = False

    def walkable(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.blocked

    def walkable_cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.walkable((x, y))
        ]

    def public_spec(self) -> dict[str, Any]:
        """Map, costs, and resources only. No EVE location, status, or clues."""
        return {
            "mission_id": self.mission_id,
            "map": {
                "width": self.width,
                "height": self.height,
                "start": list(self.start),
                "blocked": [list(cell) for cell in sorted(self.blocked)],
            },
            "resources": {"battery": self.max_battery, "time_limit": self.max_time},
            "action_costs": deepcopy(self.action_costs),
            "recharge_stations": [list(cell) for cell in sorted(self.recharge_stations)],
            "statuses": list(self.statuses),
            "noise": {"scan_range": self.scan_range},
        }

    def legal_actions(self) -> list[Action]:
        actions = [Action("scan"), Action("listen"), Action("inspect"), Action("report")]
        for direction in DIRECTIONS:
            nxt = self._neighbor(self.position, direction)
            if nxt is not None:
                actions.append(Action("move", direction))
        if self.position in self.recharge_stations:
            actions.append(Action("recharge"))
        return [action for action in actions if self._can_afford(action)]

    def step(self, action: Action) -> StepResult:
        if self.mission_over:
            return StepResult(
                ok=False,
                action=action,
                observation=Observation("mission_over", {}),
                battery=self.battery,
                time_remaining=self.time_remaining,
                position=self.position,
                message="Mission already ended",
                mission_over=True,
            )

        if action not in self.legal_actions():
            return StepResult(
                ok=False,
                action=action,
                observation=Observation("illegal", {"reason": "not_legal"}),
                battery=self.battery,
                time_remaining=self.time_remaining,
                position=self.position,
                message=f"Illegal action {action.key()}",
            )

        cost = self._cost(action)
        if cost["spend"] > self.battery or cost["time"] > self.time_remaining:
            return StepResult(
                ok=False,
                action=action,
                observation=Observation("illegal", {"reason": "insufficient_resources"}),
                battery=self.battery,
                time_remaining=self.time_remaining,
                position=self.position,
                message="Not enough battery or time",
            )

        observation, message, ended = self._apply(action)
        self.battery = min(self.max_battery, self.battery - cost["spend"] + cost["restore"])
        self.time_remaining -= cost["time"]
        if self.time_remaining <= 0 or self.battery <= 0:
            ended = True
            if observation.kind not in {"report_filed", "found_eve"}:
                observation = Observation(
                    "resources_exhausted",
                    {"battery": self.battery, "time_remaining": self.time_remaining},
                )
                message = "Battery or time exhausted"
        self.mission_over = ended
        return StepResult(
            ok=True,
            action=action,
            observation=observation,
            battery=self.battery,
            time_remaining=self.time_remaining,
            position=self.position,
            message=message,
            mission_over=ended,
            battery_delta=cost["restore"] - cost["spend"],
            time_delta=cost["time"],
        )

    def _cost(self, action: Action) -> dict[str, int]:
        raw = self.action_costs[action.name]
        spend = int(raw.get("battery", 0))
        restore = int(raw.get("restore", 0))
        time_cost = int(raw.get("time", 1))
        return {"spend": spend, "restore": restore, "time": time_cost}

    def _can_afford(self, action: Action) -> bool:
        cost = self._cost(action)
        if cost["spend"] > self.battery:
            return False
        if cost["time"] > self.time_remaining:
            return False
        return True

    def _neighbor(self, cell: tuple[int, int], direction: str) -> tuple[int, int] | None:
        dx, dy = DELTAS[direction]
        nxt = (cell[0] + dx, cell[1] + dy)
        return nxt if self.walkable(nxt) else None

    def _apply(self, action: Action) -> tuple[Observation, str, bool]:
        if action.name == "move" and action.direction:
            nxt = self._neighbor(self.position, action.direction)
            assert nxt is not None
            self.position = nxt
            return (
                Observation(
                    "moved",
                    {
                        "position": list(self.position),
                        "direction": action.direction,
                        "at_recharge_station": self.position in self.recharge_stations,
                    },
                ),
                f"Moved {action.direction} to {self.position}",
                False,
            )

        if action.name == "listen":
            distance = manhattan(self.position, self._eve_location)
            max_distance = self.width + self.height - 2
            band = signal_band_for_distance(distance, max_distance)
            return (
                Observation(
                    "signal",
                    {"band": band, "position": list(self.position)},
                ),
                f"Listened: {band} plant-signal band",
                False,
            )

        if action.name == "scan":
            distance = chebyshev(self.position, self._eve_location)
            payload: dict[str, Any] = {"position": list(self.position), "range": self.scan_range}
            if distance == 0:
                payload["lock"] = "here"
                message = "Scan lock: signature is in this cell"
            elif distance <= self.scan_range:
                payload["lock"] = compass_from_to(self.position, self._eve_location)
                message = f"Scan lock toward {payload['lock']}"
            else:
                payload["lock"] = "none"
                message = "Scan found no lock in range"
            nearby_anomaly = any(
                chebyshev(self.position, clue["at"]) <= 1 for clue in self._clues
            )
            payload["anomaly"] = nearby_anomaly
            if nearby_anomaly:
                message += "; local anomaly detected"
            return Observation("scan", payload), message, False

        if action.name == "inspect":
            if self.position == self._eve_location:
                self.found_eve = True
                return (
                    Observation(
                        "found_eve",
                        {
                            "position": list(self.position),
                            "status": self._eve_status,
                        },
                    ),
                    f"Inspected: EVE located, status={self._eve_status}",
                    False,
                )
            clue = next((item for item in self._clues if item["at"] == self.position), None)
            if clue:
                heading = compass_from_to(self.position, self._eve_location)
                return (
                    Observation(
                        "clue",
                        {
                            "kind": clue["kind"],
                            "heading": heading,
                            "status_hint": clue["status_hint"],
                            "reliability": clue["reliability"],
                            "note": clue["note"],
                            "position": list(self.position),
                        },
                    ),
                    f"Inspected clue ({clue['kind']}) heading {heading}",
                    False,
                )
            return (
                Observation("empty", {"position": list(self.position)}),
                "Inspected: nothing useful",
                False,
            )

        if action.name == "recharge":
            return (
                Observation("recharged", {"position": list(self.position)}),
                "Recharged at solar station",
                False,
            )

        if action.name == "report":
            return (
                Observation("report_filed", {"position": list(self.position)}),
                "Filed mission report",
                True,
            )

        raise ValueError(f"Unhandled action {action}")
