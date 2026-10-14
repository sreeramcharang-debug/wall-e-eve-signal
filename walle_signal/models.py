"""Shared data types. Hidden mission fields never appear on the agent side."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Direction = Literal["north", "south", "east", "west"]
ActionName = Literal["move", "scan", "listen", "inspect", "recharge", "report"]
SignalBand = Literal["none", "faint", "moderate", "strong"]

DIRECTIONS: tuple[Direction, ...] = ("north", "south", "east", "west")
DELTAS: dict[Direction, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


def opposite(direction: Direction) -> Direction:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }[direction]


def compass_from_to(origin: tuple[int, int], target: tuple[int, int]) -> Direction | None:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def signal_band_for_distance(distance: int, max_distance: int) -> SignalBand:
    if max_distance <= 0:
        return "strong"
    closeness = 1.0 - (distance / max_distance)
    if closeness >= 0.75:
        return "strong"
    if closeness >= 0.45:
        return "moderate"
    if closeness >= 0.15:
        return "faint"
    return "none"


@dataclass(frozen=True)
class Action:
    name: ActionName
    direction: Direction | None = None

    def key(self) -> str:
        if self.direction:
            return f"{self.name}:{self.direction}"
        return self.name


@dataclass
class Observation:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    ok: bool
    action: Action
    observation: Observation
    battery: int
    time_remaining: int
    position: tuple[int, int]
    message: str
    mission_over: bool = False
    battery_delta: int = 0
    time_delta: int = 0
