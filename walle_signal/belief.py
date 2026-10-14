"""Explainable Bayesian belief over EVE location and status."""

from __future__ import annotations

from .models import (
    Observation,
    chebyshev,
    compass_from_to,
    manhattan,
    signal_band_for_distance,
)


class Belief:
    def __init__(
        self,
        cells: list[tuple[int, int]],
        statuses: list[str],
        width: int,
        height: int,
        scan_range: int,
    ):
        if not cells:
            raise ValueError("Need at least one walkable cell")
        self.cells = cells
        self.statuses = statuses
        self.width = width
        self.height = height
        self.scan_range = scan_range
        uniform_loc = 1.0 / len(cells)
        uniform_st = 1.0 / len(statuses)
        self.location = {cell: uniform_loc for cell in cells}
        self.status = {name: uniform_st for name in statuses}
        self.evidence: list[str] = []

    def update(self, observation: Observation, position: tuple[int, int]) -> str:
        if observation.kind == "signal":
            explanation = self._update_from_signal(observation.payload["band"], position)
        elif observation.kind == "scan":
            explanation = self._update_from_scan(observation.payload, position)
        elif observation.kind == "empty":
            explanation = self._update_from_empty(position)
        elif observation.kind == "clue":
            explanation = self._update_from_clue(observation.payload, position)
        elif observation.kind == "found_eve":
            explanation = self._update_from_found(observation.payload, position)
        else:
            explanation = f"No belief change for {observation.kind}"
        self._normalize()
        return explanation

    def estimated_location(self) -> tuple[int, int]:
        return max(self.location, key=self.location.get)

    def estimated_status(self) -> str:
        return max(self.status, key=self.status.get)

    def location_confidence(self) -> float:
        return self.location[self.estimated_location()]

    def status_confidence(self) -> float:
        return self.status[self.estimated_status()]

    def joint_confidence(self) -> float:
        return self.location_confidence() * self.status_confidence()

    def top_locations(self, k: int = 3) -> list[tuple[tuple[int, int], float]]:
        ranked = sorted(self.location.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def entropy_proxy(self) -> float:
        """Sum of squared probabilities: 1 means certain, 1/n means uniform."""
        return sum(p * p for p in self.location.values())

    def _normalize(self) -> None:
        loc_total = sum(self.location.values())
        st_total = sum(self.status.values())
        if loc_total <= 0:
            even = 1.0 / len(self.cells)
            self.location = {cell: even for cell in self.cells}
        else:
            self.location = {cell: value / loc_total for cell, value in self.location.items()}
        if st_total <= 0:
            even = 1.0 / len(self.statuses)
            self.status = {name: even for name in self.statuses}
        else:
            self.status = {name: value / st_total for name, value in self.status.items()}

    def _update_from_signal(self, band: str, position: tuple[int, int]) -> str:
        max_distance = self.width + self.height - 2
        for cell in self.cells:
            expected = signal_band_for_distance(manhattan(position, cell), max_distance)
            self.location[cell] *= 0.85 if expected == band else 0.15
        self.evidence.append(f"listen:{band}@{position}")
        return f"Updated location beliefs from listen band '{band}'"

    def _update_from_scan(self, payload: dict, position: tuple[int, int]) -> str:
        lock = payload.get("lock")
        if lock == "here":
            for cell in self.cells:
                self.location[cell] *= 20.0 if cell == position else 0.05
            self.evidence.append(f"scan:here@{position}")
            return "Scan lock on this cell; concentrated location belief here"
        if lock in {"north", "south", "east", "west"}:
            for cell in self.cells:
                in_range = chebyshev(position, cell) <= self.scan_range and cell != position
                heading = compass_from_to(position, cell)
                if in_range and heading == lock:
                    self.location[cell] *= 8.0
                elif in_range:
                    self.location[cell] *= 0.2
                else:
                    self.location[cell] *= 0.9
            self.evidence.append(f"scan:{lock}@{position}")
            return f"Boosted cells toward {lock} within scan range"
        for cell in self.cells:
            if chebyshev(position, cell) <= self.scan_range:
                self.location[cell] *= 0.25
        self.evidence.append(f"scan:none@{position}")
        return "No scan lock; reduced belief on cells inside scan range"

    def _update_from_empty(self, position: tuple[int, int]) -> str:
        if position in self.location:
            self.location[position] *= 0.02
        self.evidence.append(f"empty@{position}")
        return "Empty inspect: this cell is very unlikely to hold EVE"

    def _update_from_clue(self, payload: dict, position: tuple[int, int]) -> str:
        heading = payload.get("heading")
        reliability = float(payload.get("reliability", 0.8))
        if position in self.location:
            self.location[position] *= 0.02
        if heading:
            for cell in self.cells:
                if compass_from_to(position, cell) == heading:
                    self.location[cell] *= 1.0 + 4.0 * reliability
                else:
                    self.location[cell] *= 1.0 - 0.4 * reliability
        hint = payload.get("status_hint")
        if hint in self.status:
            self.status[hint] *= 1.0 + 5.0 * reliability
            for name in self.statuses:
                if name != hint:
                    self.status[name] *= 1.0 - 0.3 * reliability
        self.evidence.append(f"clue:{payload.get('kind')}->{heading}/{hint}")
        return (
            f"Clue '{payload.get('kind')}' points {heading} "
            f"with status hint '{hint}' (reliability {reliability:.2f})"
        )

    def _update_from_found(self, payload: dict, position: tuple[int, int]) -> str:
        for cell in self.cells:
            self.location[cell] = 1.0 if cell == position else 0.0
        found_status = payload.get("status")
        for name in self.statuses:
            self.status[name] = 1.0 if name == found_status else 0.0
        self.evidence.append(f"found:{found_status}@{position}")
        return f"Direct contact: EVE is at {position} with status {found_status}"
