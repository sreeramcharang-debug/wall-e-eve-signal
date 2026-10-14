"""Greedy, explainable WALL-E controller. Never reads hidden mission fields."""

from __future__ import annotations

from .belief import Belief
from .environment import Environment
from .models import Action, DIRECTIONS, StepResult, manhattan


class Agent:
    def __init__(self, env: Environment):
        spec = env.public_spec()
        self.env = env
        self.spec = spec
        self.cells = env.walkable_cells()
        self.belief = Belief(
            cells=self.cells,
            statuses=list(spec["statuses"]),
            width=spec["map"]["width"],
            height=spec["map"]["height"],
            scan_range=spec["noise"]["scan_range"],
        )
        self.visited: set[tuple[int, int]] = {tuple(spec["map"]["start"])}
        self.inspected: set[tuple[int, int]] = set()
        self.listened: set[tuple[int, int]] = set()
        self.scanned: set[tuple[int, int]] = set()
        self.found_eve = False
        self.trace: list[dict] = []
        self.confidence_stop = 0.72

    def run(self) -> list[dict]:
        while not self.env.mission_over:
            action, explanation = self.select_action()
            result = self.env.step(action)
            belief_note = self.belief.update(result.observation, result.position)
            self._remember(result)
            self.trace.append(self._trace_entry(result, explanation, belief_note))
            if result.mission_over:
                break
        return self.trace

    def select_action(self) -> tuple[Action, str]:
        legal = self.env.legal_actions()
        if not legal:
            return Action("report"), "No affordable actions remain; filing report"

        if self.found_eve:
            report = next((action for action in legal if action.name == "report"), None)
            if report:
                return report, "EVE found; filing report with current evidence"

        if self._should_report(legal):
            report = next(action for action in legal if action.name == "report")
            return report, self._report_reason()

        scored: list[tuple[float, Action, str]] = []
        for action in legal:
            score, why = self._score(action)
            scored.append((score, action, why))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        return best[1], best[2]

    def _should_report(self, legal: list[Action]) -> bool:
        if not any(action.name == "report" for action in legal):
            return False
        low_time = self.env.time_remaining <= 2
        low_battery = self.env.battery <= 3 and not self._can_reach_recharge()
        confident = self.belief.joint_confidence() >= self.confidence_stop
        explored = len(self.inspected) >= max(3, len(self.cells) // 2)
        return low_time or low_battery or (confident and explored)

    def _report_reason(self) -> str:
        loc = self.belief.estimated_location()
        status = self.belief.estimated_status()
        conf = self.belief.joint_confidence()
        return (
            f"Ending survey: estimate EVE at {loc} status={status} "
            f"joint_confidence={conf:.2f}"
        )

    def _score(self, action: Action) -> tuple[float, str]:
        pos = self.env.position
        target = self.belief.estimated_location()

        if action.name == "inspect" and pos not in self.inspected:
            return 12.0 + 4.0 * self.belief.location.get(pos, 0), f"Inspect unseen cell {pos}"
        if action.name == "inspect":
            return -2.0, f"Already inspected {pos}"

        if action.name == "listen" and pos not in self.listened:
            return 6.5, f"Listen for plant-signal at {pos}"
        if action.name == "listen":
            return 0.5, "Already listened here"

        if action.name == "scan" and pos not in self.scanned:
            return 7.0, f"Scan local signature at {pos}"
        if action.name == "scan":
            return 0.4, "Already scanned here"

        if action.name == "recharge":
            need = 1.0 - (self.env.battery / max(1, self.env.max_battery))
            return 9.0 * need, f"Recharge (battery {self.env.battery})"

        if action.name == "move" and action.direction:
            nxt = self._peek(pos, action.direction)
            if nxt is None:
                return -10.0, "Blocked"
            toward = manhattan(pos, target) - manhattan(nxt, target)
            unvisited = 3.0 if nxt not in self.visited else 0.0
            uninspected = 2.5 if nxt not in self.inspected else 0.0
            belief_pull = 6.0 * self.belief.location.get(nxt, 0.0)
            # Keep a path home if battery is tight.
            station = self._nearest_station()
            home_bias = 0.0
            if station and self.env.battery <= 8:
                home_bias = 4.0 * (manhattan(pos, station) - manhattan(nxt, station))
            score = 3.0 * toward + unvisited + uninspected + belief_pull + home_bias
            return score, (
                f"Move {action.direction} to {nxt} (toward estimate {toward}, "
                f"belief {self.belief.location.get(nxt, 0):.2f})"
            )

        if action.name == "report":
            return -1.0, "Report is a last resort while survey remaining"

        return 0.0, action.key()

    def _peek(self, cell: tuple[int, int], direction: str) -> tuple[int, int] | None:
        from .models import DELTAS

        dx, dy = DELTAS[direction]
        nxt = (cell[0] + dx, cell[1] + dy)
        blocked = {(int(x), int(y)) for x, y in self.spec["map"]["blocked"]}
        width = self.spec["map"]["width"]
        height = self.spec["map"]["height"]
        x, y = nxt
        if 0 <= x < width and 0 <= y < height and nxt not in blocked:
            return nxt
        return None

    def _nearest_station(self) -> tuple[int, int] | None:
        stations = [tuple(cell) for cell in self.spec["recharge_stations"]]
        if not stations:
            return None
        return min(stations, key=lambda cell: manhattan(self.env.position, cell))

    def _can_reach_recharge(self) -> bool:
        station = self._nearest_station()
        if station is None:
            return False
        move_cost = int(self.spec["action_costs"]["move"]["battery"])
        return manhattan(self.env.position, station) * move_cost < self.env.battery

    def _remember(self, result: StepResult) -> None:
        self.visited.add(result.position)
        if result.observation.kind in {"empty", "clue", "found_eve"}:
            self.inspected.add(result.position)
        if result.observation.kind == "signal":
            self.listened.add(result.position)
        if result.observation.kind == "scan":
            self.scanned.add(result.position)
        if result.observation.kind == "found_eve":
            self.found_eve = True

    def _trace_entry(self, result: StepResult, decision: str, belief_note: str) -> dict:
        top = self.belief.top_locations(3)
        return {
            "action": result.action.key(),
            "ok": result.ok,
            "position": list(result.position),
            "battery": result.battery,
            "time_remaining": result.time_remaining,
            "battery_delta": result.battery_delta,
            "time_delta": result.time_delta,
            "observation": {
                "kind": result.observation.kind,
                "payload": result.observation.payload,
            },
            "message": result.message,
            "decision": decision,
            "belief_update": belief_note,
            "estimate": {
                "location": list(self.belief.estimated_location()),
                "status": self.belief.estimated_status(),
                "location_confidence": round(self.belief.location_confidence(), 4),
                "status_confidence": round(self.belief.status_confidence(), 4),
                "joint_confidence": round(self.belief.joint_confidence(), 4),
                "top_locations": [
                    {"cell": list(cell), "p": round(prob, 4)} for cell, prob in top
                ],
            },
        }

    # Kept so tests can show the agent never consults DIRECTIONS from hidden state.
    _cardinals = DIRECTIONS
