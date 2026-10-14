import json
import unittest
from pathlib import Path

from walle_signal.environment import Environment
from walle_signal.models import Action


ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "missions" / "small_test.json"


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment.from_json(SMALL)

    def test_public_spec_omits_hidden_state(self):
        spec = self.env.public_spec()
        blob = json.dumps(spec)
        self.assertNotIn("hidden", spec)
        self.assertNotIn("eve_location", spec)
        self.assertNotIn("clues", spec)
        self.assertNotIn("eve_status", spec)
        self.assertNotIn("scorched_panel", blob)

    def test_move_and_costs(self):
        start_battery = self.env.battery
        start_time = self.env.time_remaining
        result = self.env.step(Action("move", "east"))
        self.assertTrue(result.ok)
        self.assertEqual(result.position, (1, 0))
        self.assertEqual(self.env.battery, start_battery - 1)
        self.assertEqual(self.env.time_remaining, start_time - 1)

    def test_inspect_empty_does_not_reveal_eve(self):
        result = self.env.step(Action("inspect"))
        self.assertEqual(result.observation.kind, "empty")
        self.assertNotIn("status", result.observation.payload)

    def test_inspect_clue_gives_heading_not_coordinates(self):
        self.env.step(Action("move", "east"))
        result = self.env.step(Action("inspect"))
        self.assertEqual(result.observation.kind, "clue")
        self.assertEqual(result.observation.payload["heading"], "south")
        self.assertNotIn("eve_location", result.observation.payload)

    def test_found_eve_only_on_her_cell(self):
        self.env.step(Action("move", "east"))
        self.env.step(Action("move", "south"))
        result = self.env.step(Action("inspect"))
        self.assertEqual(result.observation.kind, "found_eve")
        self.assertEqual(result.observation.payload["status"], "damaged")

    def test_recharge_only_at_station(self):
        self.env.step(Action("move", "east"))
        names = [action.key() for action in self.env.legal_actions()]
        self.assertNotIn("recharge", names)
        self.env.step(Action("move", "west"))
        names = [action.key() for action in self.env.legal_actions()]
        self.assertIn("recharge", names)

    def test_listen_stronger_near_eve(self):
        far = self.env.step(Action("listen"))
        self.env.step(Action("move", "east"))
        self.env.step(Action("move", "south"))
        near = self.env.step(Action("listen"))
        order = ["none", "faint", "moderate", "strong"]
        self.assertGreater(
            order.index(near.observation.payload["band"]),
            order.index(far.observation.payload["band"]),
        )


if __name__ == "__main__":
    unittest.main()
