import unittest

from walle_signal.belief import Belief
from walle_signal.models import Observation


class BeliefTests(unittest.TestCase):
    def setUp(self):
        self.cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.belief = Belief(
            cells=self.cells,
            statuses=["operational", "damaged"],
            width=2,
            height=2,
            scan_range=1,
        )

    def test_empty_inspect_downweights_current_cell(self):
        before = self.belief.location[(0, 0)]
        self.belief.update(Observation("empty", {"position": [0, 0]}), (0, 0))
        self.assertLess(self.belief.location[(0, 0)], before)
        self.assertGreater(self.belief.location[(1, 1)], self.belief.location[(0, 0)])

    def test_clue_updates_status_and_heading(self):
        self.belief.update(
            Observation(
                "clue",
                {
                    "kind": "panel",
                    "heading": "south",
                    "status_hint": "damaged",
                    "reliability": 0.9,
                },
            ),
            (1, 0),
        )
        self.assertEqual(self.belief.estimated_status(), "damaged")
        south_mass = self.belief.location[(0, 1)] + self.belief.location[(1, 1)]
        north_mass = self.belief.location[(0, 0)] + self.belief.location[(1, 0)]
        self.assertGreater(south_mass, north_mass)

    def test_found_eve_is_certain(self):
        self.belief.update(
            Observation("found_eve", {"status": "damaged", "position": [1, 1]}),
            (1, 1),
        )
        self.assertEqual(self.belief.estimated_location(), (1, 1))
        self.assertEqual(self.belief.estimated_status(), "damaged")
        self.assertAlmostEqual(self.belief.joint_confidence(), 1.0)

    def test_listen_prefers_matching_distance_band(self):
        self.belief.update(Observation("signal", {"band": "strong"}), (1, 1))
        self.assertEqual(self.belief.estimated_location(), (1, 1))


if __name__ == "__main__":
    unittest.main()
