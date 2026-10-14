import unittest
from pathlib import Path

from walle_signal.mission import run_mission
from walle_signal.report import format_report


ROOT = Path(__file__).resolve().parents[1]
EARTH = ROOT / "missions" / "earth_sector_7.json"
SMALL = ROOT / "missions" / "small_test.json"


class MissionTests(unittest.TestCase):
    def test_sample_mission_completes(self):
        report = run_mission(EARTH)
        self.assertEqual(report["mission_id"], "earth_sector_7")
        self.assertEqual(len(report["estimated_location"]), 2)
        self.assertIn(report["estimated_status"], ["operational", "signal_weak", "damaged", "dormant"])
        self.assertGreaterEqual(report["confidence"], 0.0)
        self.assertLessEqual(report["confidence"], 1.0)
        self.assertTrue(report["recommended_next_action"])
        self.assertGreaterEqual(report["steps"], 3)
        text = format_report(report)
        self.assertIn("Estimated EVE location", text)
        self.assertIn("Action trace:", text)

    def test_trace_records_resource_changes(self):
        report = run_mission(SMALL)
        deltas = [step["time_delta"] for step in report["trace"] if step["ok"]]
        self.assertTrue(deltas)
        self.assertTrue(all(delta >= 1 for delta in deltas))

    def test_small_map_can_recover_eve_cell(self):
        report = run_mission(SMALL)
        # 2x2 is small enough that survey plus clues should concentrate belief.
        self.assertGreater(report["location_confidence"], 0.3)


if __name__ == "__main__":
    unittest.main()
