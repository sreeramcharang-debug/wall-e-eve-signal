import ast
import unittest
from pathlib import Path

from walle_signal.agent import Agent
from walle_signal.environment import Environment
from walle_signal.mission import run_mission


ROOT = Path(__file__).resolve().parents[1]
AGENT_FILE = ROOT / "walle_signal" / "agent.py"
SMALL = ROOT / "missions" / "small_test.json"


class InformationBarrierTests(unittest.TestCase):
    def test_agent_source_does_not_read_hidden_fields(self):
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        forbidden = {"_hidden", "_eve_location", "_eve_status", "_clues", "hidden"}
        found = set()

        class Visitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                if node.attr in forbidden:
                    found.add(node.attr)
                self.generic_visit(node)

            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in forbidden:
                    found.add(node.value)

        Visitor().visit(tree)
        self.assertEqual(found, set())

    def test_public_spec_cannot_recover_eve_cell_directly(self):
        env = Environment.from_json(SMALL)
        spec = env.public_spec()
        self.assertEqual(spec["map"]["start"], [0, 0])
        self.assertNotEqual(spec["map"]["start"], [1, 1])
        self.assertIsNone(spec.get("hidden"))

    def test_agent_uses_only_observations_to_form_estimate(self):
        env = Environment.from_json(SMALL)
        agent = Agent(env)
        self.assertNotIn("_eve_location", agent.__dict__)
        self.assertNotIn("_hidden", agent.spec)
        agent.run()
        self.assertTrue(agent.trace)
        kinds = {step["observation"]["kind"] for step in agent.trace}
        self.assertTrue(kinds.intersection({"signal", "scan", "empty", "clue", "found_eve", "report_filed"}))

    def test_full_mission_report_has_required_fields(self):
        report = run_mission(SMALL)
        for field in (
            "estimated_location",
            "estimated_status",
            "confidence",
            "evidence",
            "recommended_next_action",
            "trace",
        ):
            self.assertIn(field, report)
        self.assertGreaterEqual(len(report["trace"]), 1)
        step = report["trace"][0]
        self.assertIn("decision", step)
        self.assertIn("observation", step)
        self.assertIn("action", step)


if __name__ == "__main__":
    unittest.main()
