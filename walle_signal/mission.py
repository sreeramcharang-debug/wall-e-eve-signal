"""Run a mission from a JSON file and return the report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent import Agent
from .environment import Environment
from .report import build_report


def run_mission(path: str | Path) -> dict[str, Any]:
    env = Environment.from_json(path)
    agent = Agent(env)
    agent.run()
    return build_report(agent)
