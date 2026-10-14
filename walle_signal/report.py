"""Final mission report assembled only from the agent's evidence."""

from __future__ import annotations

from typing import Any

from .agent import Agent


def recommend_next_action(agent: Agent) -> str:
    loc = agent.belief.estimated_location()
    conf = agent.belief.joint_confidence()
    if agent.found_eve:
        return "rendezvous_and_assist"
    if conf >= 0.5:
        return f"proceed_to_sector_{loc[0]}_{loc[1]}"
    return "continue_wide_area_survey"


def build_report(agent: Agent) -> dict[str, Any]:
    belief = agent.belief
    loc = belief.estimated_location()
    status = belief.estimated_status()
    return {
        "mission_id": agent.spec["mission_id"],
        "estimated_location": list(loc),
        "estimated_status": status,
        "location_confidence": round(belief.location_confidence(), 4),
        "status_confidence": round(belief.status_confidence(), 4),
        "confidence": round(belief.joint_confidence(), 4),
        "evidence": list(belief.evidence),
        "recommended_next_action": recommend_next_action(agent),
        "found_eve": agent.found_eve,
        "steps": len(agent.trace),
        "final_battery": agent.env.battery,
        "final_time_remaining": agent.env.time_remaining,
        "visited_cells": [list(cell) for cell in sorted(agent.visited)],
        "trace": agent.trace,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Mission: {report['mission_id']}",
        f"Estimated EVE location: {tuple(report['estimated_location'])}",
        f"Estimated EVE status: {report['estimated_status']}",
        f"Location confidence: {report['location_confidence']:.2f}",
        f"Status confidence: {report['status_confidence']:.2f}",
        f"Joint confidence: {report['confidence']:.2f}",
        f"Recommended next action: {report['recommended_next_action']}",
        f"Direct contact: {report['found_eve']}",
        f"Steps taken: {report['steps']}",
        f"Battery remaining: {report['final_battery']}",
        f"Time remaining: {report['final_time_remaining']}",
        "Evidence:",
    ]
    if report["evidence"]:
        lines.extend(f"  - {item}" for item in report["evidence"])
    else:
        lines.append("  - none")
    lines.append("Action trace:")
    for index, step in enumerate(report["trace"], start=1):
        lines.append(
            f"  {index:02d}. {step['action']} @ {tuple(step['position'])} "
            f"-> {step['observation']['kind']} | {step['decision']}"
        )
    return "\n".join(lines)
