# WALL-E and EVE's Signal

WALL-E surveys a wrecked Earth sector for EVE after a faint plant-signal. The simulator loads a JSON mission. Hidden fields (EVE's cell, status, and clue placements) stay inside the environment. WALL-E only sees the public map, action costs, and observations.

The controller is deliberately simple: greedy action scoring plus Bayesian belief updates over location and status.

## Requirements

- Python 3.10+
- Standard library only (no pip packages, no paid APIs)

## Setup

```powershell
cd C:\Users\chait\cd walle-eves-signal
```

No virtualenv or install step is required.

## Run a sample mission

```powershell
python -m walle_signal missions/earth_sector_7.json
```

Write the full JSON report (trace, evidence, estimate) to a file:

```powershell
python -m walle_signal missions/earth_sector_7.json --json output/report.json
```

A smaller map used by tests:

```powershell
python -m walle_signal missions/small_test.json
```

## Automated tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Mission JSON

Public fields describe the grid, start cell, blocked rubble, battery, time limit, action costs, recharge stations, and possible EVE statuses.

The `hidden` block is **environment-only**:

- `eve_location` — ground-truth cell
- `eve_status` — ground-truth status
- `clues` — inspectable debris that yields a compass heading and a status hint, never raw coordinates

`battery` on an action is the amount spent. `recharge` may also include `restore` to recover energy at a solar station.

## Agent loop

1. Choose a legal action (inspect new cells, listen, scan, move toward high-belief cells, recharge, or report).
2. The environment returns an observation and deducts battery/time.
3. Beliefs update:
   - **listen** — signal band vs. Manhattan distance
   - **scan** — local lock / no-lock within Chebyshev range
   - **inspect empty** — almost certainly not EVE
   - **clue** — boost cells in the hinted direction and the hinted status
   - **found_eve** — certainty
4. Every step is appended to a trace with the decision explanation.
5. The mission ends on `report`, or when battery/time run out. The final report lists evidence, estimated location/status, confidence, and a recommended next action.

## Information barrier

`Environment.public_spec()` strips `hidden`. The agent module does not read `_eve_location`, `_clues`, or `hidden`. Tests parse `agent.py` and fail if those names appear.
