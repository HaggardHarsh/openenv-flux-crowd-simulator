---
title: Flux
emoji: 🏟️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI-Powered Crowd Management — OpenEnv Environment

A simulation-based **OpenEnv** environment for training AI agents to learn and optimize crowd management strategies in high-density public spaces. The environment models a 6-zone stadium where an agent must prevent stampede situations by managing crowd flow, controlling access points, and issuing alerts.

## 🏗 Architecture

```
crowd_management_env/
├── crowd_env/                  # Core Python package
│   ├── models.py               # Typed dataclass definitions
│   ├── simulation.py           # Crowd physics engine
│   ├── tasks.py                # Task definitions (easy/medium/hard)
│   ├── grader.py               # Deterministic grading
│   └── environment.py          # OpenEnv-compliant environment
├── visualization/              # Web dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── demo.py                     # Random + smart agent demo
├── fastapi_server.py           # FastAPI wrapper (OpenEnv entrypoint)
├── baseline.py                 # OpenAI inference script baseline
├── openenv.yaml                # OpenEnv configuration metadata
├── Dockerfile                  # Container definition for HF Spaces
├── run_viz.py                  # Visualization server
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the OpenEnv random + smart agents demo
python demo.py

# Run the LLM baseline test (Requires OPENAI_API_KEY)
# python baseline.py

# Launch the web visualization
python run_viz.py
# Open http://localhost:8080 in your browser
```

## 📦 Deployment (Hugging Face Spaces)

This environment is fully configured for OpenEnv Hugging Face Space deployments.
```bash
docker build -t crowd-env .
docker run -p 7860:7860 crowd-env
```
The FastAPI instance will serve standard `/reset`, `/step` endpoints on `http://localhost:7860`.

## 📡 OpenEnv API

```python
from crowd_env import CrowdManagementEnv, Action

env = CrowdManagementEnv()

# Initialize episode
obs = env.reset(seed=42, options={"task": "medium"})

# Agent loop
while True:
    action = Action.redirect("A", "B")  # or any action
    result = env.step(action)

    obs = result.observation       # What the agent sees
    reward = result.reward         # Reward signal
    done = result.terminated       # Stampede occurred
    truncated = result.truncated   # Max steps reached

    if done or truncated:
        break

# Get full state and grade
state = env.state()
grade = env.grade()
print(f"Score: {grade.score:.1f} ({grade.letter_grade})")
```

## 🎯 Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `redirect` | Divert crowd flow | source_zone, target_zone |
| `gate_control` | Open/close access points | source_zone, gate_index, gate_open |
| `alert` | Issue/lift crowd alert | source_zone |
| `no_op` | Take no action | — |

```python
# Factory methods for easy action creation
Action.redirect("A", "B")
Action.close_gate("A", gate_idx=0)
Action.open_gate("E", gate_idx=1)
Action.issue_alert("D")
Action.noop()
```

## 🏟 Venue Layout

6 interconnected zones modeling a stadium:

```
       ┌─────────────┐
       │  A: Main     │ ← Entry
       │  Entrance    │
       └──┬───────┬───┘
          │       │
    ┌─────▼──┐ ┌──▼──────┐
    │B: North│ │C: South  │
    │ Stand  │ │ Stand    │
    └──┬──┬──┘ └──┬──┬───┘
       │  │       │  │
    ┌──▼──▼───────▼──▼──┐
    │   D: Central      │
    │   Arena           │
    └──┬────────────┬───┘
       │            │
  ┌────▼───┐  ┌────▼───┐
  │E: East │  │F: West │ → Exit
  │Concours│  │ Exit   │
  └────────┘  └────────┘
```

## 📊 Risk Levels

Based on real crowd safety research (people per m²):

| Level | Density | Behavior |
|-------|---------|----------|
| 🟢 Safe | < 2.0 ppm² | Free movement |
| 🟡 Elevated | 2.0–3.5 ppm² | Congested, intervention needed |
| 🔴 Critical | 3.5–5.0 ppm² | Dangerous, urgent action required |
| 💀 Stampede | ≥ 5.0 ppm² | Terminal — episode ends |

## 📝 Tasks

| Task | Name | Steps | Arrival Rate | Surges | Exit Capacity |
|------|------|-------|-------------|--------|---------------|
| Easy | Matchday Warm-Up | 100 | 15/step | 0 | 100% |
| Medium | Derby Day Rush | 200 | 30/step | 2 | 75% |
| Hard | Championship Final | 300 | 50/step | 5 | 50% |

## 📈 Grading

Deterministic scoring on 0.0–1.0 scale:

| Component | Weight | Measures |
|-----------|--------|----------|
| Safety | 40% | % of steps with all zones safe |
| Efficiency | 15% | % of actions that addressed actual threats |
| Survival | 30% | 100 if no stampede, 0 otherwise |
| Proactivity | 15% | Preemptive actions on elevated zones |

**Letter grades:** A (≥0.9), B (≥0.75), C (≥0.6), D (≥0.4), F (<0.4)

## 🖥 Visualization

Launch `python run_viz.py` and open the browser to see:
- Interactive zone map with density heatmap
- Real-time population bars and risk indicators
- Flow animations between zones
- Agent action controls
- Timeline chart
- Live scoring dashboard

## License

MIT
