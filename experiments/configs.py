"""Reward configurations for Stage 3 — Reward Engineering.

Each config defines weights for reward components.
The env sums: Σ(weight_i * signal_i) each step.
"""

DENSE = {
    "progress": 2.0,
    "hold_bonus": 0.0,
    "time_penalty": 0.0,
    "reach_penalty": 0.0,
    "fall_penalty": -1.0,
    "success_bonus": 5.0,
}

SPARSE = {
    "progress": 0.0,
    "hold_bonus": 0.0,
    "time_penalty": 0.0,
    "reach_penalty": 0.0,
    "fall_penalty": -1.0,
    "success_bonus": 5.0,
}

HYBRID = {
    "progress": 1.0,
    "hold_bonus": 0.2,
    "time_penalty": -0.01,
    "reach_penalty": -0.3,
    "fall_penalty": -1.0,
    "success_bonus": 5.0,
}

SPEED = {
    "progress": 0.5,
    "hold_bonus": 0.1,
    "time_penalty": -0.05,
    "reach_penalty": -0.3,
    "fall_penalty": -2.0,
    "success_bonus": 10.0,
}

SAFE = {
    "progress": 2.0,
    "hold_bonus": 0.3,
    "time_penalty": -0.005,
    "reach_penalty": -0.5,
    "fall_penalty": -3.0,
    "success_bonus": 3.0,
}

DYNO = {
    "progress": 1.0,
    "hold_bonus": 0.3,
    "time_penalty": -0.01,
    "reach_penalty": -0.3,
    "fall_penalty": -1.0,
    "success_bonus": 8.0,
}

ALL = {
    "dense": DENSE,
    "sparse": SPARSE,
    "hybrid": HYBRID,
    "speed": SPEED,
    "safe": SAFE,
    "dyno": DYNO,
}


def get_config(name: str) -> dict:
    if name not in ALL:
        raise ValueError(f"Unknown config '{name}'. Available: {list(ALL.keys())}")
    return dict(ALL[name])
