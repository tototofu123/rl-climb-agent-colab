# Environment Design

## Wall

- Fixed rectangle (e.g. 800 × 600 px)
- Holds placed at fixed (x,y) coordinates
- Multiple wall layouts in `experiments/wall_layouts/`
- Target hold marked (visually distinct)

## Holds

```python
@dataclass
class Hold:
    pos: tuple[float, float]    # (x, y)
    radius: float               # hold size
    hold_type: str              # "jug" | "crimp" | "sloper"
    is_target: bool = False
```

- `jug`: easy to grab (large radius, no penalty)
- `crimp`: harder (smaller, reach penalty)
- `sloper": hardest (slippery, chance of pop-off)

## Body Model

```python
@dataclass
class ClimberState:
    com_pos: tuple[float, float]        # center of mass
    com_vel: tuple[float, float]        # velocity
    limbs: list[LimbState]              # 4 limbs
    
@dataclass
class LimbState:
    anchor: tuple[float, float] | None  # hold position or None
    state: str                          # "attached" | "reaching" | "detached"
    max_reach: float                    # static max reach distance
```

## Observation Space (Gymnasium `spaces.Dict`)

```python
from gymnasium import spaces
import numpy as np

MAX_HOLDS = 10

observation_space = spaces.Dict({
    "body": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),        # pos_x, pos_y, vel_x, vel_y
    "limbs": spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),      # 4 limbs × (anchor_x, anchor_y, state)  state: 0=detached, 1=attached
    "holds": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_HOLDS * 3,)),  # holds × (x, y, is_target)
    "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),      # target_x, target_y, dist_from_com
})
```

Total: 4 + 12 + 30 + 3 = **49 values** (fixed size, pad unused holds with zeros)

## Action Space (Discrete)

```
Actions:
  0:           NOOP
  1..H:        REACH_HOLD_i (with first free limb)
  H+1..H+4:    RELEASE_LIMB_i
  H+5:         SHIFT_LEFT
  H+6:         SHIFT_RIGHT
  H+7:         DYNO (if enabled)
```

Size = 1 + H + 4 + 2 + (1 if dyno else 0) = H + 8 (with dyno) or H + 7 (without)

## Step Logic

```
step(action):
  1. Parse action
  2. If REACH_HOLD_i:
     - Find nearest free limb that's within max_reach
     - Attach limb to hold
     - If not in range: apply reach_penalty
  3. If RELEASE_LIMB_i:
     - Detach limb from hold
  4. If SHIFT:
     - Apply small lateral impulse to CoM
  5. If DYNO:
     - Release all limbs
     - Apply upward impulse to CoM
  6. physics.step()  # advance Box2D simulation
  7. Check constraints:
     - If CoM below wall bottom or no limbs attached for > threshold → fallen → done
     - If CoM within target radius → success → done
     - If max_steps exceeded → truncation
  8. Build observation
  9. Compute reward
  10. Return (obs, reward, terminated, truncated, info)
```

## Episode Configuration

```python
@dataclass
class EnvConfig:
    wall_width: int = 800
    wall_height: int = 600
    max_steps: int = 200
    gravity: float = -9.81
    max_reach: float = 150.0
    enable_dyno: bool = False
    reward_weights: dict = None
    wall_layout: str = "beginner"
```

## Rendering (Pygame)

- Wall background (light gray)
- Holds as colored circles (jug=green, crimp=orange, sloper=red, target=gold)
- Body CoM as filled circle (blue)
- Limbs as lines from CoM to holds (green=attached, gray=detached)
- Info overlay: episode step, distance to target, reward
