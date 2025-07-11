import numpy as np
from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class Task:
    """Immutable task definition."""
    id: int
    description: str
    mode: str  # '1input' or '2input'
    inputs: np.ndarray  # shape (N, 1) or (N, 2)
    targets: np.ndarray  # shape (N,)
    valid_inputs: Optional[np.ndarray] = None  # validation inputs (out-of-distribution)
    valid_targets: Optional[np.ndarray] = None  # validation targets


# Helper to create cartesian product inputs for two-input tasks
_cart = lambda a, b: np.array([[i, j] for i in a for j in b], dtype=np.int32)

# ========================= 1-INPUT TASKS =========================
_x = np.arange(1, 31, dtype=np.int32).reshape(-1, 1)  # inputs [1..30]
_x_valid = np.arange(31, 51, dtype=np.int32).reshape(-1, 1)  # validation inputs [31..50]

# ========================= 2-INPUT TASKS =========================
_xy = _cart(np.arange(1, 13, dtype=np.int32), np.arange(1, 13, dtype=np.int32))
_xy_valid = _cart(np.arange(15, 21, dtype=np.int32), np.arange(15, 21, dtype=np.int32))  # [15..20] x [15..20]

# ========================= ALL TASKS =========================
tasks = [
    # 1-input tasks
    Task(
        id=0,
        description="undefined",
        mode="1input",
        inputs=_x,
        targets=np.zeros(len(_x), dtype=np.int32),
        valid_inputs=None,  # No validation for undefined task
        valid_targets=None
    ),
    Task(
        id=1,
        description="multiply by 2",
        mode="1input", 
        inputs=_x,
        targets=(_x.flatten() * 2).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() * 2).astype(np.int32)
    ),
    Task(
        id=2,
        description="multiply by 3",
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() * 3).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() * 3).astype(np.int32)
    ),
    Task(
        id=3,
        description="multiply by 4", 
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() * 4).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() * 4).astype(np.int32)
    ),
    Task(
        id=4,
        description="divide by 2",
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() // 2).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() // 2).astype(np.int32)
    ),
    Task(
        id=5,
        description="divide by 3",
        mode="1input", 
        inputs=_x,
        targets=(_x.flatten() // 3).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() // 3).astype(np.int32)
    ),
    Task(
        id=6,
        description="add 1",
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() + 1).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() + 1).astype(np.int32)
    ),
    Task(
        id=7,
        description="subtract 1",
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() - 1).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() - 1).astype(np.int32)
    ),
    Task(
        id=8,
        description="power of 2",
        mode="1input",
        inputs=_x,
        targets=(2 ** _x.flatten()).astype(np.int32),
        valid_inputs=np.arange(31, 41, dtype=np.int32).reshape(-1, 1),  # Smaller range to avoid overflow
        valid_targets=(2 ** np.arange(31, 41, dtype=np.int32)).astype(np.int64)  # Use int64 for large values
    ),
    Task(
        id=9,
        description="round((power of 2)/4)",
        mode="1input",
        inputs=_x,
        targets=np.round((2 ** _x.flatten()) / 4).astype(np.int32),
        valid_inputs=np.arange(31, 41, dtype=np.int32).reshape(-1, 1),  # Smaller range
        valid_targets=np.round((2 ** np.arange(31, 41, dtype=np.int32)) / 4).astype(np.int64)
    ),
    Task(
        id=10,
        description="modulo 4",
        mode="1input",
        inputs=_x,
        targets=(_x.flatten() % 4).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=(_x_valid.flatten() % 4).astype(np.int32)
    ),
    Task(
        id=11,
        description="x3 mod 8",
        mode="1input",
        inputs=_x,
        targets=((_x.flatten() * 3) % 8).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=((_x_valid.flatten() * 3) % 8).astype(np.int32)
    ),
    Task(
        id=12,
        description="x2 mod 3",
        mode="1input",
        inputs=_x,
        targets=((_x.flatten() * 2) % 3).astype(np.int32),
        valid_inputs=_x_valid,
        valid_targets=((_x_valid.flatten() * 2) % 3).astype(np.int32)
    ),
    
    # 2-input tasks
    Task(
        id=20,
        description="summation (a+b)",
        mode="2input",
        inputs=_xy,
        targets=(_xy[:, 0] + _xy[:, 1]).astype(np.int32),
        valid_inputs=_xy_valid,
        valid_targets=(_xy_valid[:, 0] + _xy_valid[:, 1]).astype(np.int32)
    ),
    Task(
        id=21,
        description="multiplication (a*b)",
        mode="2input",
        inputs=_xy,
        targets=(_xy[:, 0] * _xy[:, 1]).astype(np.int32),
        valid_inputs=_xy_valid,
        valid_targets=(_xy_valid[:, 0] * _xy_valid[:, 1]).astype(np.int32)
    ),
    Task(
        id=22,
        description="subtraction (a-b)",
        mode="2input",
        inputs=_xy,
        targets=(_xy[:, 0] - _xy[:, 1]).astype(np.int32),
        valid_inputs=_xy_valid,
        valid_targets=(_xy_valid[:, 0] - _xy_valid[:, 1]).astype(np.int32)
    ),
    Task(
        id=23,
        description="maximum (max(a,b))",
        mode="2input",
        inputs=_xy,
        targets=np.maximum(_xy[:, 0], _xy[:, 1]).astype(np.int32),
        valid_inputs=_xy_valid,
        valid_targets=np.maximum(_xy_valid[:, 0], _xy_valid[:, 1]).astype(np.int32)
    ),
    Task(
        id=24,
        description="minimum (min(a,b))",
        mode="2input",
        inputs=_xy,
        targets=np.minimum(_xy[:, 0], _xy[:, 1]).astype(np.int32),
        valid_inputs=_xy_valid,
        valid_targets=np.minimum(_xy_valid[:, 0], _xy_valid[:, 1]).astype(np.int32)
    ),
    Task(
        id=25,
        description="GCD (gcd(a,b))",
        mode="2input",
        inputs=_xy,
        targets=np.array([math.gcd(a, b) for a, b in _xy], dtype=np.int32),
        valid_inputs=_xy_valid,
        valid_targets=np.array([math.gcd(a, b) for a, b in _xy_valid], dtype=np.int32)
    ),
    Task(
        id=26,
        description="LCM (lcm(a,b))",
        mode="2input",
        inputs=_xy,
        targets=np.array([abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0 
                          for a, b in _xy], dtype=np.int32),
        valid_inputs=_xy_valid,
        valid_targets=np.array([abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0 
                                for a, b in _xy_valid], dtype=np.int32)
    ),
]

# ========================= REGISTRY =========================
TASKS: dict[int, Task] = {task.id: task for task in tasks}

# ========================= CUSTOM TASK, edit this to set for your task =========================
custom_task = Task(
    id=-1,
    description="custom task",
    mode="1input",
    inputs=np.array([[1], [2], [3], [4], [5]], dtype=np.int32),
    targets=np.array([2, 4, 6, 8, 10], dtype=np.int32),
    valid_inputs=np.array([[6], [7], [8]], dtype=np.int32),  # Out-of-distribution validation
    valid_targets=np.array([12, 14, 16], dtype=np.int32)
)

TASKS[-1] = custom_task  # Add custom task

def get_dataset(task_id: int) -> Task:
    """Return Task dataclass for a given ID."""
    if task_id == -1:
        return custom_task
    return TASKS[task_id]