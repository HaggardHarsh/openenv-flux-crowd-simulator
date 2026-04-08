"""
AI-Powered Crowd Management OpenEnv Environment
=================================================
A simulation-based OpenEnv environment for training AI agents
to optimize crowd management strategies and prevent stampede situations.
"""

from crowd_env.models import (
    ZoneInfo,
    Observation,
    Action,
    State,
    StepResult,
    ActionType,
    RiskLevel,
)
from crowd_env.environment import CrowdManagementEnv
from crowd_env.tasks import TaskConfig, TASKS
from crowd_env.grader import CrowdManagementGrader

__all__ = [
    "ZoneInfo",
    "Observation",
    "Action",
    "State",
    "StepResult",
    "ActionType",
    "RiskLevel",
    "CrowdManagementEnv",
    "TaskConfig",
    "TASKS",
    "CrowdManagementGrader",
]

__version__ = "1.0.0"
