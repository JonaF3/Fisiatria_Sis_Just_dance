"""
rehab_core

Módulo central para evaluación de rehabilitación física.
"""

from .angles import (
    calculate_angle,
    calculate_angles,
    canonical_angle_name,
    filter_angles,
)

from .evaluator import RehabExerciseEvaluator
from .controller_bridge import RehabControllerBridge
from .session_state import RehabSessionState