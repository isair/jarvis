"""Cora Learning Loop v1 — structured, verified lessons in the existing profile DB.

Safety: lessons never modify code, config, models, prompts, or authorize tools.
``improvement_candidate`` is proposal-only and never auto-executed.
"""

from .types import LESSON_TYPES, Lesson, LessonStatus, LessonType, Namespace
from .store import LearningStore
from .extract import extract_lessons_from_turns, is_trivial_conversation
from .retrieve import format_lessons_for_prompt, retrieve_relevant_lessons
from .commands import try_learning_command
from .worker import LearningWorker, get_learning_worker

__all__ = [
    "LESSON_TYPES",
    "Lesson",
    "LessonStatus",
    "LessonType",
    "Namespace",
    "LearningStore",
    "extract_lessons_from_turns",
    "is_trivial_conversation",
    "format_lessons_for_prompt",
    "retrieve_relevant_lessons",
    "try_learning_command",
    "LearningWorker",
    "get_learning_worker",
]
