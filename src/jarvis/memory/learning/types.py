"""Lesson types and dataclasses for Cora Learning Loop v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LessonType(str, Enum):
    USER_PREFERENCE = "user_preference"
    USER_CORRECTION = "user_correction"
    USER_FACT = "user_fact"
    PROJECT_CONTEXT = "project_context"
    PENDING_STEP = "pending_step"
    ASR_CORRECTION = "asr_correction"
    RECURRING_FAILURE = "recurring_failure"
    IMPROVEMENT_CANDIDATE = "improvement_candidate"


class LessonStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    COMPLETED = "completed"


class Namespace(str, Enum):
    PROFILE = "profile"
    CORRECTIONS = "corrections"
    PROJECTS = "projects"
    ASR = "asr"
    WORLD = "world"
    IMPROVEMENTS = "improvements"


LESSON_TYPES = {t.value for t in LessonType}

# Max confidence by provenance — enforced on write.
CONFIDENCE_CAPS = {
    "user_explicit_correction": 1.0,
    "user_direct": 0.95,
    "tool_confirmed": 0.80,
    "model_inference": 0.40,
}

NAMESPACE_FOR_TYPE = {
    LessonType.USER_PREFERENCE: Namespace.PROFILE,
    LessonType.USER_FACT: Namespace.PROFILE,
    LessonType.USER_CORRECTION: Namespace.CORRECTIONS,
    LessonType.ASR_CORRECTION: Namespace.ASR,
    LessonType.PROJECT_CONTEXT: Namespace.PROJECTS,
    LessonType.PENDING_STEP: Namespace.PROJECTS,
    LessonType.RECURRING_FAILURE: Namespace.IMPROVEMENTS,
    LessonType.IMPROVEMENT_CANDIDATE: Namespace.IMPROVEMENTS,
}


@dataclass
class Lesson:
    id: str
    lesson_type: str
    subject_key: str
    value: str
    source_quote: str
    conversation_id: str
    turn_id: str
    created_at: str
    updated_at: str
    confidence: float
    sensitivity: str = "normal"
    status: str = LessonStatus.ACTIVE.value
    expires_at: Optional[str] = None
    namespace: str = Namespace.PROFILE.value
    occurrence_count: int = 1
    provenance: str = "user_direct"

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "lesson_type": self.lesson_type,
            "subject_key": self.subject_key,
            "value": self.value,
            "source_quote": self.source_quote,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity,
            "status": self.status,
            "expires_at": self.expires_at,
            "namespace": self.namespace,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class LessonCandidate:
    """Proposed lesson before validation / persistence."""

    lesson_type: str
    subject_key: str
    value: str
    source_quote: str
    confidence: float
    provenance: str = "user_direct"
    turn_id: str = ""
    sensitivity: str = "normal"
    meta: dict = field(default_factory=dict)
