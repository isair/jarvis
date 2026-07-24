"""Owner-triggered Development Mode state machine.

Phase 4 · Section H (state-machine half). Development Mode lets Cora research,
build, and self-test a change in an isolated workspace — but ONLY when the
owner explicitly triggers it, and NEVER as a side effect of a memory recall, a
web result, or the model's own reasoning. It is a small, auditable, persistent
state machine with hard authority limits.

Guardrails encoded here (each has a test):

* **Owner-trigger only.** ``arm`` (and the ``research`` / ``develop_and_test``
  wrappers) refuse unless ``owner_triggered is True`` — a flag only an owner
  command path sets. Web / memory / model paths cannot set it, so they can
  never arm.
* **Bounded authority.** ``develop_and_test`` authorizes isolated workspace +
  local changes + tests + adversarial review, and NOTHING else. Push, PR,
  merge, deploy, delete, n8n, paid, and external actions are always denied via
  ``is_authorized`` — regardless of state.
* **No self-applied changes.** Reaching ``APPLYING_APPROVED_CHANGE`` requires
  an explicit ``request_activation`` + ``confirm_activation`` with a matching
  token. The generic ``advance`` refuses to target that state, so the machine
  can never apply a change to itself.
* **Single-flight + persistent + auditable.** One active job per workspace; a
  second ``arm`` is refused. The job is a serializable dict (``to_dict`` /
  ``from_dict``) for restart recovery. Every transition appends an audit event
  (actor / timestamp / from / to / reason). ``stop`` cancels while keeping the
  checkpoint + audit; a timeout fails the job the same way.
* **Provider gate.** Even a fully-armed job cannot invoke a real agent unless a
  non-disabled, available provider is wired. The default is ``DisabledProvider``.

Nothing here spawns a process, touches the network, or reads/writes live
config. It is pure in-memory policy; persistence is the caller's choice.
"""

from __future__ import annotations

import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Union

from .claude_bridge import (
    AgentRequest,
    AgentResult,
    CodingAgentProvider,
    DisabledProvider,
)

__all__ = [
    "DevState",
    "Authority",
    "DevJob",
    "DevelopmentMode",
    "DEVELOP_AND_TEST_AUTHORITIES",
    "FORBIDDEN_AUTHORITIES",
    "DevModeError",
    "NotOwnerTriggered",
    "SingleFlightViolation",
    "ActivationNotConfirmed",
    "AuthorityDenied",
    "InvalidTransition",
]


class DevState(str, Enum):
    IDLE = "idle"
    RESEARCHING = "researching"
    PLAN_READY = "plan_ready"
    IMPLEMENTING_ISOLATED = "implementing_isolated"
    TESTING = "testing"
    ADVERSARIAL_REVIEW = "adversarial_review"
    AWAITING_OWNER_APPROVAL = "awaiting_owner_approval"
    APPLYING_APPROVED_CHANGE = "applying_approved_change"
    LIVE_TEST = "live_test"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class Authority(str, Enum):
    # Granted by develop_and_test.
    ISOLATED_WORKSPACE = "isolated_workspace"
    LOCAL_CHANGES = "local_changes"
    RUN_TESTS = "run_tests"
    ADVERSARIAL_REVIEW = "adversarial_review"
    # Granted by research (read-only).
    RESEARCH = "research"
    # Never granted — always denied.
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    MERGE = "merge"
    DEPLOY = "deploy"
    DELETE = "delete"
    N8N = "n8n"
    PAID = "paid"
    EXTERNAL = "external"


DEVELOP_AND_TEST_AUTHORITIES = frozenset(
    {
        Authority.ISOLATED_WORKSPACE,
        Authority.LOCAL_CHANGES,
        Authority.RUN_TESTS,
        Authority.ADVERSARIAL_REVIEW,
    }
)

# These are NEVER authorized, no matter the job kind or state.
FORBIDDEN_AUTHORITIES = frozenset(
    {
        Authority.PUSH,
        Authority.PULL_REQUEST,
        Authority.MERGE,
        Authority.DEPLOY,
        Authority.DELETE,
        Authority.N8N,
        Authority.PAID,
        Authority.EXTERNAL,
    }
)
_FORBIDDEN_VALUES = frozenset(a.value for a in FORBIDDEN_AUTHORITIES)

_TERMINAL_STATES = frozenset(
    {DevState.DONE, DevState.FAILED, DevState.CANCELLED, DevState.ROLLED_BACK}
)

# Legal forward transitions. AWAITING_OWNER_APPROVAL -> APPLYING_APPROVED_CHANGE
# is present so ``confirm_activation`` can use it, but the public ``advance``
# refuses to target APPLYING directly (must go through confirm).
_ALLOWED_TRANSITIONS: Dict[DevState, frozenset] = {
    DevState.IDLE: frozenset({DevState.RESEARCHING}),
    DevState.RESEARCHING: frozenset(
        {DevState.PLAN_READY, DevState.DONE, DevState.FAILED, DevState.CANCELLED}
    ),
    DevState.PLAN_READY: frozenset(
        {DevState.IMPLEMENTING_ISOLATED, DevState.DONE, DevState.FAILED, DevState.CANCELLED}
    ),
    DevState.IMPLEMENTING_ISOLATED: frozenset(
        {DevState.TESTING, DevState.FAILED, DevState.CANCELLED}
    ),
    DevState.TESTING: frozenset(
        {DevState.ADVERSARIAL_REVIEW, DevState.FAILED, DevState.CANCELLED}
    ),
    DevState.ADVERSARIAL_REVIEW: frozenset(
        {DevState.AWAITING_OWNER_APPROVAL, DevState.FAILED, DevState.CANCELLED}
    ),
    DevState.AWAITING_OWNER_APPROVAL: frozenset(
        {DevState.APPLYING_APPROVED_CHANGE, DevState.CANCELLED, DevState.FAILED}
    ),
    DevState.APPLYING_APPROVED_CHANGE: frozenset(
        {DevState.LIVE_TEST, DevState.ROLLED_BACK, DevState.FAILED}
    ),
    DevState.LIVE_TEST: frozenset(
        {DevState.DONE, DevState.ROLLED_BACK, DevState.FAILED}
    ),
    DevState.DONE: frozenset(),
    DevState.FAILED: frozenset(),
    DevState.CANCELLED: frozenset(),
    DevState.ROLLED_BACK: frozenset(),
}


class DevModeError(RuntimeError):
    """Base class for Development Mode refusals."""


class NotOwnerTriggered(DevModeError):
    """Arming was attempted without an explicit owner trigger."""


class SingleFlightViolation(DevModeError):
    """A second job was armed while one is already active."""


class ActivationNotConfirmed(DevModeError):
    """APPLYING was attempted without an explicit confirmation token."""


class AuthorityDenied(DevModeError):
    """A transition/action needs an authority the job was not granted."""


class InvalidTransition(DevModeError):
    """An illegal state transition was requested."""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# Untrusted-checkpoint coercion helpers
#
# A serialized job/machine dict may have been tampered with (edited on disk,
# replayed, downgraded). ``from_dict`` treats every field as untrusted and runs
# it through these coercions before it can drive the state machine again.
# --------------------------------------------------------------------------- #
def _coerce_state(value) -> str:
    """Return a known ``DevState`` value; an unknown/tampered state -> FAILED."""
    try:
        return DevState(str(value)).value
    except ValueError:
        return DevState.FAILED.value


def _authorities_allowed_for_kind(kind: str) -> frozenset:
    """The authority values a job of this ``kind`` is ever allowed to hold."""
    if kind == "develop_and_test":
        return frozenset(a.value for a in DEVELOP_AND_TEST_AUTHORITIES)
    if kind == "research":
        return frozenset({Authority.RESEARCH.value})
    return frozenset()


def _audit_has_confirmed_activation(audit: List[dict]) -> bool:
    """True only if the audit trail proves a real request+confirm activation.

    ``request_activation`` appends an ``activation_requested`` event and
    ``confirm_activation`` appends an ``activation_confirmed`` event whose
    ``to`` is APPLYING_APPROVED_CHANGE. Both must be present for a checkpoint to
    be allowed to restore directly into APPLYING.
    """
    has_request = any(
        str(e.get("reason", "")) == "activation_requested" for e in audit
    )
    has_confirm = any(
        str(e.get("reason", "")) == "activation_confirmed"
        and str(e.get("to", "")) == DevState.APPLYING_APPROVED_CHANGE.value
        for e in audit
    )
    return has_request and has_confirm


def _coerce_deadline(value) -> Optional[str]:
    """Normalize a serialized deadline to a tz-aware ISO string (or drop it).

    A timezone-naive ``deadline_at`` makes ``_is_expired`` / ``check_timeout``
    compare naive-vs-aware datetimes and raise, silently disabling the timeout
    safety net. Naive values are assumed UTC; unparseable values are dropped.
    """
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _iso(dt)


@dataclass
class DevJob:
    """A single Development Mode job — fully serializable for restart recovery."""

    job_id: str
    kind: str  # "research" | "develop_and_test"
    owner_command: str
    state: str
    authorities: List[str]
    created_at: str
    updated_at: str
    deadline_at: Optional[str] = None
    confirmation_token: Optional[str] = None
    confirmed: bool = False
    reason: Optional[str] = None
    checkpoint: Optional[dict] = None
    audit: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "owner_command": self.owner_command,
            "state": self.state,
            "authorities": list(self.authorities),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_at": self.deadline_at,
            "confirmation_token": self.confirmation_token,
            "confirmed": bool(self.confirmed),
            "reason": self.reason,
            "checkpoint": self.checkpoint,
            "audit": [dict(e) for e in self.audit],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DevJob":
        """Rebuild a job from an **untrusted** serialized dict.

        A checkpoint on disk may have been tampered with, so nothing is copied
        verbatim into a position of authority:

        * (a) an unknown/invalid ``state`` is coerced to the inert FAILED state;
        * (b) ``APPLYING_APPROVED_CHANGE`` is only restored when the job was
          genuinely confirmed (``confirmed`` flag AND a matching request+confirm
          audit pair); otherwise it is downgraded to FAILED so a tampered
          checkpoint can never resume mid-apply while bypassing the two-step
          owner confirmation;
        * (c) restored authorities are intersected with what the job ``kind`` may
          ever hold and can never include a forbidden authority;
        * (d) a timezone-naive ``deadline_at`` is normalized to tz-aware UTC (or
          dropped) so the timeout safety net cannot be disabled.
        """
        kind = str(d["kind"])
        confirmed = bool(d.get("confirmed", False))
        audit = [dict(e) for e in d.get("audit", [])]

        # (a) unknown/tampered state -> inert FAILED.
        state = _coerce_state(d.get("state"))

        # (b) never restore directly into APPLYING without proof of a real,
        # owner-confirmed activation.
        if state == DevState.APPLYING_APPROVED_CHANGE.value and not (
            confirmed and _audit_has_confirmed_activation(audit)
        ):
            state = DevState.FAILED.value

        # (c) intersect restored authorities with the kind's allowed set and
        # drop any forbidden authority (belt-and-braces; allowed sets exclude
        # forbidden authorities already).
        allowed = _authorities_allowed_for_kind(kind)
        authorities = [
            a
            for a in (str(x) for x in d.get("authorities", []))
            if a in allowed and a not in _FORBIDDEN_VALUES
        ]

        # (d) normalize/reject a naive deadline.
        deadline_at = _coerce_deadline(d.get("deadline_at"))

        return cls(
            job_id=str(d["job_id"]),
            kind=kind,
            owner_command=str(d.get("owner_command", "")),
            state=state,
            authorities=authorities,
            created_at=str(d["created_at"]),
            updated_at=str(d["updated_at"]),
            deadline_at=deadline_at,
            confirmation_token=d.get("confirmation_token"),
            confirmed=confirmed,
            reason=d.get("reason"),
            checkpoint=d.get("checkpoint"),
            audit=audit,
        )


class DevelopmentMode:
    """Single-workspace, owner-triggered Development Mode controller."""

    def __init__(
        self,
        *,
        provider: Optional[CodingAgentProvider] = None,
        workspace_id: str = "default",
        timeout_s: int = 1800,
        clock=None,
    ) -> None:
        self._provider: CodingAgentProvider = provider if provider is not None else DisabledProvider()
        self._workspace_id = workspace_id
        self._default_timeout_s = int(timeout_s)
        self._clock = clock if clock is not None else _now_utc
        self._job: Optional[DevJob] = None
        self._lock = threading.RLock()

    # -- introspection -----------------------------------------------------
    @property
    def provider(self) -> CodingAgentProvider:
        return self._provider

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def job(self) -> Optional[DevJob]:
        return self._job

    @staticmethod
    def _is_terminal(state: str) -> bool:
        try:
            return DevState(state) in _TERMINAL_STATES
        except ValueError:
            return False

    def _is_expired(self, job: DevJob) -> bool:
        if not job.deadline_at or self._is_terminal(job.state):
            return False
        return self._clock() >= datetime.fromisoformat(job.deadline_at)

    def _require_active(self) -> DevJob:
        if self._job is None or self._is_terminal(self._job.state):
            raise DevModeError("no active development job")
        return self._job

    # -- arming ------------------------------------------------------------
    def arm(
        self,
        owner_command: str,
        *,
        owner_triggered: bool,
        kind: str = "research",
        timeout_s: Optional[int] = None,
    ) -> DevJob:
        """Arm a new job. Refuses unless ``owner_triggered is True``.

        Only an owner-command path sets ``owner_triggered``; web / memory /
        model paths cannot, so they can never arm.
        """
        if owner_triggered is not True:
            raise NotOwnerTriggered(
                "Development Mode can only be armed by an explicit owner command"
            )
        if kind not in ("research", "develop_and_test"):
            raise DevModeError(f"unknown development job kind: {kind!r}")

        with self._lock:
            if self._job is not None and not self._is_terminal(self._job.state):
                raise SingleFlightViolation(
                    "a development job is already active in this workspace"
                )
            now = self._clock()
            ttl = int(timeout_s if timeout_s is not None else self._default_timeout_s)
            job = DevJob(
                job_id=uuid.uuid4().hex,
                kind=kind,
                owner_command=(owner_command or "")[:2000],
                state=DevState.IDLE.value,
                authorities=[],
                created_at=_iso(now),
                updated_at=_iso(now),
                deadline_at=_iso(now + timedelta(seconds=ttl)),
            )
            self._job = job

            if kind == "research":
                job.authorities = [Authority.RESEARCH.value]
                self._transition(DevState.RESEARCHING, actor="owner", reason="research armed")
            else:  # develop_and_test
                job.authorities = sorted(a.value for a in DEVELOP_AND_TEST_AUTHORITIES)
                self._transition(DevState.RESEARCHING, actor="owner", reason="develop_and_test armed")
                self._transition(DevState.PLAN_READY, actor="owner", reason="plan drafted")
                self._transition(
                    DevState.IMPLEMENTING_ISOLATED, actor="owner", reason="isolated implementation started"
                )
            return job

    def research(self, owner_command: str, *, owner_triggered: bool) -> DevJob:
        """Owner command: read-only research. Grants only the RESEARCH authority."""
        return self.arm(owner_command, owner_triggered=owner_triggered, kind="research")

    def develop_and_test(self, owner_command: str, *, owner_triggered: bool) -> DevJob:
        """Owner command: isolated build + local changes + tests + review only."""
        return self.arm(owner_command, owner_triggered=owner_triggered, kind="develop_and_test")

    # -- transitions -------------------------------------------------------
    def _transition(self, to_state: DevState, *, actor: str, reason: Optional[str] = None) -> None:
        job = self._job
        assert job is not None  # callers hold the lock and ensured a job
        frm = DevState(job.state)
        allowed = _ALLOWED_TRANSITIONS.get(frm, frozenset())
        if to_state not in allowed:
            raise InvalidTransition(f"{frm.value} -> {to_state.value} is not allowed")
        if to_state == DevState.IMPLEMENTING_ISOLATED and Authority.LOCAL_CHANGES.value not in job.authorities:
            raise AuthorityDenied("local changes are not authorized for this job")
        if to_state == DevState.APPLYING_APPROVED_CHANGE and not job.confirmed:
            raise ActivationNotConfirmed(
                "APPLYING_APPROVED_CHANGE requires an explicit confirmed activation"
            )
        now = self._clock()
        job.state = to_state.value
        job.updated_at = _iso(now)
        job.audit.append(
            {
                "ts": _iso(now),
                "actor": actor,
                "from": frm.value,
                "to": to_state.value,
                "reason": reason or "",
            }
        )

    def advance(
        self,
        to_state: Union[DevState, str],
        *,
        actor: str = "owner",
        reason: Optional[str] = None,
    ) -> DevJob:
        """Advance the active job one legal step.

        Refuses to target ``APPLYING_APPROVED_CHANGE`` — that state is reachable
        only through ``request_activation`` + ``confirm_activation``.
        """
        to = to_state if isinstance(to_state, DevState) else DevState(to_state)
        with self._lock:
            job = self._require_active()
            if to == DevState.APPLYING_APPROVED_CHANGE:
                raise ActivationNotConfirmed(
                    "cannot self-advance to APPLYING_APPROVED_CHANGE; "
                    "use request_activation + confirm_activation"
                )
            self._transition(to, actor=actor, reason=reason)
            return job

    # -- activation (two-step, owner-confirmed) ----------------------------
    def request_activation(self, job_id: str) -> str:
        """Mint a one-time confirmation token for the pending change.

        Requires the job to be in ``AWAITING_OWNER_APPROVAL``. Does NOT advance
        the job; the owner must still ``confirm_activation`` with the token.
        """
        with self._lock:
            job = self._require_active()
            if job.job_id != job_id:
                raise DevModeError("job_id does not match the active job")
            if job.state != DevState.AWAITING_OWNER_APPROVAL.value:
                raise InvalidTransition(
                    "activation can only be requested from AWAITING_OWNER_APPROVAL"
                )
            token = secrets.token_hex(16)
            job.confirmation_token = token
            now = self._clock()
            job.updated_at = _iso(now)
            job.audit.append(
                {
                    "ts": _iso(now),
                    "actor": "owner",
                    "from": job.state,
                    "to": job.state,
                    "reason": "activation_requested",
                }
            )
            return token

    def confirm_activation(self, job_id: str, confirmation_token: str) -> DevJob:
        """Confirm and advance to ``APPLYING_APPROVED_CHANGE``.

        Refuses unless a matching token was minted by ``request_activation``.
        """
        with self._lock:
            job = self._require_active()
            if job.job_id != job_id:
                raise DevModeError("job_id does not match the active job")
            if not job.confirmation_token or confirmation_token != job.confirmation_token:
                raise ActivationNotConfirmed("invalid or missing confirmation token")
            if job.state != DevState.AWAITING_OWNER_APPROVAL.value:
                raise InvalidTransition(
                    "activation can only be confirmed from AWAITING_OWNER_APPROVAL"
                )
            job.confirmed = True
            self._transition(
                DevState.APPLYING_APPROVED_CHANGE, actor="owner", reason="activation_confirmed"
            )
            # Consume the token so it cannot be replayed.
            job.confirmation_token = None
            return job

    # -- lifecycle ---------------------------------------------------------
    def stop(self, *, actor: str = "owner", reason: str = "owner_stop") -> Optional[DevJob]:
        """Cancel the active job from any state, keeping its checkpoint + audit."""
        with self._lock:
            if self._job is None or self._is_terminal(self._job.state):
                return None
            job = self._job
            frm = job.state
            now = self._clock()
            job.state = DevState.CANCELLED.value
            job.updated_at = _iso(now)
            job.reason = reason
            job.audit.append(
                {
                    "ts": _iso(now),
                    "actor": actor,
                    "from": frm,
                    "to": DevState.CANCELLED.value,
                    "reason": reason,
                }
            )
            return job

    def check_timeout(self, *, actor: str = "system") -> bool:
        """Fail the active job if it has passed its deadline. Keeps audit."""
        with self._lock:
            job = self._job
            if job is None or self._is_terminal(job.state) or not job.deadline_at:
                return False
            now = self._clock()
            if now < datetime.fromisoformat(job.deadline_at):
                return False
            frm = job.state
            job.state = DevState.FAILED.value
            job.updated_at = _iso(now)
            job.reason = "timeout"
            job.audit.append(
                {
                    "ts": _iso(now),
                    "actor": actor,
                    "from": frm,
                    "to": DevState.FAILED.value,
                    "reason": "timeout",
                }
            )
            return True

    # -- authority ---------------------------------------------------------
    def is_authorized(self, authority: Union[Authority, str]) -> bool:
        """True only for an authority granted to the active job.

        Forbidden authorities (push/PR/merge/deploy/delete/n8n/paid/external)
        are always denied, even if somehow present on the job.
        """
        a = authority.value if isinstance(authority, Authority) else str(authority)
        with self._lock:
            job = self._job
            if job is None or self._is_terminal(job.state):
                return False
            if a in _FORBIDDEN_VALUES:
                return False
            return a in set(job.authorities)

    # -- provider gate -----------------------------------------------------
    def can_invoke_agent(self) -> bool:
        """A real agent may run only with a non-disabled, available provider."""
        return (not isinstance(self._provider, DisabledProvider)) and bool(
            self._provider.is_available()
        )

    def invoke_agent(self, req: AgentRequest) -> AgentResult:
        """Invoke the wired provider, gated by job activity + provider status.

        Refuses (ok=False) with the default ``DisabledProvider`` or when no job
        is active. Never spawns anything itself.
        """
        with self._lock:
            job = self._job
            active = job is not None and not self._is_terminal(job.state)
        if not active:
            return AgentResult(ok=False, error="no active development job")
        if not self.can_invoke_agent():
            return AgentResult(ok=False, error="coding agent provider disabled or unavailable")
        return self._provider.run(req)

    # -- status + persistence ---------------------------------------------
    def status(self) -> dict:
        """A snapshot dict (idle sentinel when no job has been armed)."""
        with self._lock:
            if self._job is None:
                return {
                    "workspace_id": self._workspace_id,
                    "state": DevState.IDLE.value,
                    "active": False,
                    "job_id": None,
                }
            d = self._job.to_dict()
            d["workspace_id"] = self._workspace_id
            d["active"] = not self._is_terminal(self._job.state)
            d["expired"] = self._is_expired(self._job)
            return d

    def to_dict(self) -> dict:
        """Serialize for restart recovery (provider is re-attached on restore)."""
        with self._lock:
            return {
                "workspace_id": self._workspace_id,
                "default_timeout_s": self._default_timeout_s,
                "job": self._job.to_dict() if self._job is not None else None,
            }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        provider: Optional[CodingAgentProvider] = None,
        clock=None,
    ) -> "DevelopmentMode":
        inst = cls(
            provider=provider,
            workspace_id=str(data.get("workspace_id", "default")),
            timeout_s=int(data.get("default_timeout_s", 1800)),
            clock=clock,
        )
        job_d = data.get("job")
        inst._job = DevJob.from_dict(job_d) if job_d else None
        return inst
