"""Coding-agent provider abstraction for owner-triggered Development Mode.

Phase 4 · Section H (bridge half). Development Mode must be able to drive a
real coding agent (Claude Code CLI) WITHOUT welding the policy to a single
executable. Everything here binds to an *interface* (``CodingAgentProvider``),
so the state machine in ``mode.py`` never shells out directly and the danger
rules live in one place.

Design invariants (all safety-first):

* **Default is inert.** ``get_provider(name, enabled=...)`` returns the no-op
  ``DisabledProvider`` unless the owner both selects ``"claude_cli"`` *and*
  enables it. Even the enabled ``ClaudeCliProvider`` refuses to ``run()``
  unless it is explicitly enabled, available on ``PATH``, and handed an
  executor — the foundation ships **no** executor, so nothing is ever spawned.
* **argv, never a shell string.** ``build_invocation`` returns an ``ArgvSpec``
  whose ``argv`` is a ``list[str]`` passed verbatim to ``subprocess`` by any
  future executor (no ``shell=True``, no string interpolation → no shell
  injection surface).
* **No permission bypass, ever.** ``build_invocation`` raises
  ``DangerousFlagRequested`` if a request tries to smuggle
  ``--dangerously-skip-permissions`` /
  ``--allow-dangerously-skip-permissions`` through any control field, and it
  never emits such a flag itself. The advertised permission modes deliberately
  exclude ``bypassPermissions``.

No network calls, no subprocess, and no secrets are made or logged here.
"""

from __future__ import annotations

import abc
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "AgentRequest",
    "AgentResult",
    "ArgvSpec",
    "Caps",
    "CodingAgentProvider",
    "DisabledProvider",
    "ClaudeCliProvider",
    "get_provider",
    "DangerousFlagRequested",
    "ALLOWED_PERMISSION_MODES",
]

# Permission modes the policy permits. ``bypassPermissions`` is intentionally
# absent: this bridge never runs an agent that bypasses the permission prompt.
ALLOWED_PERMISSION_MODES = ("default", "plan", "acceptEdits")

# Any control token containing this substring is a permission-bypass attempt.
# Covers both ``--dangerously-skip-permissions`` and
# ``--allow-dangerously-skip-permissions``.
_DANGEROUS_SUBSTRING = "dangerously-skip-permissions"


class DangerousFlagRequested(ValueError):
    """Raised when an invocation would bypass the permission system."""


@dataclass
class AgentRequest:
    """A single non-interactive coding-agent task.

    ``allowed_tools`` is a strict allowlist (joined into ``--allowedTools``).
    ``json_schema`` is carried for callers that constrain structured output;
    it is not injected as a CLI flag here (no fabricated flags).
    """

    prompt: str
    cwd: str
    model: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    permission_mode: str = "plan"
    output_format: str = "json"
    json_schema: Optional[dict] = None
    timeout_s: float = 300.0
    max_cost: Optional[float] = None


@dataclass
class AgentResult:
    """Outcome of an agent invocation. ``ok=False`` means refused or failed."""

    ok: bool
    exit_code: Optional[int] = None
    stdout_json: Optional[dict] = None
    session_id: Optional[str] = None
    cost: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ArgvSpec:
    """A ready-to-spawn invocation: an argv **list**, an env overlay, a cwd.

    ``argv`` is always a ``list[str]`` so executors call ``subprocess`` without
    ``shell=True``; there is deliberately no shell-string form.
    """

    argv: List[str]
    env: Dict[str, str]
    cwd: str


@dataclass
class Caps:
    """What a provider can do — used to negotiate features, not to grant them."""

    json_output: bool
    structured_schema: bool
    tool_allowlist: bool
    permission_modes: tuple
    model_select: bool


# An executor turns an ArgvSpec into an AgentResult. The foundation ships none;
# it is injected only by an owner-triggered flow that actually intends to spawn.
Executor = Callable[..., AgentResult]


class CodingAgentProvider(abc.ABC):
    """Abstract coding-agent provider. Policy binds to this, not to a CLI."""

    @property
    @abc.abstractmethod
    def name(self) -> str:  # pragma: no cover - trivial
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        ...

    @abc.abstractmethod
    def supports_noninteractive(self) -> bool:
        ...

    @abc.abstractmethod
    def capabilities(self) -> Caps:
        ...

    @abc.abstractmethod
    def build_invocation(self, req: AgentRequest) -> ArgvSpec:
        ...

    @abc.abstractmethod
    def run(self, req: AgentRequest) -> AgentResult:
        ...


class DisabledProvider(CodingAgentProvider):
    """The default, inert provider. It exists, but it never invokes anything."""

    @property
    def name(self) -> str:
        return "disabled"

    def is_available(self) -> bool:
        # The no-op provider is always "available" — it just refuses to act.
        return True

    def supports_noninteractive(self) -> bool:
        return True

    def capabilities(self) -> Caps:
        return Caps(
            json_output=False,
            structured_schema=False,
            tool_allowlist=False,
            permission_modes=(),
            model_select=False,
        )

    def build_invocation(self, req: AgentRequest) -> ArgvSpec:
        # Deliberately builds nothing: the disabled provider must never produce
        # a spawnable invocation.
        raise RuntimeError("disabled provider does not build invocations")

    def run(self, req: AgentRequest) -> AgentResult:
        return AgentResult(ok=False, error="development agent disabled")


class ClaudeCliProvider(CodingAgentProvider):
    """Provider for the Claude Code CLI (``claude -p`` non-interactive mode).

    Building the argv is always safe; **running** requires an explicit
    ``enabled`` flag, CLI availability, and an injected ``executor``. The
    foundation injects no executor, so ``run()`` refuses by default and never
    spawns a subprocess (in tests or at rest).
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        executable: str = "claude",
        available: Optional[bool] = None,
        executor: Optional[Executor] = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._executable = executable
        # ``available`` overrides PATH detection (used by tests so they never
        # depend on whether the host actually has claude installed).
        self._available_override = available
        self._executor = executor

    @property
    def name(self) -> str:
        return "claude_cli"

    def is_available(self) -> bool:
        if self._available_override is not None:
            return bool(self._available_override)
        return shutil.which(self._executable) is not None

    def supports_noninteractive(self) -> bool:
        return True

    def capabilities(self) -> Caps:
        return Caps(
            json_output=True,
            structured_schema=True,
            tool_allowlist=True,
            # Note: no ``bypassPermissions`` — policy never permits it.
            permission_modes=ALLOWED_PERMISSION_MODES,
            model_select=True,
        )

    def build_invocation(self, req: AgentRequest) -> ArgvSpec:
        """Emit a safe argv **list** for ``claude -p``.

        Raises ``DangerousFlagRequested`` if any control field tries to request
        a permission-bypass flag, and ``ValueError`` for an out-of-policy
        permission mode. Never emits a bypass flag itself.
        """
        # 1) Refuse any permission-bypass smuggled through a control field.
        self._assert_no_dangerous(req)

        mode = (req.permission_mode or "plan").strip() or "plan"
        if mode not in ALLOWED_PERMISSION_MODES:
            raise ValueError(
                f"permission_mode {mode!r} not allowed; "
                f"choose from {ALLOWED_PERMISSION_MODES}"
            )

        argv: List[str] = [self._executable, "-p", "--output-format", (req.output_format or "json")]
        argv += ["--permission-mode", mode]
        if req.allowed_tools:
            argv += ["--allowedTools", ",".join(str(t) for t in req.allowed_tools)]
        if req.model:
            argv += ["--model", str(req.model)]
        # The prompt is a trailing positional value, never a shell string. The
        # timeout is honoured by run()/the executor (subprocess timeout), not a
        # CLI flag. Emit an explicit ``--`` end-of-options sentinel immediately
        # before it so a prompt that begins with ``--add-dir`` / ``--mcp-config``
        # / any other flag-looking text is parsed as the positional prompt and
        # can never be interpreted as a CLI option.
        argv.append("--")
        argv.append(str(req.prompt))

        # Belt-and-braces: guarantee no bypass flag ever leaves this method.
        for token in argv:
            if _DANGEROUS_SUBSTRING in str(token).lower():
                raise DangerousFlagRequested(
                    "refusing to emit a permission-bypass flag"
                )

        return ArgvSpec(argv=argv, env={}, cwd=req.cwd)

    def run(self, req: AgentRequest) -> AgentResult:
        """Refuse unless explicitly enabled AND available AND given an executor.

        The foundation never supplies an executor, so this refuses (ok=False)
        without ever spawning a process.
        """
        if not self._enabled:
            return AgentResult(ok=False, error="development agent not enabled")
        if not self.is_available():
            return AgentResult(ok=False, error="claude CLI not available on PATH")
        try:
            spec = self.build_invocation(req)
        except (DangerousFlagRequested, ValueError) as exc:
            return AgentResult(ok=False, error=str(exc))
        if self._executor is None:
            # No executor wired: build-only foundation, never spawns.
            return AgentResult(
                ok=False,
                error="no executor wired (foundation is build-only; refuses to spawn)",
            )
        return self._executor(spec, timeout_s=req.timeout_s)

    @staticmethod
    def _assert_no_dangerous(req: AgentRequest) -> None:
        tokens = [req.permission_mode or "", req.model or "", *(req.allowed_tools or [])]
        for tok in tokens:
            if _DANGEROUS_SUBSTRING in str(tok).lower():
                raise DangerousFlagRequested(
                    "permission-bypass flag is forbidden in Development Mode"
                )


def get_provider(name: str, *, enabled: bool) -> CodingAgentProvider:
    """Return a provider by name. Fail-safe to the inert ``DisabledProvider``.

    Only ``name == "claude_cli"`` AND ``enabled`` yields a ``ClaudeCliProvider``
    (and even that refuses to run without an executor). Anything else — a
    disabled flag, an unknown name — is the no-op provider.
    """
    if (name or "").strip().lower() == "claude_cli" and enabled:
        return ClaudeCliProvider(enabled=True)
    return DisabledProvider()
