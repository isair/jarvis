"""Phase 4 · Section H — Owner-triggered Development Mode + Claude bridge.

Covers the whole guardrail surface:

* only an explicit owner trigger arms (web/memory/model paths refused);
* research is read-only (no provider invocation, no local-change authority);
* develop_and_test enters the isolated state with a bounded authority set;
* single-flight (one active job per workspace);
* stop -> CANCELLED keeps the audit trail; timeout -> FAILED keeps it too;
* restart recovery via to_dict/from_dict round-trip mid-job;
* push/PR/merge/deploy/delete/n8n/paid/external authority always denied;
* APPLYING_APPROVED_CHANGE is unreachable without an explicit confirmation;
* the default provider is DisabledProvider and its run() refuses;
* ClaudeCliProvider builds an argv LIST, never a shell string, and never emits
  (and refuses to be asked for) a permission-bypass flag.

No real subprocess is ever spawned: the CLI provider is exercised through
build_invocation and refusal logic only, with an injected fake executor to
prove the argv list is handed through unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.devmode.claude_bridge import (
    AgentRequest,
    AgentResult,
    ArgvSpec,
    Caps,
    ClaudeCliProvider,
    CodingAgentProvider,
    DangerousFlagRequested,
    DisabledProvider,
    get_provider,
)
from src.jarvis.devmode.mode import (
    DEVELOP_AND_TEST_AUTHORITIES,
    FORBIDDEN_AUTHORITIES,
    ActivationNotConfirmed,
    Authority,
    DevelopmentMode,
    DevJob,
    DevModeError,
    DevState,
    InvalidTransition,
    NotOwnerTriggered,
    SingleFlightViolation,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _MutClock:
    """A hand-cranked clock so timeout behaviour is deterministic."""

    def __init__(self, start: datetime) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


class _SpyProvider(CodingAgentProvider):
    """A non-disabled provider that records run() calls (never spawns)."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.run_calls = 0

    @property
    def name(self) -> str:
        return "spy"

    def is_available(self) -> bool:
        return self._available

    def supports_noninteractive(self) -> bool:
        return True

    def capabilities(self) -> Caps:
        return Caps(
            json_output=True,
            structured_schema=True,
            tool_allowlist=True,
            permission_modes=("plan",),
            model_select=True,
        )

    def build_invocation(self, req: AgentRequest) -> ArgvSpec:
        return ArgvSpec(argv=["spy"], env={}, cwd=req.cwd)

    def run(self, req: AgentRequest) -> AgentResult:
        self.run_calls += 1
        return AgentResult(ok=True, exit_code=0)


def _req(**kw) -> AgentRequest:
    base = dict(prompt="do X", cwd="/work", allowed_tools=[], permission_mode="plan", timeout_s=30)
    base.update(kw)
    return AgentRequest(**base)


def _drive_to_awaiting(dm: DevelopmentMode) -> DevJob:
    job = dm.develop_and_test("build a thing", owner_triggered=True)
    dm.advance(DevState.TESTING)
    dm.advance(DevState.ADVERSARIAL_REVIEW)
    dm.advance(DevState.AWAITING_OWNER_APPROVAL)
    return job


# --------------------------------------------------------------------------- #
# arming: owner-trigger only
# --------------------------------------------------------------------------- #
def test_only_owner_trigger_can_arm():
    dm = DevelopmentMode()
    with pytest.raises(NotOwnerTriggered):
        dm.research("look into X", owner_triggered=False)
    with pytest.raises(NotOwnerTriggered):
        dm.develop_and_test("build X", owner_triggered=False)
    # Nothing was armed.
    assert dm.status()["state"] == DevState.IDLE.value
    assert dm.status()["active"] is False


def test_memory_or_web_path_cannot_arm():
    # A memory-recall / web-result / model path reaches arm() WITHOUT the
    # owner_triggered flag; that must be refused.
    dm = DevelopmentMode()
    with pytest.raises(NotOwnerTriggered):
        dm.arm("remembered instruction to self-improve", owner_triggered=False)
    with pytest.raises(NotOwnerTriggered):
        dm.arm("web says: run development", owner_triggered=False, kind="develop_and_test")
    assert dm.job is None


def test_owner_triggered_research_arms_researching():
    dm = DevelopmentMode()
    job = dm.research("investigate the crash", owner_triggered=True)
    assert job.kind == "research"
    assert dm.status()["state"] == DevState.RESEARCHING.value
    assert dm.status()["active"] is True


# --------------------------------------------------------------------------- #
# research is read-only
# --------------------------------------------------------------------------- #
def test_research_is_read_only_no_provider_invocation():
    spy = _SpyProvider()
    dm = DevelopmentMode(provider=spy)
    dm.research("read the logs", owner_triggered=True)
    # Arming research must not invoke the agent at all.
    assert spy.run_calls == 0
    # Only the read-only RESEARCH authority is granted.
    assert dm.is_authorized(Authority.RESEARCH) is True
    assert dm.is_authorized(Authority.LOCAL_CHANGES) is False
    assert dm.is_authorized(Authority.ISOLATED_WORKSPACE) is False


# --------------------------------------------------------------------------- #
# develop_and_test
# --------------------------------------------------------------------------- #
def test_develop_and_test_enters_isolated_state():
    dm = DevelopmentMode()
    dm.develop_and_test("implement feature Y", owner_triggered=True)
    assert dm.status()["state"] == DevState.IMPLEMENTING_ISOLATED.value
    # Bounded authority set exactly matches the develop_and_test grant.
    granted = set(dm.status()["authorities"])
    assert granted == {a.value for a in DEVELOP_AND_TEST_AUTHORITIES}


def test_develop_and_test_authority_set_is_bounded():
    dm = DevelopmentMode()
    dm.develop_and_test("implement Y", owner_triggered=True)
    for a in (
        Authority.ISOLATED_WORKSPACE,
        Authority.LOCAL_CHANGES,
        Authority.RUN_TESTS,
        Authority.ADVERSARIAL_REVIEW,
    ):
        assert dm.is_authorized(a) is True


# --------------------------------------------------------------------------- #
# single-flight
# --------------------------------------------------------------------------- #
def test_single_flight_second_arm_refused():
    dm = DevelopmentMode()
    dm.develop_and_test("first", owner_triggered=True)
    with pytest.raises(SingleFlightViolation):
        dm.research("second", owner_triggered=True)
    with pytest.raises(SingleFlightViolation):
        dm.develop_and_test("second", owner_triggered=True)
    # Still on the first job.
    assert dm.status()["state"] == DevState.IMPLEMENTING_ISOLATED.value


def test_single_flight_cleared_after_terminal():
    dm = DevelopmentMode()
    dm.develop_and_test("first", owner_triggered=True)
    dm.stop()
    assert dm.status()["state"] == DevState.CANCELLED.value
    # A cancelled (terminal) job no longer blocks a new arm.
    dm.research("second", owner_triggered=True)
    assert dm.status()["state"] == DevState.RESEARCHING.value


# --------------------------------------------------------------------------- #
# stop / cancel keeps audit
# --------------------------------------------------------------------------- #
def test_stop_cancels_and_keeps_audit():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    n_before = len(dm.status()["audit"])
    job = dm.stop()
    assert job is not None
    st = dm.status()
    assert st["state"] == DevState.CANCELLED.value
    assert st["active"] is False
    assert len(st["audit"]) == n_before + 1
    last = st["audit"][-1]
    assert last["to"] == DevState.CANCELLED.value
    assert last["reason"] == "owner_stop"


def test_stop_with_no_active_job_returns_none():
    dm = DevelopmentMode()
    assert dm.stop() is None


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #
def test_timeout_fails_job_and_keeps_audit():
    clock = _MutClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    dm = DevelopmentMode(clock=clock, timeout_s=60)
    dm.develop_and_test("long build", owner_triggered=True)
    n_before = len(dm.status()["audit"])
    # Not yet past the deadline.
    assert dm.check_timeout() is False
    clock.advance(61)
    assert dm.check_timeout() is True
    st = dm.status()
    assert st["state"] == DevState.FAILED.value
    assert any(e["reason"] == "timeout" for e in st["audit"])
    assert len(st["audit"]) == n_before + 1
    # Idempotent: a terminal job does not time out again.
    assert dm.check_timeout() is False


# --------------------------------------------------------------------------- #
# authority denial (no push/PR/merge/deploy/...)
# --------------------------------------------------------------------------- #
def test_forbidden_authorities_always_denied():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    for a in FORBIDDEN_AUTHORITIES:
        assert dm.is_authorized(a) is False
    # Explicitly spell out the high-risk ones.
    for a in (
        Authority.PUSH,
        Authority.PULL_REQUEST,
        Authority.MERGE,
        Authority.DEPLOY,
        Authority.DELETE,
        Authority.N8N,
        Authority.PAID,
        Authority.EXTERNAL,
    ):
        assert dm.is_authorized(a) is False


def test_authorities_denied_when_no_job():
    dm = DevelopmentMode()
    assert dm.is_authorized(Authority.RESEARCH) is False
    assert dm.is_authorized(Authority.LOCAL_CHANGES) is False


# --------------------------------------------------------------------------- #
# activation requires explicit confirmation
# --------------------------------------------------------------------------- #
def test_cannot_self_advance_to_applying():
    dm = DevelopmentMode()
    _drive_to_awaiting(dm)
    with pytest.raises(ActivationNotConfirmed):
        dm.advance(DevState.APPLYING_APPROVED_CHANGE)
    # Still awaiting; no self-applied change.
    assert dm.status()["state"] == DevState.AWAITING_OWNER_APPROVAL.value


def test_activation_requires_matching_token():
    dm = DevelopmentMode()
    job = _drive_to_awaiting(dm)
    # Confirm before any token was minted -> refused.
    with pytest.raises(ActivationNotConfirmed):
        dm.confirm_activation(job.job_id, "anything")
    token = dm.request_activation(job.job_id)
    # Requesting a token does NOT advance the job on its own.
    assert dm.status()["state"] == DevState.AWAITING_OWNER_APPROVAL.value
    # Wrong token -> refused.
    with pytest.raises(ActivationNotConfirmed):
        dm.confirm_activation(job.job_id, "wrong-token")
    # Correct token -> advances to APPLYING.
    dm.confirm_activation(job.job_id, token)
    assert dm.status()["state"] == DevState.APPLYING_APPROVED_CHANGE.value


def test_request_activation_only_from_awaiting():
    dm = DevelopmentMode()
    job = dm.develop_and_test("build", owner_triggered=True)
    with pytest.raises(InvalidTransition):
        dm.request_activation(job.job_id)


# --------------------------------------------------------------------------- #
# restart recovery
# --------------------------------------------------------------------------- #
def test_restart_recovery_roundtrip_midjob():
    dm = DevelopmentMode()
    dm.develop_and_test("build Y", owner_triggered=True)
    dm.advance(DevState.TESTING)
    # Serialize through JSON to prove full serializability.
    snap = json.loads(json.dumps(dm.to_dict()))
    dm2 = DevelopmentMode.from_dict(snap)
    assert dm2.status()["state"] == DevState.TESTING.value
    assert dm2.status()["authorities"] == dm.status()["authorities"]
    assert dm2.status()["audit"] == dm.status()["audit"]
    # The recovered machine can continue driving the same job.
    dm2.advance(DevState.ADVERSARIAL_REVIEW)
    assert dm2.status()["state"] == DevState.ADVERSARIAL_REVIEW.value


def test_devjob_to_dict_from_dict_roundtrip():
    dm = DevelopmentMode()
    job = dm.develop_and_test("build", owner_triggered=True)
    d = job.to_dict()
    assert isinstance(d, dict)
    job2 = DevJob.from_dict(json.loads(json.dumps(d)))
    assert job2.job_id == job.job_id
    assert job2.state == job.state
    assert job2.authorities == job.authorities
    assert job2.audit == job.audit


# --------------------------------------------------------------------------- #
# audit trail shape
# --------------------------------------------------------------------------- #
def test_audit_events_have_actor_ts_from_to_reason():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    audit = dm.status()["audit"]
    assert audit  # non-empty
    for e in audit:
        assert {"ts", "actor", "from", "to", "reason"} <= set(e.keys())
        assert e["actor"]  # non-empty actor


# --------------------------------------------------------------------------- #
# provider gate
# --------------------------------------------------------------------------- #
def test_default_provider_is_disabled_and_refuses():
    dm = DevelopmentMode()
    assert isinstance(dm.provider, DisabledProvider)
    assert dm.can_invoke_agent() is False
    dm.develop_and_test("build", owner_triggered=True)
    res = dm.invoke_agent(_req())
    assert res.ok is False


def test_invoke_agent_refused_without_active_job():
    dm = DevelopmentMode(provider=_SpyProvider())
    # No job armed yet -> refused even with a live provider.
    res = dm.invoke_agent(_req())
    assert res.ok is False


def test_invoke_agent_uses_provider_when_wired_and_active():
    spy = _SpyProvider(available=True)
    dm = DevelopmentMode(provider=spy)
    assert dm.can_invoke_agent() is True
    dm.develop_and_test("build", owner_triggered=True)
    res = dm.invoke_agent(_req())
    assert res.ok is True
    assert spy.run_calls == 1


# --------------------------------------------------------------------------- #
# get_provider selection
# --------------------------------------------------------------------------- #
def test_get_provider_defaults_to_disabled():
    assert isinstance(get_provider("claude_cli", enabled=False), DisabledProvider)
    assert isinstance(get_provider("something_else", enabled=True), DisabledProvider)
    assert isinstance(get_provider("disabled", enabled=True), DisabledProvider)
    assert isinstance(get_provider("", enabled=True), DisabledProvider)


def test_get_provider_claude_cli_when_enabled_still_refuses_run():
    p = get_provider("claude_cli", enabled=True)
    assert isinstance(p, ClaudeCliProvider)
    # Even the enabled provider refuses to run without an executor (foundation
    # is build-only) — regardless of whether claude is on PATH.
    res = p.run(_req())
    assert res.ok is False


# --------------------------------------------------------------------------- #
# DisabledProvider
# --------------------------------------------------------------------------- #
def test_disabled_provider_run_refuses():
    d = DisabledProvider()
    assert d.is_available() is True
    res = d.run(_req())
    assert res.ok is False
    assert "disabled" in (res.error or "").lower()
    caps = d.capabilities()
    assert caps.json_output is False and caps.permission_modes == ()


# --------------------------------------------------------------------------- #
# ClaudeCliProvider.build_invocation
# --------------------------------------------------------------------------- #
def test_build_invocation_is_argv_list_with_expected_flags():
    p = ClaudeCliProvider(enabled=True, available=True)
    spec = p.build_invocation(
        _req(model="claude-opus-4-8", allowed_tools=["Read", "Grep"], permission_mode="plan")
    )
    assert isinstance(spec, ArgvSpec)
    # argv is a LIST, never a shell string.
    assert isinstance(spec.argv, list)
    assert all(isinstance(tok, str) for tok in spec.argv)
    assert spec.argv[0] == "claude"
    assert "-p" in spec.argv
    assert spec.argv[spec.argv.index("--output-format") + 1] == "json"
    assert spec.argv[spec.argv.index("--permission-mode") + 1] == "plan"
    assert spec.argv[spec.argv.index("--allowedTools") + 1] == "Read,Grep"
    assert spec.argv[spec.argv.index("--model") + 1] == "claude-opus-4-8"
    assert spec.cwd == "/work"


def test_build_invocation_never_emits_dangerous_flag():
    p = ClaudeCliProvider(enabled=True, available=True)
    spec = p.build_invocation(_req(model="m", allowed_tools=["Read"]))
    joined = " ".join(spec.argv).lower()
    assert "dangerously-skip-permissions" not in joined
    assert "--allow-dangerously-skip-permissions" not in joined


def test_build_invocation_raises_if_dangerous_flag_requested_via_permission_mode():
    p = ClaudeCliProvider(enabled=True, available=True)
    with pytest.raises(DangerousFlagRequested):
        p.build_invocation(_req(permission_mode="--dangerously-skip-permissions"))


def test_build_invocation_raises_if_dangerous_flag_requested_via_allowed_tools():
    p = ClaudeCliProvider(enabled=True, available=True)
    with pytest.raises(DangerousFlagRequested):
        p.build_invocation(
            _req(allowed_tools=["Read", "--allow-dangerously-skip-permissions"])
        )


def test_build_invocation_rejects_bypass_permission_mode():
    p = ClaudeCliProvider(enabled=True, available=True)
    # bypassPermissions is a real claude mode but policy forbids it.
    with pytest.raises(ValueError):
        p.build_invocation(_req(permission_mode="bypassPermissions"))


# --------------------------------------------------------------------------- #
# ClaudeCliProvider.run refusal + executor wiring (never a real subprocess)
# --------------------------------------------------------------------------- #
def test_claude_run_refuses_when_not_enabled():
    called = {"n": 0}

    def _exec(spec, *, timeout_s):
        called["n"] += 1
        return AgentResult(ok=True)

    p = ClaudeCliProvider(enabled=False, available=True, executor=_exec)
    res = p.run(_req())
    assert res.ok is False
    # Executor must NOT be called when the provider is not enabled.
    assert called["n"] == 0


def test_claude_run_refuses_when_not_available():
    p = ClaudeCliProvider(enabled=True, available=False, executor=lambda s, *, timeout_s: AgentResult(ok=True))
    res = p.run(_req())
    assert res.ok is False


def test_claude_run_refuses_without_executor():
    p = ClaudeCliProvider(enabled=True, available=True)  # no executor wired
    res = p.run(_req())
    assert res.ok is False


def test_claude_run_hands_argv_list_to_executor():
    captured = {}

    def _exec(spec, *, timeout_s):
        captured["spec"] = spec
        captured["timeout"] = timeout_s
        return AgentResult(ok=True, exit_code=0, stdout_json={"result": "ok"}, session_id="s1")

    p = ClaudeCliProvider(enabled=True, available=True, executor=_exec)
    res = p.run(_req(model="m", allowed_tools=["Read"], timeout_s=42))
    assert res.ok is True
    spec = captured["spec"]
    assert isinstance(spec, ArgvSpec)
    assert isinstance(spec.argv, list)
    assert spec.argv[0] == "claude"
    assert captured["timeout"] == 42


# --------------------------------------------------------------------------- #
# build_invocation: end-of-options ("--") sentinel before the prompt
# --------------------------------------------------------------------------- #
def test_build_invocation_inserts_end_of_options_sentinel_before_prompt():
    p = ClaudeCliProvider(enabled=True, available=True)
    spec = p.build_invocation(_req(prompt="do X"))
    # The prompt is the last argv element and is immediately preceded by "--".
    assert spec.argv[-1] == "do X"
    assert spec.argv[-2] == "--"
    # Exactly one bare "--" separator (flags like --model are not bare "--").
    assert spec.argv.count("--") == 1


def test_build_invocation_flag_like_prompt_lands_after_sentinel():
    p = ClaudeCliProvider(enabled=True, available=True)
    spec = p.build_invocation(_req(prompt="--add-dir /etc/passwd"))
    sep = spec.argv.index("--")
    # A prompt that starts with a flag is parsed as the prompt: it sits AFTER
    # the "--" end-of-options separator, not as an --add-dir option.
    assert spec.argv[sep + 1] == "--add-dir /etc/passwd"
    assert spec.argv[-1] == "--add-dir /etc/passwd"
    # "--add-dir" never appears as its own argv token (which would be an option).
    assert "--add-dir" not in spec.argv


def test_build_invocation_mcp_config_prompt_is_not_parsed_as_flag():
    p = ClaudeCliProvider(enabled=True, available=True)
    spec = p.build_invocation(_req(prompt="--mcp-config evil.json"))
    sep = spec.argv.index("--")
    assert spec.argv[sep + 1] == "--mcp-config evil.json"
    assert "--mcp-config" not in spec.argv
    # The existing dangerous-flag rejection is untouched.
    with pytest.raises(DangerousFlagRequested):
        p.build_invocation(_req(permission_mode="--dangerously-skip-permissions"))


# --------------------------------------------------------------------------- #
# from_dict treats the serialized checkpoint as UNTRUSTED
# --------------------------------------------------------------------------- #
def test_tampered_applying_state_does_not_resume_in_applying():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    d = dm.to_dict()
    # Tamper: force APPLYING with confirmed=False and no confirm audit.
    d["job"]["state"] = DevState.APPLYING_APPROVED_CHANGE.value
    d["job"]["confirmed"] = False
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    assert dm2.status()["state"] != DevState.APPLYING_APPROVED_CHANGE.value
    assert dm2.status()["state"] == DevState.FAILED.value
    # And it cannot be pushed into APPLYING without a real confirm.
    with pytest.raises(DevModeError):
        dm2.advance(DevState.APPLYING_APPROVED_CHANGE)


def test_applying_with_confirmed_flag_but_no_audit_is_downgraded():
    # A checkpoint can set confirmed=True by hand; without a matching
    # request+confirm audit pair it is still refused a direct APPLYING restore.
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    d = dm.to_dict()
    d["job"]["state"] = DevState.APPLYING_APPROVED_CHANGE.value
    d["job"]["confirmed"] = True
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    assert dm2.status()["state"] == DevState.FAILED.value


def test_legit_applying_state_restores_intact():
    dm = DevelopmentMode()
    job = _drive_to_awaiting(dm)
    token = dm.request_activation(job.job_id)
    dm.confirm_activation(job.job_id, token)
    assert dm.status()["state"] == DevState.APPLYING_APPROVED_CHANGE.value
    snap = json.loads(json.dumps(dm.to_dict()))
    dm2 = DevelopmentMode.from_dict(snap)
    # A genuine confirmed activation (flag + request/confirm audit) round-trips.
    assert dm2.status()["state"] == DevState.APPLYING_APPROVED_CHANGE.value


def test_unknown_state_coerced_to_failed():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    d = dm.to_dict()
    d["job"]["state"] = "totally-made-up-state"
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    assert dm2.status()["state"] == DevState.FAILED.value
    assert dm2.status()["active"] is False


def test_naive_deadline_does_not_crash_status_or_check_timeout():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    d = dm.to_dict()
    # Tamper: strip the timezone offset -> a timezone-naive deadline. Before the
    # fix, this made check_timeout()/status() raise (naive vs aware compare),
    # silently disabling the timeout safety net.
    d["job"]["deadline_at"] = "2026-01-01T00:00:00"  # naive, in the past
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    st = dm2.status()  # must not raise
    assert st["job_id"] == d["job"]["job_id"]
    # Deadline is in the past -> the safety net still fires (no crash).
    assert dm2.check_timeout() is True
    assert dm2.status()["state"] == DevState.FAILED.value


def test_restored_develop_authorities_drop_forbidden_and_out_of_kind():
    dm = DevelopmentMode()
    dm.develop_and_test("build", owner_triggered=True)
    d = dm.to_dict()
    d["job"]["authorities"] = [
        Authority.LOCAL_CHANGES.value,   # legit for develop_and_test
        Authority.PUSH.value,            # forbidden
        Authority.MERGE.value,           # forbidden
        Authority.DEPLOY.value,          # forbidden
        Authority.RESEARCH.value,        # out-of-kind for develop_and_test
        "totally-made-up",               # unknown
    ]
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    restored = set(dm2.status()["authorities"])
    assert Authority.LOCAL_CHANGES.value in restored
    assert restored <= {a.value for a in DEVELOP_AND_TEST_AUTHORITIES}
    for a in FORBIDDEN_AUTHORITIES:
        assert a.value not in restored
        assert dm2.is_authorized(a) is False
    assert Authority.RESEARCH.value not in restored
    assert "totally-made-up" not in restored


def test_restored_research_job_cannot_gain_local_change_authority():
    dm = DevelopmentMode()
    dm.research("look", owner_triggered=True)
    d = dm.to_dict()
    d["job"]["authorities"] = [
        Authority.RESEARCH.value,
        Authority.LOCAL_CHANGES.value,       # NOT allowed for a research job
        Authority.ISOLATED_WORKSPACE.value,  # NOT allowed for a research job
        Authority.PUSH.value,                # forbidden
    ]
    dm2 = DevelopmentMode.from_dict(json.loads(json.dumps(d)))
    restored = set(dm2.status()["authorities"])
    assert restored == {Authority.RESEARCH.value}
    assert dm2.is_authorized(Authority.LOCAL_CHANGES) is False
    assert dm2.is_authorized(Authority.ISOLATED_WORKSPACE) is False
    assert dm2.is_authorized(Authority.RESEARCH) is True
