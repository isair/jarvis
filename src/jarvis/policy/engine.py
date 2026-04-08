"""
Policy engine — central evaluation point for all tool invocations.

Every tool call must pass through :func:`evaluate` before it is permitted.
The function produces a :class:`PolicyDecision` that callers are expected to
check (or call :meth:`PolicyDecision.assert_allowed`) before executing.

Policy evaluation order
-----------------------
1. ``PolicyMode.DENY_ALL``  →  deny immediately.
2. Classify the tool into a :class:`ToolClass`.
3. Assess risk with :mod:`jarvis.approval`.
4. ``PolicyMode.ALWAYS_ALLOW``  →  allow with no further checks.
5. For file-system operations: run :mod:`.path_guard`.
6. MCP tools: check declared capability metadata.
7. Determine whether approval is required based on ``PolicyMode`` and ``ToolClass``.
8. Check whether a prior :class:`ScopedGrant <.approvals.ScopedGrant>` already covers this.
9. Emit final :class:`PolicyDecision`.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..debug import debug_log
from ..approval import RiskLevel as LegacyRisk, assess_risk as legacy_assess_risk
from .models import (
    AccessMode,
    AppliedConstraint,
    NetworkClass,
    PolicyDecision,
    PolicyDeniedError,
    PolicyMode,
    RiskLevel,
    ToolClass,
)
from .approvals import ApprovalStore
from .path_guard import PathGuard


# ---------------------------------------------------------------------------
# Tool class registry
# ---------------------------------------------------------------------------

#: Built-in tool name → ToolClass (operation-independent default).
_TOOL_CLASS_MAP: Dict[str, ToolClass] = {
    "screenshot":          ToolClass.INFORMATIONAL,
    "recallConversation":  ToolClass.INFORMATIONAL,
    "getWeather":          ToolClass.READ_ONLY_OPERATIONAL,
    "webSearch":           ToolClass.READ_ONLY_OPERATIONAL,
    "fetchWebPage":        ToolClass.READ_ONLY_OPERATIONAL,
    "refreshMCPTools":     ToolClass.READ_ONLY_OPERATIONAL,
    "stop":                ToolClass.INFORMATIONAL,
    "logMeal":             ToolClass.WRITE_OPERATIONAL,
    "fetchMeals":          ToolClass.INFORMATIONAL,
    "deleteMeal":          ToolClass.DESTRUCTIVE,
    "localFiles":          ToolClass.WRITE_OPERATIONAL,   # may be overridden per operation
}

#: localFiles operation → ToolClass
_LOCAL_FILES_OP_CLASS: Dict[str, ToolClass] = {
    "list":   ToolClass.INFORMATIONAL,
    "read":   ToolClass.INFORMATIONAL,
    "write":  ToolClass.WRITE_OPERATIONAL,
    "append": ToolClass.WRITE_OPERATIONAL,
    "delete": ToolClass.DESTRUCTIVE,
}


def _classify_tool(tool_name: str, tool_args: Optional[Dict[str, Any]]) -> ToolClass:
    """Determine the :class:`ToolClass` for a given invocation."""
    if "__" in tool_name:
        # MCP tool — classified as external delegated by default
        return ToolClass.EXTERNAL_DELEGATED

    if tool_name == "localFiles" and tool_args:
        op = str(tool_args.get("operation", "")).lower()
        return _LOCAL_FILES_OP_CLASS.get(op, ToolClass.WRITE_OPERATIONAL)

    return _TOOL_CLASS_MAP.get(tool_name, ToolClass.EXTERNAL_DELEGATED)


def _legacy_to_policy_risk(risk: LegacyRisk) -> RiskLevel:
    """Convert the legacy :class:`~jarvis.approval.RiskLevel` to the policy :class:`RiskLevel`."""
    mapping = {
        LegacyRisk.SAFE:     RiskLevel.SAFE,
        LegacyRisk.MODERATE: RiskLevel.MODERATE,
        LegacyRisk.HIGH:     RiskLevel.HIGH,
    }
    return mapping.get(risk, RiskLevel.MODERATE)


def _approval_required_for_mode(
    mode: PolicyMode, tool_class: ToolClass, risk: RiskLevel
) -> bool:
    """Return whether approval must be obtained given the policy mode and tool class."""
    if mode == PolicyMode.ALWAYS_ALLOW:
        return False
    if mode == PolicyMode.DENY_ALL:
        # Irrelevant — the call is denied anyway before this point.
        return False
    if mode == PolicyMode.ASK_EVERY_TIME:
        return True
    # ASK_DESTRUCTIVE (default)
    if tool_class == ToolClass.DESTRUCTIVE or risk == RiskLevel.HIGH:
        return True
    return False


# ---------------------------------------------------------------------------
# PolicyEngine class
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Stateful policy evaluator.

    Instantiate once at daemon startup and inject into the reply engine and
    any tool that needs path validation.

    Args:
        cfg: ``Settings`` object (or any object with the relevant attributes).
        approval_store: Shared :class:`ApprovalStore` instance.
    """

    def __init__(self, cfg, approval_store: Optional[ApprovalStore] = None) -> None:
        self._cfg = cfg
        self._approval_store: ApprovalStore = approval_store or ApprovalStore()
        self._path_guard = PathGuard(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        *,
        audit_id: Optional[str] = None,
    ) -> PolicyDecision:
        """
        Evaluate whether *tool_name* with *tool_args* may be executed.

        Args:
            tool_name: Canonical tool identifier (camelCase or ``server__tool``).
            tool_args: Arguments that will be passed to the tool.
            audit_id: Optional pre-assigned audit identifier.

        Returns:
            :class:`PolicyDecision` — caller must check ``allowed`` before proceeding.
        """
        aid = audit_id or uuid.uuid4().hex
        constraints: List[AppliedConstraint] = []
        mode = self._get_mode()

        # 1. Classify tool
        tool_class = _classify_tool(tool_name, tool_args)

        # 2. Assess risk (reuse existing logic for consistency)
        raw_risk = legacy_assess_risk(tool_name, tool_args)
        risk = _legacy_to_policy_risk(raw_risk)

        debug_log(
            f"policy.evaluate: tool={tool_name} class={tool_class.value} risk={risk.value} mode={mode.value}",
            "policy",
        )

        # 3. Deny-all mode
        if mode == PolicyMode.DENY_ALL:
            return PolicyDecision(
                allowed=False,
                decision_reason="Policy mode is DENY_ALL — no tool execution permitted.",
                risk_level=risk,
                approval_required=False,
                tool_class=tool_class,
                applied_constraints=[AppliedConstraint("deny_all", "PolicyMode.DENY_ALL is active")],
                audit_id=aid,
                denied_reason="Policy mode DENY_ALL blocks all tools.",
            )

        # 4. Always-allow mode (skip all remaining checks)
        if mode == PolicyMode.ALWAYS_ALLOW:
            constraints.append(AppliedConstraint("always_allow", "PolicyMode.ALWAYS_ALLOW bypasses all checks"))
            return PolicyDecision(
                allowed=True,
                decision_reason="PolicyMode.ALWAYS_ALLOW — no restrictions applied.",
                risk_level=risk,
                approval_required=False,
                tool_class=tool_class,
                applied_constraints=constraints,
                audit_id=aid,
            )

        # 5. Path guard for file-system tools
        if tool_name == "localFiles" and tool_args:
            path_str = str(tool_args.get("path", ""))
            op = str(tool_args.get("operation", "")).lower()
            access_mode = {
                "read":   AccessMode.READ,
                "list":   AccessMode.LIST,
                "write":  AccessMode.WRITE,
                "append": AccessMode.WRITE,
                "delete": AccessMode.DELETE,
            }.get(op, AccessMode.READ)

            try:
                resolved = self._path_guard.validate(path_str, access_mode)
                constraints.append(
                    AppliedConstraint(
                        "path_guard",
                        f"Path resolved and validated: {resolved}",
                    )
                )
            except PolicyDeniedError as exc:
                return PolicyDecision(
                    allowed=False,
                    decision_reason=str(exc),
                    risk_level=risk,
                    approval_required=False,
                    tool_class=tool_class,
                    applied_constraints=constraints,
                    audit_id=aid,
                    denied_reason=str(exc),
                )

        # 6. MCP capability check
        if tool_class == ToolClass.EXTERNAL_DELEGATED:
            mcp_decision = self._evaluate_mcp_capability(tool_name, tool_args, risk)
            if mcp_decision is not None:
                mcp_decision = PolicyDecision(
                    allowed=mcp_decision.allowed,
                    decision_reason=mcp_decision.decision_reason,
                    risk_level=risk,
                    approval_required=mcp_decision.approval_required,
                    tool_class=tool_class,
                    applied_constraints=constraints + mcp_decision.applied_constraints,
                    audit_id=aid,
                    denied_reason=mcp_decision.denied_reason,
                )
                if not mcp_decision.allowed:
                    return mcp_decision
                constraints.extend(mcp_decision.applied_constraints)

        # 7. Determine whether approval is required
        approval_required = _approval_required_for_mode(mode, tool_class, risk)

        # 8. Check existing grants
        if approval_required:
            op = str((tool_args or {}).get("operation", "*"))
            path = str((tool_args or {}).get("path", ""))
            if self._approval_store.is_granted(tool_name, op, path):
                approval_required = False
                constraints.append(
                    AppliedConstraint("prior_grant", "Covered by existing scoped approval grant.")
                )

        reason = (
            f"Tool '{tool_name}' ({tool_class.value}) evaluated as risk={risk.value}. "
            + ("Approval required." if approval_required else "Permitted.")
        )

        return PolicyDecision(
            allowed=True,
            decision_reason=reason,
            risk_level=risk,
            approval_required=approval_required,
            tool_class=tool_class,
            applied_constraints=constraints,
            audit_id=aid,
        )

    @property
    def approval_store(self) -> ApprovalStore:
        """Shared approval store for recording user grants."""
        return self._approval_store

    @property
    def path_guard(self) -> PathGuard:
        """Path guard instance (can be injected into tools directly)."""
        return self._path_guard

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_mode(self) -> PolicyMode:
        """Read the active PolicyMode from configuration."""
        raw = getattr(self._cfg, "policy_mode", "ask_destructive")
        try:
            return PolicyMode(str(raw).lower())
        except ValueError:
            debug_log(f"policy: unknown policy_mode '{raw}', defaulting to ask_destructive", "policy")
            return PolicyMode.ASK_DESTRUCTIVE

    def _evaluate_mcp_capability(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        risk: RiskLevel,
    ) -> Optional[PolicyDecision]:
        """
        Check declared MCP capabilities for an external-delegated tool.

        Returns a preliminary :class:`PolicyDecision` if a restriction applies,
        or ``None`` to continue normal evaluation.
        """
        mcps_config: dict = getattr(self._cfg, "mcps", {}) or {}
        if not mcps_config:
            return None

        # Derive server name from tool_name (format: server__toolname)
        server_name = tool_name.split("__")[0] if "__" in tool_name else None
        if server_name is None:
            return None

        server_cfg = mcps_config.get(server_name, {})
        capabilities: dict = server_cfg.get("capabilities", {})

        constraints: List[AppliedConstraint] = []

        if not capabilities:
            # Default to restricted when no capabilities declared
            constraints.append(
                AppliedConstraint(
                    "mcp_no_capabilities",
                    f"MCP server '{server_name}' has no capability declaration — defaulting to restricted.",
                )
            )
            # Write/destructive operations are denied without explicit capability
            if risk in (RiskLevel.MODERATE, RiskLevel.HIGH):
                return PolicyDecision(
                    allowed=False,
                    decision_reason=(
                        f"MCP server '{server_name}' lacks capability metadata. "
                        "Write/destructive operations are denied by default."
                    ),
                    risk_level=risk,
                    approval_required=False,
                    tool_class=ToolClass.EXTERNAL_DELEGATED,
                    applied_constraints=constraints,
                    audit_id=uuid.uuid4().hex,
                    denied_reason=(
                        f"MCP server '{server_name}' requires explicit 'capabilities' declaration "
                        "in config to perform write or destructive operations."
                    ),
                )
            return None  # Safe reads are allowed from undeclared servers

        cap_mode = capabilities.get("mode", "restricted")
        if cap_mode == "read_only" and tool_args:
            # Infer intent from args if available
            op = str(tool_args.get("operation", "")).lower()
            if op in ("write", "append", "delete", "create", "update", "post", "put", "patch"):
                return PolicyDecision(
                    allowed=False,
                    decision_reason=(
                        f"MCP server '{server_name}' is declared read_only but "
                        f"operation '{op}' implies a write."
                    ),
                    risk_level=risk,
                    approval_required=False,
                    tool_class=ToolClass.EXTERNAL_DELEGATED,
                    applied_constraints=constraints,
                    audit_id=uuid.uuid4().hex,
                    denied_reason=f"MCP capability mode 'read_only' blocks operation '{op}'.",
                )

        constraints.append(
            AppliedConstraint(
                "mcp_capabilities",
                f"MCP server '{server_name}' capability mode='{cap_mode}'.",
            )
        )
        return None  # Permit; caller adds constraints


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_engine: Optional[PolicyEngine] = None


def configure(cfg, approval_store: Optional[ApprovalStore] = None) -> PolicyEngine:
    """
    Initialise the module-level :class:`PolicyEngine`.

    Call once from the daemon or service container at startup.  After this,
    :func:`evaluate` can be used without passing an engine explicitly.
    """
    global _default_engine
    _default_engine = PolicyEngine(cfg, approval_store)
    debug_log("policy engine configured", "policy")
    return _default_engine


def evaluate(
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    *,
    audit_id: Optional[str] = None,
) -> PolicyDecision:
    """
    Evaluate a tool invocation against the module-level policy engine.

    Requires :func:`configure` to have been called first.  Falls back to a
    permissive decision if no engine has been configured (to avoid breaking
    existing code paths).
    """
    if _default_engine is None:
        # No engine configured — fall through with a passive allow
        debug_log(
            "policy.evaluate called before configure() — assuming permissive",
            "policy",
        )
        return PolicyDecision(
            allowed=True,
            decision_reason="Policy engine not configured — permissive default.",
            risk_level=RiskLevel.SAFE,
            approval_required=False,
            tool_class=_classify_tool(tool_name, tool_args),
            audit_id=audit_id or uuid.uuid4().hex,
        )
    return _default_engine.evaluate(tool_name, tool_args, audit_id=audit_id)


def get_engine() -> Optional[PolicyEngine]:
    """Return the module-level engine, or ``None`` if not yet configured."""
    return _default_engine
