"""
Policy engine — central evaluation point for all tool invocations.

Every tool call must pass through :func:`evaluate` before it is permitted.
The function produces a :class:`PolicyDecision` that callers are expected to
check (or call :meth:`PolicyDecision.assert_allowed`) before executing.

Policy evaluation order
-----------------------
1. ``PolicyMode.DENY``  →  deny immediately (kill-switch).
2. Classify the tool into a :class:`ToolClass`.
3. Assess risk with :mod:`jarvis.approval`.
4. For file-system operations: run :mod:`.path_guard`.
5. MCP tools: check declared capability metadata.
6. Emit final :class:`PolicyDecision` (risk handling delegated to act-then-undo model).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..debug import debug_log
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

def _classify_tool(tool_name: str, tool_args: Optional[Dict[str, Any]]) -> ToolClass:
    """Determine the :class:`ToolClass` for a given invocation.

    Delegates to the tool's own ``classify()`` method so that classification
    stays co-located with the tool definition.  MCP tools default to
    ``EXTERNAL_DELEGATED``.
    """
    if "__" in tool_name:
        # MCP tool — classified as external delegated by default
        return ToolClass.EXTERNAL_DELEGATED

    from ..tools.registry import BUILTIN_TOOLS
    tool = BUILTIN_TOOLS.get(tool_name)
    if tool is not None:
        return tool.classify(tool_args)

    return ToolClass.EXTERNAL_DELEGATED


def _legacy_to_policy_risk(risk) -> RiskLevel:
    """Passthrough — both modules now share the same :class:`RiskLevel` enum."""
    return risk


def _approval_required_for_mode(
    mode: PolicyMode, tool_class: ToolClass, risk: RiskLevel
) -> bool:
    """Always returns False — Jarvis uses act-then-undo, not blocking approval gates.

    Retained as a function to minimise churn in the evaluate() call site.
    """
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
        # Lazy import to avoid circular dependency:
        # approval -> tools.registry -> tools.builtin -> policy -> approval
        from ..approval import assess_risk as _assess_risk
        raw_risk = _assess_risk(tool_name, tool_args)
        risk = _legacy_to_policy_risk(raw_risk)

        debug_log(
            f"policy.evaluate: tool={tool_name} class={tool_class.value} risk={risk.value} mode={mode.value}",
            "policy",
        )

        # 3. Deny mode (kill-switch)
        if mode == PolicyMode.DENY:
            return PolicyDecision(
                allowed=False,
                decision_reason="Policy mode is DENY — no tool execution permitted.",
                risk_level=risk,
                tool_class=tool_class,
                applied_constraints=[AppliedConstraint("deny", "PolicyMode.DENY is active")],
                audit_id=aid,
                denied_reason="Policy mode DENY blocks all tools.",
            )

        # 4. ACTIVE mode — continue to path guard, MCP checks, etc.

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

                    tool_class=tool_class,
                    applied_constraints=constraints,
                    audit_id=aid,
                    denied_reason=str(exc),
                )

        # 6. MCP capability check
        if tool_class == ToolClass.EXTERNAL_DELEGATED:
            mcp_decision = self._evaluate_mcp_capability(tool_name, tool_args, risk)
            if mcp_decision is not None:
                # Merge MCP constraints into the running list, then rebuild
                # the decision with the combined set.
                constraints.extend(mcp_decision.applied_constraints)
                mcp_decision = PolicyDecision(
                    allowed=mcp_decision.allowed,
                    decision_reason=mcp_decision.decision_reason,
                    risk_level=risk,
                    tool_class=tool_class,
                    applied_constraints=constraints,
                    audit_id=aid,
                    denied_reason=mcp_decision.denied_reason,
                )
                if not mcp_decision.allowed:
                    return mcp_decision

        # 7. Permitted — risk handling delegated to the act-then-undo model
        reason = (
            f"Tool '{tool_name}' ({tool_class.value}) evaluated as risk={risk.value}. Permitted."
        )

        return PolicyDecision(
            allowed=True,
            decision_reason=reason,
            risk_level=risk,
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
        raw = getattr(self._cfg, "policy_mode", "active")
        normalised = str(raw).lower()
        # Accept legacy names for backwards compatibility
        if normalised in ("deny_all", "deny"):
            return PolicyMode.DENY
        # Everything else (including legacy ask_destructive / always_allow)
        # maps to ACTIVE — the act-then-undo model handles risk decisions.
        return PolicyMode.ACTIVE

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
            tool_class=_classify_tool(tool_name, tool_args),
            audit_id=audit_id or uuid.uuid4().hex,
        )
    return _default_engine.evaluate(tool_name, tool_args, audit_id=audit_id)


def get_engine() -> Optional[PolicyEngine]:
    """Return the module-level engine, or ``None`` if not yet configured."""
    return _default_engine
