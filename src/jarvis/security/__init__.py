"""Cora Security Center — local read-only monitoring (Phase 1).

This package never stops processes, never modifies the firewall/registry,
and never loads data to the cloud. Feature flag defaults to OFF.
"""

from __future__ import annotations

from jarvis.security.service import SecurityCenterService

__all__ = ["SecurityCenterService"]
