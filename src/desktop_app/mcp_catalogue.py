"""
🔌 Curated catalogue of popular, verified MCP servers.

Shared between the setup wizard (quick picks) and settings window (full management).
Each entry contains the config needed to add the server to config.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MCPEntry:
    """A curated MCP server entry."""
    name: str               # Config key / server name
    display_name: str       # Human-readable name
    description: str        # Short description of what it does
    command: str            # Executable (e.g. "npx")
    args: List[str]         # Command arguments
    env: Dict[str, str] = field(default_factory=dict)
    needs_api_key: bool = False        # Requires user to supply an API key
    api_key_env_var: Optional[str] = None  # Which env var holds the key
    api_key_hint: Optional[str] = None     # Help text for obtaining the key
    wizard_featured: bool = False      # Show in setup wizard quick picks
    category: str = "general"          # Grouping for display

    def to_config(self) -> Dict:
        """Convert to the config.json MCP entry format."""
        cfg: Dict = {
            "transport": "stdio",
            "command": self.command,
            "args": list(self.args),
        }
        if self.env:
            cfg["env"] = dict(self.env)
        return cfg


# ---------------------------------------------------------------------------
# Catalogue entries — order matters for display
# ---------------------------------------------------------------------------

CATALOGUE: List[MCPEntry] = [
    # -- Wizard-featured (safe, no API key needed) --
    MCPEntry(
        name="filesystem",
        display_name="📁 Filesystem",
        description="Read, write, and search files in specified directories",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "~"],
        wizard_featured=True,
        category="files",
    ),
    MCPEntry(
        name="memory",
        display_name="🧠 Memory",
        description="Persistent key-value memory store across conversations",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        wizard_featured=True,
        category="knowledge",
    ),
    MCPEntry(
        name="fetch",
        display_name="🌐 Fetch",
        description="Retrieve and convert web pages and APIs to text",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-fetch"],
        wizard_featured=True,
        category="web",
    ),

    # -- Available in settings (may need API keys) --
    MCPEntry(
        name="brave-search",
        display_name="🔍 Brave Search",
        description="Web and local search via the Brave Search API",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        needs_api_key=True,
        api_key_env_var="BRAVE_API_KEY",
        api_key_hint="Get a free key at https://brave.com/search/api/",
        category="web",
    ),
    MCPEntry(
        name="github",
        display_name="🐙 GitHub",
        description="Manage repositories, issues, pull requests, and more",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        needs_api_key=True,
        api_key_env_var="GITHUB_PERSONAL_ACCESS_TOKEN",
        api_key_hint="Create a token at https://github.com/settings/tokens",
        category="dev",
    ),
    MCPEntry(
        name="sqlite",
        display_name="🗄️ SQLite",
        description="Query and manage SQLite databases",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sqlite"],
        category="dev",
    ),
    MCPEntry(
        name="puppeteer",
        display_name="🎭 Puppeteer",
        description="Browser automation — navigate, screenshot, and interact with web pages",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        category="web",
    ),
]

CATALOGUE_BY_NAME: Dict[str, MCPEntry] = {e.name: e for e in CATALOGUE}


def get_wizard_entries() -> List[MCPEntry]:
    """Return only entries suitable for the setup wizard (no API key needed)."""
    return [e for e in CATALOGUE if e.wizard_featured]
