"""
🔌 Curated catalogue of popular, verified MCP servers.

Shared between the setup wizard (quick picks) and settings window (full management).
Each entry contains the config needed to add the server to config.json.

Selection criteria:
- Must NOT duplicate Jarvis built-in tools (web search, page fetch, file ops,
  memory/recall, weather, screenshot/OCR, meals).
- Wizard-featured entries must be zero-config (no API keys).
- All entries must be from the official @modelcontextprotocol org or widely trusted.
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

    def to_config(self, extra_env: Optional[Dict[str, str]] = None) -> Dict:
        """Convert to the config.json MCP entry format.

        Args:
            extra_env: Additional env vars to merge (e.g. user-supplied API keys).
                       Never mutates the entry's own env dict.
        """
        cfg: Dict = {
            "transport": "stdio",
            "command": self.command,
            "args": list(self.args),
        }
        merged_env = {**self.env, **(extra_env or {})}
        if merged_env:
            cfg["env"] = merged_env
        return cfg


# ---------------------------------------------------------------------------
# Catalogue entries — order matters for display
# ---------------------------------------------------------------------------

CATALOGUE: List[MCPEntry] = [
    # -- Wizard-featured (zero-config, genuinely novel capabilities) --
    MCPEntry(
        name="puppeteer",
        display_name="🎭 Browser Automation",
        description="Control a real browser — fill forms, click buttons, take screenshots, "
                    "and automate web tasks that simple page fetching can't handle",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        wizard_featured=True,
        category="automation",
    ),
    MCPEntry(
        name="sequential-thinking",
        display_name="🧩 Sequential Thinking",
        description="Dynamic problem-solving through structured thought chains — "
                    "helps Jarvis reason through complex multi-step questions",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
        wizard_featured=True,
        category="reasoning",
    ),

    # -- Available in settings (may need API keys or extra config) --
    MCPEntry(
        name="github",
        display_name="🐙 GitHub",
        description="Manage repositories, issues, pull requests, and code search — "
                    "your coding workflow from voice",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        needs_api_key=True,
        api_key_env_var="GITHUB_PERSONAL_ACCESS_TOKEN",
        api_key_hint="Create a token at https://github.com/settings/tokens",
        category="dev",
    ),
    MCPEntry(
        name="gitlab",
        display_name="🦊 GitLab",
        description="Manage GitLab projects, merge requests, issues, and pipelines",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-gitlab"],
        needs_api_key=True,
        api_key_env_var="GITLAB_PERSONAL_ACCESS_TOKEN",
        api_key_hint="Create a token at https://gitlab.com/-/user_settings/personal_access_tokens",
        category="dev",
    ),
    MCPEntry(
        name="google-maps",
        display_name="🗺️ Google Maps",
        description="Directions, place search, distance calculations, and geocoding — "
                    "real navigation and points of interest",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        needs_api_key=True,
        api_key_env_var="GOOGLE_MAPS_API_KEY",
        api_key_hint="Get a key at https://console.cloud.google.com/google/maps-apis",
        category="location",
    ),
    MCPEntry(
        name="slack",
        display_name="💬 Slack",
        description="Read channels, send messages, search conversations, "
                    "and manage your Slack workspace by voice",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        needs_api_key=True,
        api_key_env_var="SLACK_BOT_TOKEN",
        api_key_hint="Create a Slack app at https://api.slack.com/apps and add a Bot token",
        category="comms",
    ),
    MCPEntry(
        name="spotify",
        display_name="🎵 Spotify",
        description="Control music playback, search tracks, manage playlists, "
                    "and discover new music — all by voice",
        command="npx",
        args=["-y", "mcp-spotify"],
        needs_api_key=True,
        api_key_env_var="SPOTIFY_CLIENT_SECRET",
        api_key_hint="Create an app at https://developer.spotify.com/dashboard",
        category="media",
    ),
    MCPEntry(
        name="sqlite",
        display_name="🗄️ SQLite",
        description="Query and manage SQLite databases — run SQL, inspect schemas, "
                    "and analyse data hands-free",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sqlite"],
        category="dev",
    ),
    MCPEntry(
        name="whatsapp",
        display_name="💬 WhatsApp",
        description="Search chats, send messages, share media and voice notes — "
                    "all locally via WhatsApp Web bridge (QR code auth)",
        command="uvx",
        args=["whatsapp-mcp-server"],
        api_key_hint="Requires Go, UV, and a one-time QR code scan. "
                     "See https://github.com/lharries/whatsapp-mcp",
        category="comms",
    ),
    MCPEntry(
        name="everything",
        display_name="🔍 Everything Search",
        description="Instant file search across your entire system using Voidtools Everything "
                    "(Windows only)",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        category="files",
    ),
]

CATALOGUE_BY_NAME: Dict[str, MCPEntry] = {e.name: e for e in CATALOGUE}


def get_wizard_entries() -> List[MCPEntry]:
    """Return only entries suitable for the setup wizard (no API key needed)."""
    return [e for e in CATALOGUE if e.wizard_featured]
