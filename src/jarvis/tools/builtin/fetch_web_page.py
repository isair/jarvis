"""Fetch web page tool implementation for extracting content from URLs."""

import requests
from typing import Dict, Any, Optional
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class FetchWebPageTool(Tool):
    """Tool for fetching and extracting content from web pages."""

    @property
    def name(self) -> str:
        return "fetchWebPage"

    @property
    def description(self) -> str:
        return "Fetch and extract text content from a web page URL."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch content from"},
                "include_links": {"type": "boolean", "description": "Whether to include links found on the page"}
            },
            "required": ["url"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Fetch and extract content from a web page."""
        context.user_print("🌐 Fetching page content…")
        try:
            if not (args and isinstance(args, dict)):
                return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a JSON object with 'url'.")
            url = str(args.get("url", "")).strip()
            include_links = bool(args.get("include_links", False))
            if not url:
                return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a valid 'url'.")
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            debug_log(f"fetchWebPage: fetching {url}", "web")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style", "meta", "link", "noscript"]):
                    script.decompose()
                title = ""
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
                text_content = soup.get_text()
                lines = []
                for line in text_content.split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line and len(cleaned_line) > 3:
                        lines.append(cleaned_line)
                seen_lines = set()
                unique_lines = []
                for line in lines:
                    if line not in seen_lines:
                        unique_lines.append(line)
                        seen_lines.add(line)
                content = '\n'.join(unique_lines[:500])
                links_section = ""
                if include_links:
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '').strip()
                        link_text = link.get_text().strip()
                        if href and link_text and len(link_text) > 3:
                            if href.startswith('/'):
                                from urllib.parse import urljoin
                                href = urljoin(url, href)
                            elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                                continue
                            links.append(f"• {link_text}: {href}")
                    if links:
                        links_section = f"\n\n**Links found on page:**\n" + '\n'.join(links[:20])
                reply_parts = []
                if title:
                    reply_parts.append(f"**Title:** {title}")
                reply_parts.append(f"**URL:** {url}")
                reply_parts.append(f"**Content:**\n{content}")
                if links_section:
                    reply_parts.append(links_section)
                reply_text = '\n\n'.join(reply_parts)
                max_chars = 50_000
                if len(reply_text) > max_chars:
                    reply_text = f"[Truncated to {max_chars} chars]\n\n" + reply_text[:max_chars]
                debug_log(f"fetchWebPage: extracted {len(content)} chars of content", "web")
                context.user_print("✅ Page content fetched.")
                return ToolExecutionResult(success=True, reply_text=reply_text)
            except ImportError:
                text = response.text[:10000]
                reply_text = f"**URL:** {url}\n**Raw Content:**\n{text}"
                debug_log("fetchWebPage: BeautifulSoup not available, returning raw text", "web")
                context.user_print("✅ Page content fetched (raw).")
                return ToolExecutionResult(success=True, reply_text=reply_text)
        except requests.exceptions.RequestException as e:
            debug_log(f"fetchWebPage: request failed: {e}", "web")
            context.user_print("⚠️ Failed to fetch page.")
            return ToolExecutionResult(success=False, reply_text=f"Failed to fetch page: {e}")
        except Exception as e:  # pragma: no cover (safety net)
            debug_log(f"fetchWebPage: error: {e}", "web")
            context.user_print("⚠️ Error fetching page.")
            return ToolExecutionResult(success=False, reply_text=f"Error fetching page: {e}")
