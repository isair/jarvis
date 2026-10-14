"""Fetch web page tool implementation for extracting content from URLs."""

import requests
import codecs
import time

from lxml import etree
from bs4.dammit import EncodingDetector
from typing import Dict, Any, Optional
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult
from ..http_response import discard_redirect_body
from ..page_content import PageText


_MAX_FETCH_BYTES = 32 * 1024 * 1024
_FETCH_SECONDS = 15
_READ_CHUNK_BYTES = 64 * 1024


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
            url = args.get("url", "")
            if not isinstance(url, str):
                debug_log("fetchWebPage: rejected a non-string URL argument", "web")
                return ToolExecutionResult(success=False, reply_text="fetchWebPage 'url' must be a JSON string.")
            include_links = args.get("include_links", False)
            if not isinstance(include_links, bool):
                debug_log("fetchWebPage: rejected a non-boolean link flag", "web")
                return ToolExecutionResult(success=False, reply_text="fetchWebPage 'include_links' must be a JSON boolean.")
            url = url.strip()
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
            deadline = time.monotonic() + _FETCH_SECONDS
            with requests.get(url, headers=headers, timeout=_FETCH_SECONDS,
                              allow_redirects=True, stream=True,
                              hooks={"response": discard_redirect_body}) as response:
                response.raise_for_status()
                final_url = response.url if isinstance(response.url, str) else url
                page = PageText(final_url, include_links)
                content_type = response.headers.get('Content-Type', '')
                content_type = content_type if isinstance(content_type, str) else ''
                header_encoding = response.encoding if 'charset=' in content_type.casefold() else None
                def parser_for(prefix):
                    _, bom_encoding = EncodingDetector.strip_byte_order_mark(prefix)
                    encoding = (bom_encoding or header_encoding
                                or EncodingDetector.find_declared_encoding(prefix, is_html=True)
                                or 'utf-8')
                    try:
                        codecs.lookup(encoding)
                        return etree.HTMLParser(target=page, encoding=encoding, no_network=True)
                    except (LookupError, TypeError):
                        return etree.HTMLParser(target=page, encoding='utf-8', no_network=True)

                parser = None
                initial = bytearray()
                downloaded = 0
                for chunk in response.iter_content(chunk_size=_READ_CHUNK_BYTES):
                    remaining = _MAX_FETCH_BYTES-downloaded
                    if time.monotonic() >= deadline:
                        page.truncated = True
                        break
                    if not chunk:
                        continue
                    retained = chunk[:remaining]
                    if parser is None:
                        initial.extend(retained)
                        if len(initial) >= 1024:
                            prefix = bytes(initial)
                            parser = parser_for(prefix)
                            parser.feed(prefix)
                            initial.clear()
                    else:
                        parser.feed(retained)
                    downloaded += min(len(chunk), remaining)
                    if len(chunk) > remaining or page.truncated:
                        page.truncated = True
                        break
                if parser is None and initial:
                    prefix = bytes(initial)
                    parser = parser_for(prefix)
                    parser.feed(prefix)
                if parser is not None:
                    parser.close()
                if parser is not None and any(error.level >= etree.ErrorLevels.FATAL for error in parser.feed_error_log):
                    page.truncated = True
            content = page.text()
            if not content:
                if page.truncated:
                    debug_log("fetchWebPage: bounded download produced no readable excerpt", "web")
                    context.user_print("⚠️ Page exceeds the reading limit without readable content.")
                    return ToolExecutionResult(success=False, reply_text="Page could not be read within the download limit; no readable content was obtained.")
                context.user_print("⚠️ No readable page content found.")
                return ToolExecutionResult(success=False, reply_text="The page returned no readable text content.")
            reply_parts = []
            if page.title.strip():
                reply_parts.append(f"**Title:** {page.title.strip()}")
            reply_parts.extend((f"**URL:** {final_url}", f"**Content:**\n{content}"))
            if page.links:
                reply_parts.append("**Links found on page:**\n" + '\n'.join(
                    f"• {label}: {href}" for label, href in page.links))
            reply_text = '\n\n'.join(reply_parts)
            if len(reply_text) > 50_000:
                page.truncated = True
                reply_text = reply_text[:50_000]
            if page.truncated:
                label = "[Partial page excerpt: reading or content limit reached.]\n\n"
                reply_text = label + reply_text[:50_000-len(label)]
            debug_log(f"fetchWebPage: {downloaded} decoded bytes, {len(content)} text chars, partial={page.truncated}", "web")
            context.user_print("✅ Page excerpt fetched." if page.truncated else "✅ Page content fetched.")
            return ToolExecutionResult(success=True, reply_text=reply_text)
        except requests.exceptions.RequestException as e:
            debug_log(f"fetchWebPage: request failed: {e}", "web")
            context.user_print("⚠️ Failed to fetch page.")
            return ToolExecutionResult(success=False, reply_text=f"Failed to fetch page: {e}")
        except Exception as e:  # pragma: no cover (safety net)
            debug_log(f"fetchWebPage: error: {e}", "web")
            context.user_print("⚠️ Error fetching page.")
            return ToolExecutionResult(success=False, reply_text=f"Error fetching page: {e}")
