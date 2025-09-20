"""Fetch web page tool implementation for extracting content from URLs."""

import requests
from typing import Dict, Any, Optional, Callable
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def execute_fetch_web_page(
    tool_args: Optional[Dict[str, Any]],
    _user_print: callable
) -> ToolExecutionResult:
    """Fetch and extract content from a web page.
    
    Args:
        tool_args: Dictionary containing url and optional include_links parameter
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with page content
    """
    _user_print("🌐 Fetching page content…")
    try:
        if not (tool_args and isinstance(tool_args, dict)):
            return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a JSON object with 'url'.")
        
        url = str(tool_args.get("url", "")).strip()
        include_links = bool(tool_args.get("include_links", False))
        
        if not url:
            return ToolExecutionResult(success=False, reply_text="fetchWebPage requires a valid 'url'.")
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        debug_log(f"fetchWebPage: fetching {url}", "web")
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        # Parse the HTML content
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link", "noscript"]):
                script.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract main text content
            text_content = soup.get_text()
            
            # Clean up the text
            lines = []
            for line in text_content.split('\n'):
                cleaned_line = line.strip()
                if cleaned_line and len(cleaned_line) > 3:  # Filter out very short lines
                    lines.append(cleaned_line)
            
            # Remove duplicate lines and combine
            seen_lines = set()
            unique_lines = []
            for line in lines:
                if line not in seen_lines:
                    unique_lines.append(line)
                    seen_lines.add(line)
            
            content = '\n'.join(unique_lines[:500])  # Limit to 500 lines to avoid huge responses
            
            # Extract links if requested
            links_section = ""
            if include_links:
                links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '').strip()
                    link_text = link.get_text().strip()
                    if href and link_text and len(link_text) > 3:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                        elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                            continue  # Skip non-URL links
                        
                        links.append(f"• {link_text}: {href}")
                
                if links:
                    links_section = f"\n\n**Links found on page:**\n" + '\n'.join(links[:20])  # Limit to 20 links
            
            # Format the response
            reply_parts = []
            if title:
                reply_parts.append(f"**Title:** {title}")
            reply_parts.append(f"**URL:** {url}")
            reply_parts.append(f"**Content:**\n{content}")
            if links_section:
                reply_parts.append(links_section)
            
            reply_text = '\n\n'.join(reply_parts)
            
            # Limit total response size
            max_chars = 50_000
            if len(reply_text) > max_chars:
                reply_text = f"[Truncated to {max_chars} chars]\n\n" + reply_text[:max_chars]
            
            debug_log(f"fetchWebPage: extracted {len(content)} chars of content", "web")
            _user_print("✅ Page content fetched.")
            
            return ToolExecutionResult(success=True, reply_text=reply_text)
            
        except ImportError:
            # Fallback without BeautifulSoup - just return raw text
            text = response.text[:10000]  # Limit to 10k chars
            reply_text = f"**URL:** {url}\n**Raw Content:**\n{text}"
            debug_log("fetchWebPage: BeautifulSoup not available, returning raw text", "web")
            _user_print("✅ Page content fetched (raw).")
            return ToolExecutionResult(success=True, reply_text=reply_text)
            
    except requests.exceptions.RequestException as e:
        debug_log(f"fetchWebPage: request failed: {e}", "web")
        _user_print("⚠️ Failed to fetch page.")
        return ToolExecutionResult(success=False, reply_text=f"Failed to fetch page: {e}")
    except Exception as e:
        debug_log(f"fetchWebPage: error: {e}", "web")
        _user_print("⚠️ Error fetching page.")
        return ToolExecutionResult(success=False, reply_text=f"Error fetching page: {e}")


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
        """Execute the fetch web page tool."""
        return execute_fetch_web_page(args, context.user_print)