"""Web search tool implementation using DuckDuckGo."""

import requests
from typing import Dict, Any, Optional, List, Tuple
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def _fetch_page_content(url: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch and extract text content from a URL.

    Returns extracted text content, or None if fetch fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove non-content elements
        for element in soup(["script", "style", "meta", "link", "noscript", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Get text content
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 3]

        # Deduplicate consecutive identical lines
        deduped = []
        prev_line = None
        for line in lines:
            if line != prev_line:
                deduped.append(line)
                prev_line = line

        content = '\n'.join(deduped)

        # Truncate to max_chars
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        return content if content else None

    except Exception as e:
        debug_log(f"Failed to fetch page content from {url}: {e}", "web")
        return None


class WebSearchTool(Tool):
    """Tool for performing web searches using DuckDuckGo."""

    @property
    def name(self) -> str:
        return "webSearch"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo for current information, news, or general queries."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "The search query to look up"}
            },
            "required": ["search_query"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute web search using DuckDuckGo."""
        context.user_print("🌐 Searching the web…")
        cfg = context.cfg
        try:
            if not getattr(cfg, "web_search_enabled", True):
                return ToolExecutionResult(
                    success=False,
                    reply_text="Web search is currently disabled in your configuration. To enable it, set 'web_search_enabled': true in your config.json file."
                )

            search_query = ""
            if args and isinstance(args, dict):
                search_query = str(args.get("search_query", "")).strip()
            if not search_query:
                return ToolExecutionResult(success=False, reply_text="Please provide a search query for the web search.")

            debug_log(f"    🌐 searching for '{search_query}'", "web")

            # Gather instant answers
            instant_results = []
            try:
                ddg_instant_url = "https://api.duckduckgo.com/"
                ddg_instant_params = {
                    "q": search_query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                instant_response = requests.get(ddg_instant_url, params=ddg_instant_params, timeout=5)
                instant_response.raise_for_status()
                instant_data = instant_response.json()
                if instant_data.get("Abstract"):
                    instant_results.append(f"Quick Answer: {instant_data['Abstract']}")
                    if instant_data.get("AbstractURL"):
                        instant_results.append(f"  Source: {instant_data['AbstractURL']}")
                if instant_data.get("Answer"):
                    instant_results.append(f"Instant Answer: {instant_data['Answer']}")
                if instant_data.get("Definition"):
                    instant_results.append(f"Definition: {instant_data['Definition']}")
            except Exception:
                pass

            # Web search parsing
            search_results: list[str] = []
            result_urls: List[Tuple[str, str]] = []  # (title, url) pairs for auto-fetch
            try:
                import urllib.parse
                from bs4 import BeautifulSoup
                encoded_query = urllib.parse.quote_plus(search_query)
                ddg_lite_url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
                headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
                ddg_response = requests.get(ddg_lite_url, headers=headers, timeout=10)
                if ddg_response.status_code == 200:
                    soup = BeautifulSoup(ddg_response.content, 'html.parser')
                    links = soup.find_all('a', href=True)
                    result_count = 0
                    debug_log(f"Found {len(links)} total links on DDG page", "web")
                    for i, link in enumerate(links):
                        if result_count >= 5:
                            break
                        href = link.get('href', '')
                        title = link.get_text().strip()
                        if i < 10:
                            debug_log(f"Link {i}: href='{href[:50]}...', title='{title[:50]}...'", "web")
                        actual_url = href
                        if href.startswith('//duckduckgo.com/l/') and 'uddg=' in href:
                            try:
                                import urllib.parse
                                parsed = urllib.parse.urlparse(href)
                                qs = urllib.parse.parse_qs(parsed.query)
                                if 'uddg' in qs:
                                    actual_url = urllib.parse.unquote(qs['uddg'][0])
                            except Exception:
                                actual_url = href
                        if ((href.startswith('http') or href.startswith('//duckduckgo.com/l/')) and
                            len(title) > 10 and
                            not any(skip in title.lower() for skip in ['settings', 'privacy', 'about', 'help'])):
                            result_count += 1
                            search_results.append(f"{result_count}. **{title}**")
                            search_results.append(f"   Link: {actual_url}")
                            search_results.append("")
                            result_urls.append((title, actual_url))
                            debug_log(f"Accepted result {result_count}: '{title[:50]}...'", "web")
                    debug_log(f"DuckDuckGo found {result_count} results", "web")
                else:
                    debug_log(f"DuckDuckGo returned status {ddg_response.status_code}", "web")
            except ImportError:
                debug_log("BeautifulSoup not available", "web")
            except Exception as ddg_error:
                debug_log(f"DuckDuckGo search failed: {ddg_error}", "web")

            # Auto-fetch content from top results to provide actual data.
            # Cascade through up to the first 3 results: small-model replies
            # collapse to "here are some links" when the Content block is empty,
            # so we try harder to populate it. Field failures on 2026-04-20
            # showed top-1 fetches silently returning None (timeout / TLS /
            # decode) — stopping at one attempt left the reply answerless.
            fetched_content: Optional[str] = None
            fetch_attempted_any = False
            if result_urls and not instant_results:
                # Only fetch if we don't already have instant answers
                context.user_print("📄 Reading top result...")
                for attempt_idx, (attempt_title, attempt_url) in enumerate(result_urls[:3]):
                    fetch_attempted_any = True
                    debug_log(
                        f"Auto-fetching content from result #{attempt_idx + 1}: {attempt_url}",
                        "web",
                    )
                    fetched_content = _fetch_page_content(attempt_url)
                    if fetched_content:
                        debug_log(
                            f"Fetched {len(fetched_content)} chars from result #{attempt_idx + 1}",
                            "web",
                        )
                        break
                    debug_log(
                        f"Fetch failed for result #{attempt_idx + 1}, trying next",
                        "web",
                    )

            if not search_results:
                search_results.extend([
                    "🔍 **Search Information**",
                    f"   I wasn't able to find current results for '{search_query}'.",
                    "   This could be due to:",
                    "   • Search engines blocking automated requests",
                    "   • Network limitations",
                    "   • The topic requiring very recent information",
                    "",
                    "   For current information, you might try:",
                    "   • Searching manually on DuckDuckGo, Google, or Bing",
                    "   • Visiting specific websites related to your query",
                    ""
                ])

            all_results: list[str] = []
            if instant_results:
                all_results.extend(instant_results)
                all_results.append("")

            # Include fetched content from top result if available
            if fetched_content:
                all_results.append("**Content from top result:**")
                all_results.append(fetched_content)
                all_results.append("")

            if search_results:
                if instant_results or fetched_content:
                    all_results.append("**Other search results:**")
                all_results.extend(search_results)

            # Format results with explicit instruction for the LLM to use this data.
            # Small LLMs often need explicit guidance to use tool results.
            #
            # When we attempted to fetch page content but every attempt failed,
            # the payload ends up as just a link list with no facts to answer
            # from. In that case we label the envelope so the model produces an
            # honest "I couldn't read the pages" reply rather than either
            # hallucinating facts or pretending the links themselves are an
            # answer. This is the field failure mode observed 2026-04-20 on
            # 'Possessor movie': no instant answer + fetch-all-failed →
            # reply collapsed to 'Links to sources like Wikipedia'.
            if all_results:
                content_missing = (
                    fetch_attempted_any and not fetched_content and not instant_results
                )
                if content_missing:
                    envelope = (
                        f"Web search for '{search_query}' returned links but none of the top "
                        f"pages could be fetched for reading. Tell the user you couldn't read "
                        f"the page contents this time, and — if they'd find it useful — offer "
                        f"to summarise one of the following links if they pick one, or to retry. "
                        f"Do NOT invent facts about the topic from prior knowledge.\n\n"
                    )
                else:
                    envelope = (
                        f"Here are the web search results for '{search_query}'. "
                        f"Use this information to reply to the user's query:\n\n"
                    )
                reply_text = envelope + "\n".join(all_results)
            else:
                reply_text = (
                    f"The web search for '{search_query}' returned no results. "
                    f"This could be due to network issues or search service limitations. "
                    f"Let the user know you couldn't find results and suggest they try different search terms or check manually."
                )

            if getattr(cfg, "voice_debug", False):
                try:
                    instant_count = len(instant_results)
                    web_count = len([r for r in search_results if r.strip() and not r.startswith("   ")])
                    debug_log(f"      ✅ found {instant_count} instant answers, {web_count} web results", "web")
                except Exception:
                    pass
            try:
                count_results = len([r for r in (search_results or []) if r.strip() and not r.startswith("   ")])
                if count_results > 0:
                    context.user_print(f"✅ Found {count_results} results.")
                else:
                    context.user_print("⚠️ No web results found.")
            except Exception:
                pass

            return ToolExecutionResult(success=True, reply_text=reply_text)
        except Exception as search_error:
            debug_log(f"search failed: {search_error}", "web")
            return ToolExecutionResult(
                success=False,
                reply_text=f"I wasn't able to perform a web search for '{search_query}' at the moment. This could be due to network issues or search service limitations. Please try again later or search manually."
            )
        except Exception as e:  # pragma: no cover (safety net)
            debug_log(f"error {e}", "web")
            return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble performing the web search.")
