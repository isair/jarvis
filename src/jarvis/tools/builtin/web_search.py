"""Web search tool implementation using DuckDuckGo."""

import requests
from typing import Dict, Any, Optional, Callable
from ...debug import debug_log
from ...config import Settings
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def execute_web_search(
    cfg: Settings,
    tool_args: Optional[Dict[str, Any]],
    _user_print: callable
) -> ToolExecutionResult:
    """Execute web search using DuckDuckGo.
    
    Args:
        cfg: Settings configuration object
        tool_args: Dictionary containing search_query parameter
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with search results
    """
    _user_print("🌐 Searching the web…")
    try:
        # Check if web search is enabled
        if not getattr(cfg, "web_search_enabled", True):
            return ToolExecutionResult(
                success=False, 
                reply_text="Web search is currently disabled in your configuration. To enable it, set 'web_search_enabled': true in your config.json file."
            )
        
        search_query = ""
        if tool_args and isinstance(tool_args, dict):
            search_query = str(tool_args.get("search_query", "")).strip()
        
        if not search_query:
            return ToolExecutionResult(success=False, reply_text="Please provide a search query for the web search.")
        
        debug_log(f"    🌐 searching for '{search_query}'", "web")
        
        # Use DuckDuckGo search with web scraping for comprehensive results
        try:
            # First try DuckDuckGo instant answers for quick facts
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
                
                # Extract instant answers if available
                if instant_data.get("Abstract"):
                    instant_results.append(f"Quick Answer: {instant_data['Abstract']}")
                    if instant_data.get("AbstractURL"):
                        instant_results.append(f"  Source: {instant_data['AbstractURL']}")
                
                if instant_data.get("Answer"):
                    instant_results.append(f"Instant Answer: {instant_data['Answer']}")
                
                if instant_data.get("Definition"):
                    instant_results.append(f"Definition: {instant_data['Definition']}")
            except Exception:
                pass  # Continue to web search if instant answers fail
            
            # Use DuckDuckGo search for everything - it includes Wikipedia and other sources
            search_results = []
            try:
                import urllib.parse
                from bs4 import BeautifulSoup
                
                encoded_query = urllib.parse.quote_plus(search_query)
                
                # Try DuckDuckGo lite search for reliable results
                ddg_lite_url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                
                ddg_response = requests.get(ddg_lite_url, headers=headers, timeout=10)
                
                if ddg_response.status_code == 200:
                    soup = BeautifulSoup(ddg_response.content, 'html.parser')
                    
                    # Extract search results from DuckDuckGo lite
                    links = soup.find_all('a', href=True)
                    result_count = 0
                    
                    debug_log(f"Found {len(links)} total links on DDG page", "web")
                    
                    for i, link in enumerate(links):
                        if result_count >= 5:  # Limit to top 5 results
                            break
                            
                        href = link.get('href', '')
                        title = link.get_text().strip()
                        
                        # Debug: show first few links for troubleshooting
                        if i < 10:
                            debug_log(f"Link {i}: href='{href[:50]}...', title='{title[:50]}...'", "web")
                        
                        # Extract actual URL from DuckDuckGo redirect if needed
                        actual_url = href
                        if href.startswith('//duckduckgo.com/l/') and 'uddg=' in href:
                            try:
                                import urllib.parse
                                # Extract the actual URL from the uddg parameter
                                parsed = urllib.parse.urlparse(href)
                                query_params = urllib.parse.parse_qs(parsed.query)
                                if 'uddg' in query_params:
                                    actual_url = urllib.parse.unquote(query_params['uddg'][0])
                            except Exception:
                                actual_url = href
                        
                        # Filter for actual result links (not navigation)
                        if ((href.startswith('http') or href.startswith('//duckduckgo.com/l/')) and 
                            len(title) > 10 and
                            not any(skip in title.lower() for skip in ['settings', 'privacy', 'about', 'help'])):
                            
                            result_count += 1
                            search_results.append(f"{result_count}. **{title}**")
                            search_results.append(f"   Link: {actual_url}")
                            search_results.append("")
                            
                            debug_log(f"Accepted result {result_count}: '{title[:50]}...'", "web")
                    
                    debug_log(f"DuckDuckGo found {result_count} results", "web")
                else:
                    debug_log(f"DuckDuckGo returned status {ddg_response.status_code}", "web")
            
            except ImportError:
                debug_log("BeautifulSoup not available", "web")
            except Exception as ddg_error:
                debug_log(f"DuckDuckGo search failed: {ddg_error}", "web")
            
            # No fallback - if primary search fails, the search fails
            
            # If still no results, provide helpful guidance
            if not search_results:
                search_results.append("🔍 **Search Information**")
                search_results.append(f"   I wasn't able to find current results for '{search_query}'.")
                search_results.append("   This could be due to:")
                search_results.append("   • Search engines blocking automated requests")
                search_results.append("   • Network limitations")
                search_results.append("   • The topic requiring very recent information")
                search_results.append("")
                search_results.append("   For current information, you might try:")
                search_results.append("   • Searching manually on DuckDuckGo, Google, or Bing")
                search_results.append("   • Visiting specific websites related to your query")
                search_results.append("")
            
            # Content synthesis is now handled by profile processing in daemon.py
            
            # Return raw search results for profile processing
            # Combine instant answers and search results
            all_results = []
            if instant_results:
                all_results.extend(instant_results)
                all_results.append("")  # Add spacing
            
            if search_results:
                if instant_results:
                    all_results.append("Web Search Results:")
                all_results.extend(search_results)
            
            if all_results:
                reply_text = f"Web search results for '{search_query}':\n\n" + "\n".join(all_results)
            else:
                # If no results from any method
                reply_text = f"I searched for '{search_query}' but didn't find any results. This could be due to network issues or search service limitations. Please try different search terms or check manually."
            
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
                    _user_print(f"✅ Found {count_results} results.")
                else:
                    _user_print("⚠️ No web results found.")
            except Exception:
                pass
            
            return ToolExecutionResult(success=True, reply_text=reply_text)
            
        except Exception as search_error:
            debug_log(f"search failed: {search_error}", "web")
            
            # Fallback response when search fails
            return ToolExecutionResult(
                success=False, 
                reply_text=f"I wasn't able to perform a web search for '{search_query}' at the moment. This could be due to network issues or search service limitations. Please try again later or search manually."
            )
            
    except Exception as e:
        debug_log(f"error {e}", "web")
        return ToolExecutionResult(success=False, reply_text="Sorry, I had trouble performing the web search.")


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
        """Execute the web search tool."""
        return execute_web_search(context.cfg, args, context.user_print)