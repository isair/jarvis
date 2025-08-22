"""
Multi-tool execution system for running multiple tools and combining their results.
"""

from __future__ import annotations
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .tools import run_tool_with_retries, ToolExecutionResult


@dataclass
class MultiToolResult:
    """Container for results from multiple tool executions."""
    tool_results: Dict[str, ToolExecutionResult]
    combined_context: str
    success_count: int
    total_count: int
    execution_summary: str


def execute_multiple_tools(
    db,
    cfg,
    tools_to_execute: List[str],
    query_analysis: Dict[str, Any],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    voice_debug: bool = False
) -> MultiToolResult:
    """
    Execute multiple tools and combine their results.
    
    Args:
        db: Database instance
        cfg: Configuration
        tools_to_execute: List of tool names to execute
        query_analysis: Analysis from query_analyzer
        system_prompt: System prompt for tools
        original_prompt: Original prompt text
        redacted_text: Redacted user text
        voice_debug: Whether to print debug info
        
    Returns:
        MultiToolResult with combined results
    """
    
    if voice_debug:
        try:
            print(f"[debug] executing {len(tools_to_execute)} tools: {tools_to_execute}", file=sys.stderr)
        except Exception:
            pass
    
    tool_results = {}
    successful_tools = []
    failed_tools = []
    combined_contexts = []
    
    # Execute each tool
    for tool_name in tools_to_execute:
        if voice_debug:
            try:
                print(f"[debug] executing tool: {tool_name}", file=sys.stderr)
            except Exception:
                pass
        
        try:
            # Prepare tool arguments based on query analysis
            tool_args = _prepare_tool_args(tool_name, query_analysis)
            
            # Execute the tool
            result = run_tool_with_retries(
                db=db,
                cfg=cfg,
                tool_name=tool_name,
                tool_args=tool_args,
                system_prompt=system_prompt,
                original_prompt=original_prompt,
                redacted_text=redacted_text,
                max_retries=1
            )
            
            tool_results[tool_name] = result
            
            if result.success and result.reply_text:
                successful_tools.append(tool_name)
                # Add tool result to combined context
                combined_contexts.append(f"[{tool_name}] {result.reply_text}")
                
                if voice_debug:
                    try:
                        print(f"[debug] {tool_name} succeeded: {len(result.reply_text)} chars", file=sys.stderr)
                    except Exception:
                        pass
            else:
                failed_tools.append(tool_name)
                if voice_debug:
                    try:
                        error_msg = result.error_message or "unknown error"
                        print(f"[debug] {tool_name} failed: {error_msg}", file=sys.stderr)
                    except Exception:
                        pass
                        
        except Exception as e:
            failed_tools.append(tool_name)
            tool_results[tool_name] = ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=str(e)
            )
            if voice_debug:
                try:
                    print(f"[debug] {tool_name} exception: {str(e)}", file=sys.stderr)
                except Exception:
                    pass
    
    # Combine all tool results into a single context
    combined_context = "\n\n".join(combined_contexts) if combined_contexts else ""
    
    # Generate execution summary
    success_count = len(successful_tools)
    total_count = len(tools_to_execute)
    
    summary_parts = []
    if successful_tools:
        summary_parts.append(f"Successfully executed: {', '.join(successful_tools)}")
    if failed_tools:
        summary_parts.append(f"Failed: {', '.join(failed_tools)}")
    
    execution_summary = "; ".join(summary_parts) if summary_parts else "No tools executed"
    
    return MultiToolResult(
        tool_results=tool_results,
        combined_context=combined_context,
        success_count=success_count,
        total_count=total_count,
        execution_summary=execution_summary
    )


def _prepare_tool_args(tool_name: str, query_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prepare arguments for a specific tool based on query analysis.
    
    Args:
        tool_name: Name of the tool to prepare args for
        query_analysis: Analysis from query_analyzer
        
    Returns:
        Dictionary of tool arguments or None
    """
    
    tool_args = {}
    
    if tool_name == "RECALL_CONVERSATION":
        # Add search query from keywords
        keywords = query_analysis.get('keywords', [])
        if keywords:
            tool_args['search_query'] = ' '.join(keywords[:3])  # Top 3 keywords
        
        # Add time range if available
        time_range = query_analysis.get('time_range')
        if time_range:
            tool_args['from'] = time_range.get('from')
            tool_args['to'] = time_range.get('to')
        
        return tool_args if tool_args else None
    
    elif tool_name == "FETCH_MEALS":
        # Use time range for meal fetching
        time_range = query_analysis.get('time_range')
        if time_range:
            return {
                'since_utc': time_range.get('from'),
                'until_utc': time_range.get('to')
            }
        return None
    
    elif tool_name == "WEB_SEARCH":
        # Create search query from keywords and intent
        keywords = query_analysis.get('keywords', [])
        intent = query_analysis.get('intent', '')
        
        # Build search query
        if intent == 'weather' and keywords:
            search_query = ' '.join(keywords) + ' current'
        elif keywords:
            search_query = ' '.join(keywords[:4])  # Top 4 keywords
        else:
            return None
            
        return {'search_query': search_query}
    
    elif tool_name == "SCREENSHOT":
        # Screenshot doesn't need arguments
        return None
    
    elif tool_name == "LOG_MEAL":
        # LOG_MEAL would need specific meal data - not suitable for auto-execution
        return None
    
    elif tool_name == "DELETE_MEAL":
        # DELETE_MEAL needs specific meal ID - not suitable for auto-execution
        return None
    
    return None


def format_multi_tool_context(
    multi_result: MultiToolResult,
    original_query: str
) -> str:
    """
    Format the results of multiple tool executions into a coherent context string.
    
    Args:
        multi_result: Results from execute_multiple_tools
        original_query: Original user query
        
    Returns:
        Formatted context string for LLM
    """
    
    if not multi_result.combined_context:
        return f"User query: {original_query}\n\nNo relevant tool results found."
    
    context_parts = [
        f"User query: {original_query}",
        "",
        "Relevant information from tools:",
        multi_result.combined_context
    ]
    
    # Add execution summary if there were any failures
    if multi_result.success_count < multi_result.total_count:
        context_parts.extend([
            "",
            f"Tool execution summary: {multi_result.execution_summary}"
        ])
    
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Test tool argument preparation
    test_analysis = {
        'intent': 'food',
        'keywords': ['eat', 'yesterday', 'meals'],
        'time_range': {
            'from': '2025-08-21T00:00:00Z',
            'to': '2025-08-21T23:59:59Z'
        },
        'suggested_tools': ['FETCH_MEALS', 'RECALL_CONVERSATION'],
        'confidence': 0.9
    }
    
    for tool in ['RECALL_CONVERSATION', 'FETCH_MEALS', 'WEB_SEARCH']:
        args = _prepare_tool_args(tool, test_analysis)
        print(f"{tool}: {args}")
