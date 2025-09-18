#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.jarvis.tools.registry import discover_mcp_tools
from src.jarvis.config import load_settings

cfg = load_settings()
mcps = getattr(cfg, 'mcps', {})
print(f'Found {len(mcps)} MCP servers configured: {list(mcps.keys())}')

if mcps:
    try:
        mcp_tools = discover_mcp_tools(mcps)
        print(f'Discovered {len(mcp_tools)} MCP tools:')
        for name in mcp_tools.keys():
            print(f'  - {name}')
    except Exception as e:
        print(f'Error discovering tools: {e}')
        import traceback
        traceback.print_exc()
else:
    print('No MCP servers configured')
