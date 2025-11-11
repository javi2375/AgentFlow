#!/usr/bin/env python3
"""
Direct test of MCP server functionality without subprocess.
"""

import asyncio
import json
import sys
from io import StringIO

# Add AgentFlow to path
sys.path.insert(0, "/Users/javi/dev/agent/AgentFlow/Untitled")

from mcp_agentflow_server import MCPServer

async def test_mcp_server_direct():
    """Test the MCP server directly."""
    print("Testing AgentFlow MCP Server directly...")
    
    server = MCPServer()
    
    try:
        # Test 1: Initialize
        print("\n1. Testing initialization...")
        init_success = await server.agentflow_server.initialize()
        print(f"Initialization result: {init_success}")
        
        # Test 2: Tools list
        print("\n2. Testing tools/list...")
        tools_request = {
            "method": "tools/list"
        }
        tools_response = await server.handle_request(tools_request)
        print(f"Tools list response: {json.dumps(tools_response, indent=2)}")
        
        # Test 3: Status check
        print("\n3. Testing check_agentflow_status...")
        status_request = {
            "method": "tools/call",
            "params": {
                "name": "check_agentflow_status",
                "arguments": {}
            }
        }
        status_response = await server.handle_request(status_request)
        print(f"Status check response: {json.dumps(status_response, indent=2)}")
        
        # Test 4: Simple question (might fail if LM Studio not running)
        print("\n4. Testing solve_with_agentflow...")
        question_request = {
            "method": "tools/call",
            "params": {
                "name": "solve_with_agentflow",
                "arguments": {
                    "question": "What is 2 + 2?"
                }
            }
        }
        question_response = await server.handle_request(question_request)
        print(f"Question response: {json.dumps(question_response, indent=2)}")
        
        # Test 5: Error handling
        print("\n5. Testing error handling...")
        error_request = {
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {}
            }
        }
        error_response = await server.handle_request(error_request)
        print(f"Error response: {json.dumps(error_response, indent=2)}")
        
        print("\n✅ MCP Server direct test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_server_direct())