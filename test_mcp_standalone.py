#!/usr/bin/env python3
"""
Test script for standalone MCP server functionality.
This tests the MCP server without LM Studio integration.
"""

import asyncio
import json
import sys
import subprocess
import time
from pathlib import Path

async def test_mcp_server():
    """Test the MCP server standalone."""
    print("Testing AgentFlow MCP Server standalone...")
    
    # Start the MCP server process
    server_process = subprocess.Popen(
        [sys.executable, "mcp_agentflow_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Give the server a moment to start
        await asyncio.sleep(2)
        
        # Test 1: Initialize request
        print("\n1. Testing initialize request...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(init_request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"Initialize response: {response}")
        
        # Test 2: Tools list request
        print("\n2. Testing tools/list request...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        server_process.stdin.write(json.dumps(tools_request) + "\n")
        server_process.stdin.flush()
        
        response_line = server_process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"Tools list response: {response}")
        
        # Test 3: Status check
        print("\n3. Testing check_agentflow_status...")
        status_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "check_agentflow_status",
                "arguments": {}
            }
        }
        
        server_process.stdin.write(json.dumps(status_request) + "\n")
        server_process.stdin.flush()
        
        response_line = server_process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"Status check response: {response}")
        
        # Test 4: Simple question (might fail if LM Studio not running)
        print("\n4. Testing solve_with_agentflow...")
        question_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "solve_with_agentflow",
                "arguments": {
                    "question": "What is 2 + 2?"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(question_request) + "\n")
        server_process.stdin.flush()
        
        # Wait a bit longer for this response as it might take time
        response_line = server_process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"Question response: {response}")
        
        print("\n✅ MCP Server test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        
        # Check for any stderr output
        stderr_output = server_process.stderr.read()
        if stderr_output:
            print(f"Server stderr: {stderr_output}")
    
    finally:
        # Clean up
        server_process.terminate()
        await asyncio.sleep(1)
        if server_process.poll() is None:
            server_process.kill()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())