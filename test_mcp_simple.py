#!/usr/bin/env python3
"""
Simple test script for the AgentFlow MCP Server
"""

import json
import sys
import os

def test_mcp_protocol():
    """Test MCP protocol by sending a simple initialize request"""
    
    print("Testing MCP protocol...")
    
    # Read from stdin if available, otherwise simulate
    if not sys.stdin.isatty():
        print("Reading from stdin...")
        for line in sys.stdin:
            try:
                data = json.loads(line.strip())
                print(f"Received: {json.dumps(data, indent=2)}")
                
                # Send response
                if data.get("method") == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {
                                    "listChanged": True
                                }
                            },
                            "serverInfo": {
                                "name": "agentflow-mcp-server",
                                "version": "1.0.0"
                            }
                        }
                    }
                    print(f"Sending response: {json.dumps(response)}")
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}")
    else:
        print("No stdin input, simulating MCP test...")
        
        # Test initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        print(f"Test request: {json.dumps(init_request, indent=2)}")

if __name__ == "__main__":
    test_mcp_protocol()