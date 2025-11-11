#!/usr/bin/env python3
"""
Test script for the AgentFlow MCP Server with proper model configuration
"""

import subprocess
import json
import sys
import os
import time

def test_mcp_server():
    """Test the MCP server with proper stdin/stdout communication"""
    
    print("Testing AgentFlow MCP Server with agentflow-planner-7b-mlx model...")
    
    # Start the server process
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/javi/dev/agent/AgentFlow/Untitled/agentflow"
    
    # Use a simpler approach - just start the server and send one request
    cmd = [
        "source", "/Users/javi/dev/agent/AgentFlow/Untitled/agentflow/.venv/bin/activate", 
        "&&", 
        "cd", "/Users/javi/dev/agent/AgentFlow/Untitled",
        "&&",
        "python", "mcp_agentflow_server_robust.py"
    ]
    
    print(f"Starting server with command: {' '.join(cmd)}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            ' '.join(cmd),
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=0  # Unbuffered
        )
        
        # Send initialize request
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
        
        request_json = json.dumps(init_request) + "\n"
        print(f"Sending: {request_json.strip()}")
        
        # Send the request
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Wait a bit and read response
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server is still running after initialization")
            
            # Send tools/list request
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            tools_json = json.dumps(tools_request) + "\n"
            print(f"Sending: {tools_json.strip()}")
            
            process.stdin.write(tools_json)
            process.stdin.flush()
            
            # Wait for response
            time.sleep(2)
            
            # Terminate gracefully
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
                print(f"Server output: {stdout}")
                if stderr:
                    print(f"Server errors: {stderr}")
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                print("Server had to be killed forcefully")
                
        else:
            print("❌ Server exited prematurely")
            stdout, stderr = process.communicate()
            print(f"Server output: {stdout}")
            if stderr:
                print(f"Server errors: {stderr}")
                
    except Exception as e:
        print(f"Error testing server: {e}")
        if 'process' in locals():
            process.terminate()
            process.wait()

if __name__ == "__main__":
    test_mcp_server()