#!/usr/bin/env python3
"""
Test script for the AgentFlow MCP Server
"""

import json
import subprocess
import sys
import time
import os

def test_mcp_server():
    """Test the MCP server with a simple initialization sequence"""
    
    print("Starting AgentFlow MCP Server...")
    
    # Start the server process
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/javi/dev/agent/AgentFlow/Untitled/agentflow"
    
    server_process = subprocess.Popen(
        [
            "source", "/Users/javi/dev/agent/AgentFlow/Untitled/agentflow/.venv/bin/activate", 
            "&&", 
            "cd", "/Users/javi/dev/agent/AgentFlow/Untitled",
            "&&",
            "python", "mcp_agentflow_server_robust.py"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        env=env
    )
    
    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print(f"Sending initialize request: {json.dumps(init_request)}")
        
        # Send the request
        request_json = json.dumps(init_request) + "\n"
        try:
            server_process.stdin.write(request_json)
            server_process.stdin.flush()
        except BrokenPipeError:
            print("Server process closed stdin during initialization")
            return
        
        # Wait a bit for response
        time.sleep(2)
        
        # Check if process is still running before sending more
        if server_process.poll() is not None:
            print("❌ Server exited during initialization")
            stdout, stderr = server_process.communicate()
            if stdout:
                print(f"Server output: {stdout}")
            if stderr:
                print(f"Server errors: {stderr}")
            return
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        print(f"Sending initialized notification: {json.dumps(initialized_notification)}")
        
        try:
            notification_json = json.dumps(initialized_notification) + "\n"
            server_process.stdin.write(notification_json)
            server_process.stdin.flush()
        except BrokenPipeError:
            print("Server process closed stdin after initialization")
            return
        
        # Wait for server to process
        time.sleep(3)
        
        # Check if process is still running
        if server_process.poll() is None:
            print("✅ Server is still running after initialization")
            
            # Send a simple test request
            test_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            print(f"Sending tools/list request: {json.dumps(test_request)}")
            
            test_json = json.dumps(test_request) + "\n"
            server_process.stdin.write(test_json)
            server_process.stdin.flush()
            
            # Wait for response
            time.sleep(2)
            
            # Read any output
            try:
                # Check if process is still running before trying to communicate
                if server_process.poll() is None:
                    stdout, stderr = server_process.communicate(timeout=1)
                    if stdout:
                        print(f"Server output: {stdout}")
                    if stderr:
                        print(f"Server errors: {stderr}")
                else:
                    print("Server process has already terminated")
            except subprocess.TimeoutExpired:
                print("Server is running but not responding within timeout")
                server_process.terminate()
                server_process.wait()
        else:
            print("❌ Server exited prematurely")
            stdout, stderr = server_process.communicate()
            if stdout:
                print(f"Server output: {stdout}")
            if stderr:
                print(f"Server errors: {stderr}")
                
    except Exception as e:
        print(f"Error testing server: {e}")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_mcp_server()