#!/usr/bin/env python3
"""
Robust AgentFlow MCP Server for LM Studio Integration

This server provides MCP (Model Context Protocol) integration for AgentFlow agents,
with enhanced stability and error handling for LM Studio compatibility.
"""

import asyncio
import json
import sys
import os
import signal
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add AgentFlow to path
sys.path.insert(0, "/Users/javi/dev/agent/AgentFlow/Untitled")

try:
    from agentflow.agentflow.solver import construct_solver
    from agentflow.agent_logging import get_logger
except ImportError as e:
    print(f"Error importing AgentFlow: {e}", file=sys.stderr)
    print("Please ensure AgentFlow is properly installed and accessible.", file=sys.stderr)
    sys.exit(1)

logger = get_logger(__name__)


class AgentFlowMCPServer:
    """MCP Server for AgentFlow agent integration with enhanced stability."""
    
    def __init__(self):
        self.solver = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the AgentFlow solver with enhanced error handling."""
        try:
            logger.info("Initializing AgentFlow solver...")
            
            # Check if LM Studio is available first
            import urllib.request
            import urllib.error
            
            try:
                url = "http://localhost:1234/v1/models"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        logger.info("LM Studio is available")
                    else:
                        logger.warning(f"LM Studio returned status {response.status}")
            except urllib.error.URLError as e:
                logger.warning(f"LM Studio not available: {e}")
                logger.warning("AgentFlow will be initialized but may not work until LM Studio is running")
            
            # Construct solver with LM Studio configuration
            self.solver = construct_solver(
                llm_engine_name="lmstudio-agentflow-planner-7b-mlx",  # Use specific model name
                enabled_tools=["Base_Generator_Tool"],
                tool_engine=["self"],
                output_types="final,direct",
                max_steps=5,
                max_time=60,
                max_tokens=1000,
                root_cache_dir="mcp_cache",
                verbose=True,
                base_url="http://localhost:1234/v1",
                temperature=0.7
            )
            self.initialized = True
            logger.info("AgentFlow MCP Server initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AgentFlow MCP Server: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.initialized = False
            return False
    
    async def solve_with_agentflow(self, question: str) -> Dict[str, Any]:
        """
        Solve a question using AgentFlow with enhanced error handling.
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Received question: {question}")
            
            # Solve using AgentFlow
            result = self.solver.solve(question)
            
            # Extract relevant information from result
            response = {
                "question": question,
                "success": True,
                "final_output": result.get("final_output"),
                "direct_output": result.get("direct_output"),
                "steps": result.get("steps", []),
                "tools_used": result.get("tools_used", []),
                "execution_time": result.get("execution_time", 0),
                "error": None
            }
            
            logger.info(f"Successfully solved question: {question}")
            return response
            
        except Exception as e:
            logger.error(f"Error solving question '{question}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "question": question,
                "success": False,
                "final_output": None,
                "direct_output": None,
                "steps": [],
                "tools_used": [],
                "execution_time": 0,
                "error": str(e)
            }
    
    async def check_agentflow_status(self) -> Dict[str, Any]:
        """
        Check the status of AgentFlow and LM Studio connection.
        """
        try:
            # Test LM Studio connection
            import urllib.request
            import urllib.error
            
            url = "http://localhost:1234/v1/models"
            req = urllib.request.Request(url)
            
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        data = response.read().decode('utf-8')
                        models = json.loads(data)
                        lmstudio_status = "connected"
                        model_count = len(models.get('data', []))
                    else:
                        lmstudio_status = "error"
                        model_count = 0
            except urllib.error.URLError:
                lmstudio_status = "disconnected"
                model_count = 0
            
            # Check AgentFlow initialization
            agentflow_status = "initialized" if self.initialized else "not_initialized"
            
            return {
                "agentflow_status": agentflow_status,
                "lmstudio_status": lmstudio_status,
                "model_count": model_count,
                "server_ready": self.initialized and lmstudio_status == "connected"
            }
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            return {
                "agentflow_status": "error",
                "lmstudio_status": "error",
                "model_count": 0,
                "server_ready": False,
                "error": str(e)
            }


class RobustMCPServer:
    """Robust MCP Server implementation for LM Studio compatibility."""
    
    def __init__(self):
        self.agentflow_server = AgentFlowMCPServer()
        self.running = True
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request with enhanced error handling."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name == "solve_with_agentflow":
                    question = arguments.get("question")
                    if not question:
                        return {
                            "error": {
                                "code": -32602,
                                "message": "Missing required parameter: question"
                            }
                        }
                    result = await self.agentflow_server.solve_with_agentflow(question)
                    return {"result": result}
                    
                elif tool_name == "check_agentflow_status":
                    result = await self.agentflow_server.check_agentflow_status()
                    return {"result": result}
                    
                else:
                    return {
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    }
            
            elif method == "tools/list":
                return {
                    "result": {
                        "tools": [
                            {
                                "name": "solve_with_agentflow",
                                "description": "Solve a question using AgentFlow agent",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": "The question or prompt to solve"
                                        }
                                    },
                                    "required": ["question"]
                                }
                            },
                            {
                                "name": "check_agentflow_status",
                                "description": "Check AgentFlow and LM Studio status",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                        ]
                    }
                }
            
            elif method == "initialize":
                # Handle MCP initialization
                await self.agentflow_server.initialize()
                return {
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "agentflow-mcp-server",
                            "version": "1.0.0"
                        }
                    }
                }
            
            else:
                return {
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": {
                    "code": -32603,
                    "message": "Internal error"
                }
            }
    
    async def run(self):
        """Run the MCP server with robust error handling."""
        logger.info("Starting Robust AgentFlow MCP Server...")
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize AgentFlow
        try:
            await self.agentflow_server.initialize()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            # Continue anyway - server can handle requests even if init fails
        
        # Main communication loop
        try:
            while self.running:
                try:
                    # Read from stdin with timeout
                    line = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                        timeout=1.0
                    )
                    
                    if not line:
                        logger.info("EOF received, shutting down...")
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        request = json.loads(line)
                        response = await self.handle_request(request)
                        
                        # Add request ID if present
                        if "id" in request:
                            response["id"] = request["id"]
                        
                        # Write response
                        print(json.dumps(response), flush=True)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        error_response = {
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            }
                        }
                        print(json.dumps(error_response), flush=True)
                        
                except asyncio.TimeoutError:
                    # Timeout is normal - just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue running even if there's an error
                    
        except KeyboardInterrupt:
            logger.info("Robust MCP Server stopped by user")
        except Exception as e:
            logger.error(f"Robust MCP Server error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logger.info("Robust MCP Server shutdown complete")


async def main():
    """Main entry point."""
    server = RobustMCPServer()
    await server.run()


if __name__ == "__main__":
    # Set up stdout buffering for immediate output
    sys.stdout.reconfigure(line_buffering=False)
    asyncio.run(main())
