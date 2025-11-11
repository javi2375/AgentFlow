#!/usr/bin/env python3
"""
Simplified MCP Server for AgentFlow integration with LM Studio.

This MCP server exposes AgentFlow's agent capabilities as MCP tools,
allowing LM Studio (and other MCP clients) to invoke the AgentFlow agent
with questions or prompts directly.
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List

# MCP imports
try:
    import mcp.types as types
except ImportError as e:
    print(f"Error: MCP packages not installed. Please install: pip install mcp")
    print(f"Details: {e}")
    sys.exit(1)

# AgentFlow imports
try:
    sys.path.insert(0, os.environ.get('AGENTFLOW_PATH', '/Users/javi/dev/agent/AgentFlow/Untitled'))
    from agentflow.agentflow.solver import construct_solver
except ImportError as e:
    print(f"Error: AgentFlow packages not found. Please install AgentFlow.")
    print(f"Details: {e}")
    sys.exit(1)


class SimpleAgentFlowMCPServer:
    """
    Simplified MCP Server for AgentFlow integration.
    """

    def __init__(self):
        """Initialize MCP server with AgentFlow integration."""
        self._setup_tools()

    def _setup_tools(self):
        """Register MCP tools that expose AgentFlow functionality."""
        # Tools are handled in the handle_request method
        pass

    async def solve_question(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """
        Handle solve_question tool calls.
        """
        try:
            # Extract question
            question = arguments.get("question", "")
            
            if not question:
                return [types.TextContent(
                    type="text",
                    text="Error: 'question' argument is required"
                )]

            # Get configuration from environment
            lmstudio_url = os.environ.get('LMSTUDIO_URL', 'http://localhost:1234/v1')
            model = os.environ.get('LMSTUDIO_MODEL', 'Qwen2.5-7B-Instruct')
            
            print(f"[AgentFlow MCP] Solving: {question[:50]}...", file=sys.stderr)
            
            # Construct solver with LM Studio configuration
            solver = construct_solver(
                llm_engine_name="lmstudio",
                enabled_tools=["all"],
                output_types="final,direct",
                max_steps=10,
                max_time=300,
                base_url=lmstudio_url,
                verbose=False
            )

            # Solve the question
            result = solver.solve(question)
            
            # Format result for display
            if isinstance(result, dict):
                response_text = json.dumps(result, indent=2)
            else:
                response_text = str(result)

            print(f"[AgentFlow MCP] Solution complete", file=sys.stderr)
            
            return [types.TextContent(
                type="text",
                text=response_text
            )]

        except Exception as e:
            error_msg = f"Error solving question: {str(e)}"
            print(f"[AgentFlow MCP] {error_msg}", file=sys.stderr)
            return [types.TextContent(
                type="text",
                text=error_msg
            )]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "solve_question":
                result = await self.solve_question(arguments)
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
                            "name": "solve_question",
                            "description": "Solve a question or problem using AgentFlow agent with LM Studio",
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
                        }
                    ]
                }
            }
        
        else:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }

    async def run(self):
        """Run the MCP server."""
        print("[AgentFlow MCP] Starting server...", file=sys.stderr)
        
        # Simple stdin/stdout communication for MCP
        try:
            while True:
                try:
                    # Read line from stdin
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                    
                    if not line:
                        print("[AgentFlow MCP] EOF received, shutting down...", file=sys.stderr)
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
                        
                        print(json.dumps(response), flush=True)
                        
                    except json.JSONDecodeError as e:
                        error_response = {
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            }
                        }
                        print(json.dumps(error_response), flush=True)
                        
                except Exception as e:
                    error_response = {
                        "error": {
                            "code": -32603,
                            "message": "Internal error"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
        except KeyboardInterrupt:
            print("[AgentFlow MCP] Server stopped by user", file=sys.stderr)
        except Exception as e:
            print(f"[AgentFlow MCP] Server error: {e}", file=sys.stderr)


async def main():
    """Main entry point for the MCP server."""
    server = SimpleAgentFlowMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[AgentFlow MCP] Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"[AgentFlow MCP] Server crashed: {e}", file=sys.stderr)
        sys.exit(1)
