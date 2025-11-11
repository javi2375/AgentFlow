#!/usr/bin/env python3
"""
Test script for AgentFlow MCP Server integration with LM Studio.

This script tests the MCP server functionality to ensure it works correctly
with AgentFlow and LM Studio integration.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add AgentFlow to path
sys.path.insert(0, "/Users/javi/dev/agent/AgentFlow/Untitled")

try:
    from agentflow.agentflow.solver import construct_solver
    from agentflow.agent_logging import get_logger
except ImportError as e:
    print(f"Error importing AgentFlow: {e}")
    print("Please ensure AgentFlow is properly installed and accessible.")
    sys.exit(1)

logger = get_logger(__name__)


async def test_agentflow_solver():
    """Test the AgentFlow solver directly."""
    print("🧪 Testing AgentFlow Solver...")
    
    try:
        # Construct solver with LM Studio configuration
        solver = construct_solver(
            llm_engine_name="lmstudio",
            enabled_tools=["Base_Generator_Tool"],
            tool_engine=["Default"],
            output_types="final,direct",
            max_steps=5,
            max_time=60,
            max_tokens=1000,
            root_cache_dir="test_cache",
            verbose=True,
            base_url="http://localhost:1234/v1",
            temperature=0.7
        )
        
        # Test with a simple question
        test_question = "What is the capital of France?"
        print(f"📝 Question: {test_question}")
        
        result = solver.solve(test_question)
        
        # Check result
        if result and (result.get("final_output") or result.get("direct_output")):
            print("✅ AgentFlow solver test PASSED")
            print(f"📋 Final Output: {result.get('final_output', 'N/A')}")
            print(f"🎯 Direct Output: {result.get('direct_output', 'N/A')}")
            return True
        else:
            print("❌ AgentFlow solver test FAILED - No valid output")
            print(f"🔍 Result: {json.dumps(result, indent=2)}")
            return False
            
    except Exception as e:
        print(f"❌ AgentFlow solver test FAILED with error: {e}")
        return False


async def test_lmstudio_connection():
    """Test LM Studio connection."""
    print("🔗 Testing LM Studio Connection...")
    
    try:
        import urllib.request
        import urllib.error
        import json
        
        url = "http://localhost:1234/v1/models"
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = response.read().decode('utf-8')
                    models = json.loads(data)
                    print("✅ LM Studio connection test PASSED")
                    print(f"📊 Available models: {len(models.get('data', []))}")
                    return True
                else:
                    print(f"❌ LM Studio connection test FAILED - HTTP {response.status}")
                    return False
        except urllib.error.URLError as e:
            print(f"❌ LM Studio connection test FAILED with error: {e}")
            print("💡 Make sure LM Studio is running on localhost:1234")
            return False
            
    except Exception as e:
        print(f"❌ LM Studio connection test FAILED with error: {e}")
        print("💡 Make sure LM Studio is running on localhost:1234")
        return False


async def test_mcp_server():
    """Test MCP server functionality."""
    print("🔧 Testing MCP Server...")
    
    try:
        # Check if MCP server files exist
        mcp_server_path = Path("/Users/javi/Documents/Cline/MCP/agentflow-mcp-server/build/index.js")
        if mcp_server_path.exists():
            print("✅ MCP server build exists")
            return True
        else:
            print("❌ MCP server build not found")
            print("💡 Run 'npm run build' in the MCP server directory")
            return False
            
    except Exception as e:
        print(f"❌ MCP server test FAILED with error: {e}")
        return False


async def test_python_mcp_server():
    """Test Python MCP server implementation."""
    print("🐍 Testing Python MCP Server...")
    
    try:
        # Check if Python MCP server files exist
        mcp_server_path = Path("mcp_agentflow_server.py")
        if mcp_server_path.exists():
            print("✅ Python MCP server file exists")
            
            # Try to import the server
            sys.path.insert(0, ".")
            try:
                import mcp_agentflow_server
                print("✅ Python MCP server can be imported")
                return True
            except ImportError as e:
                print(f"❌ Python MCP server import failed: {e}")
                return False
        else:
            print("❌ Python MCP server file not found")
            return False
            
    except Exception as e:
        print(f"❌ Python MCP server test FAILED with error: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting AgentFlow MCP Integration Tests\n")
    
    tests = [
        ("LM Studio Connection", test_lmstudio_connection),
        ("AgentFlow Solver", test_agentflow_solver),
        ("Python MCP Server", test_python_mcp_server),
        ("MCP Server Build", test_mcp_server),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Test: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AgentFlow MCP integration is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


def main():
    """Main entry point."""
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
