#!/usr/bin/env python3
"""
LM Studio Connectivity Diagnostic Tool

This script helps diagnose and troubleshoot LM Studio connection issues.
It tests various aspects of the LM Studio server connectivity and provides detailed feedback.
"""

import sys
import time
import os
import socket
import requests
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agentflow.engine.factory import create_llm_engine
    from agent_logging import get_logger
except ImportError as e:
    print(f"❌ Failed to import AgentFlow modules: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

logger = get_logger(__name__)

# Try to import requests, provide helpful error if not available
try:
    import requests
except ImportError:
    print("❌ 'requests' module not found. Please install it with:")
    print("   pip install requests")
    print("Or run: pip install -r agentflow/requirements.txt")
    sys.exit(1)


class LMStudioDiagnostic:
    """Comprehensive LM Studio connectivity diagnostic tool."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout
        self.results = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Log a diagnostic result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {}
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
        print()
        
    def test_socket_connection(self) -> bool:
        """Test basic socket connectivity to LM Studio server."""
        try:
            # Extract host and port from URL
            parsed_url = self.base_url.replace("http://", "").replace("https://", "").split("/")[0]
            host, port = parsed_url.split(":") if ":" in parsed_url else (parsed_url, "80")
            
            print(f"🔍 Testing socket connection to {host}:{port}")
            
            # Test socket connection
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            connection_time = time.time() - start_time
            
            if result == 0:
                self.log_result(
                    "Socket Connection Test",
                    True,
                    f"Connected successfully in {connection_time:.2f}s",
                    {"host": host, "port": port, "response_time": f"{connection_time:.2f}s"}
                )
                return True
            else:
                self.log_result(
                    "Socket Connection Test",
                    False,
                    f"Connection failed with error code {result}",
                    {"host": host, "port": port, "error_code": result}
                )
                return False
                
        except Exception as e:
            self.log_result(
                "Socket Connection Test",
                False,
                f"Exception occurred: {e}",
                {"exception_type": type(e).__name__}
            )
            return False
            
    def test_http_endpoint(self) -> bool:
        """Test HTTP connectivity to LM Studio API endpoint."""
        try:
            print(f"🌐 Testing HTTP endpoint: {self.base_url}/models")
            
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/models",
                timeout=self.timeout,
                headers={"Authorization": "Bearer lm-studio"}  # Default API key
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                models = response.json()
                model_count = len(models.get("data", []))
                self.log_result(
                    "HTTP Endpoint Test",
                    True,
                    f"API responded in {response_time:.2f}s with {model_count} models",
                    {
                        "status_code": response.status_code,
                        "response_time": f"{response_time:.2f}s",
                        "model_count": model_count,
                        "available_models": [m.get("id", "unknown") for m in models.get("data", [])[:3]]
                    }
                )
                return True
            else:
                self.log_result(
                    "HTTP Endpoint Test",
                    False,
                    f"HTTP error {response.status_code}: {response.reason}",
                    {"status_code": response.status_code, "reason": response.reason}
                )
                return False
                
        except requests.exceptions.Timeout:
            self.log_result(
                "HTTP Endpoint Test",
                False,
                f"Request timed out after {self.timeout}s",
                {"timeout": self.timeout}
            )
            return False
        except requests.exceptions.ConnectionError as e:
            self.log_result(
                "HTTP Endpoint Test",
                False,
                f"Connection error: {e}",
                {"exception_type": type(e).__name__}
            )
            return False
        except Exception as e:
            self.log_result(
                "HTTP Endpoint Test",
                False,
                f"Unexpected error: {e}",
                {"exception_type": type(e).__name__}
            )
            return False
            
    def test_agentflow_integration(self) -> bool:
        """Test AgentFlow LM Studio engine integration."""
        try:
            print("🔧 Testing AgentFlow LM Studio engine creation")
            
            start_time = time.time()
            engine = create_llm_engine(
                "lmstudio-test-model",  # Use a test model string
                base_url=self.base_url,
                connection_timeout=self.timeout,
                max_retries=1  # Don't retry for diagnostic
            )
            init_time = time.time() - start_time
            
            self.log_result(
                "AgentFlow Integration Test",
                True,
                f"Engine created successfully in {init_time:.2f}s",
                {
                    "engine_type": type(engine).__name__,
                    "model_string": engine.model_string,
                    "init_time": f"{init_time:.2f}s"
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "AgentFlow Integration Test",
                False,
                f"Engine creation failed: {e}",
                {"exception_type": type(e).__name__}
            )
            return False
            
    def test_model_availability(self, model_name: str = None) -> bool:
        """Test if a specific model is available in LM Studio."""
        try:
            print(f"🤖 Checking model availability")
            
            # Get available models
            response = requests.get(f"{self.base_url}/models", timeout=self.timeout)
            response.raise_for_status()
            models_data = response.json()
            
            available_models = [m.get("id", "") for m in models_data.get("data", [])]
            
            if model_name:
                if model_name in available_models:
                    self.log_result(
                        "Model Availability Test",
                        True,
                        f"Model '{model_name}' is available",
                        {"requested_model": model_name, "found": True}
                    )
                    return True
                else:
                    self.log_result(
                        "Model Availability Test",
                        False,
                        f"Model '{model_name}' not found",
                        {
                            "requested_model": model_name,
                            "found": False,
                            "available_models": available_models[:5]  # Show first 5
                        }
                    )
                    return False
            else:
                self.log_result(
                    "Model Availability Test",
                    True,
                    f"Found {len(available_models)} available models",
                    {"model_count": len(available_models), "sample_models": available_models[:3]}
                )
                return True
                
        except Exception as e:
            self.log_result(
                "Model Availability Test",
                False,
                f"Failed to check models: {e}",
                {"exception_type": type(e).__name__}
            )
            return False
            
    def run_full_diagnostic(self, model_name: str = None) -> Dict:
        """Run comprehensive diagnostic suite."""
        print("🚀 Starting LM Studio Connectivity Diagnostic")
        print("=" * 50)
        print(f"Target URL: {self.base_url}")
        print(f"Timeout: {self.timeout}s")
        print("=" * 50)
        print()
        
        # Run all tests
        socket_ok = self.test_socket_connection()
        http_ok = self.test_http_endpoint()
        models_ok = self.test_model_availability(model_name)
        
        # Only run integration test if basic connectivity works
        if socket_ok and http_ok:
            integration_ok = self.test_agentflow_integration()
        else:
            integration_ok = False
            self.log_result(
                "AgentFlow Integration Test",
                False,
                "Skipped due to connectivity failures",
                {"reason": "Basic connectivity tests failed"}
            )
            
        # Summary
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        
        print(f"Tests passed: {passed}/{total}")
        print()
        
        if passed == total:
            print("🎉 All tests passed! LM Studio is properly configured and accessible.")
        else:
            print("❌ Some tests failed. See details above for troubleshooting.")
            print("\n🔧 TROUBLESHOOTING RECOMMENDATIONS:")
            
            failed_tests = [r for r in self.results if not r["success"]]
            for test in failed_tests:
                if "Socket" in test["test"]:
                    print("• Check if LM Studio is running")
                    print("• Verify the port number is correct")
                    print("• Check firewall settings")
                elif "HTTP" in test["test"]:
                    print("• Ensure OpenAI-compatible server is enabled in LM Studio")
                    print("• Check server URL format")
                elif "AgentFlow" in test["test"]:
                    print("• Update AgentFlow to latest version")
                    print("• Check Python environment and dependencies")
                elif "Model" in test["test"]:
                    print("• Load a model in LM Studio")
                    print("• Check model name spelling")
                    
        return {
            "overall_success": passed == total,
            "tests_passed": passed,
            "total_tests": total,
            "results": self.results
        }


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description="LM Studio Connectivity Diagnostic Tool")
    parser.add_argument("--url", default="http://localhost:1234/v1", 
                       help="LM Studio server URL")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Connection timeout in seconds")
    parser.add_argument("--model", help="Specific model to check availability")
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = LMStudioDiagnostic(base_url=args.url, timeout=args.timeout)
    result = diagnostic.run_full_diagnostic(model_name=args.model)
    
    # Exit with appropriate code
    sys.exit(0 if result["overall_success"] else 1)


if __name__ == "__main__":
    main()
