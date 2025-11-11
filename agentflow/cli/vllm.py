from typing import List

try:
    from vllm.entrypoints.cli.main import main
    from agentflow.instrumentation.vllm import instrument_vllm
    VLLM_AVAILABLE = True
except ImportError:
    print("Error: vllm is not installed. Please install it with 'pip install vllm'")
    print("This script is used to run vLLM with AgentFlow instrumentation.")
    VLLM_AVAILABLE = False


if __name__ == "__main__":
    if not VLLM_AVAILABLE:
        exit(1)
    
    instrument_vllm()
    main()
