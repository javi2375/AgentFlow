#!/usr/bin/env python3
"""
Example script showing how to use LMStudio with AgentFlow.

This script demonstrates the correct way to use LMStudio with the AgentFlow framework.
The AgentFlowClient in client.py is for connecting to AgentFlow servers, not LMStudio directly.
Instead, use the solver system which integrates with LMStudio through the engine system.
"""

from agentflow.agentflow.solver import construct_solver

def main():
    """Main function demonstrating LMStudio usage."""
    
    print("🚀 Starting LMStudio integration example...")
    
    # Configure LMStudio settings
    llm_engine_name = "lmstudio"  # Use LMStudio engine
    
    # You can specify the model name as it appears in LMStudio
    # This should match the model you have loaded in LMStudio
    model_name = "Qwen2.5-7B-Instruct"  # Change this to match your LMStudio model
    
    # Optional: Configure LMStudio server URL (defaults to http://localhost:1234/v1)
    # Set this if your LMStudio is running on a different port
    lmstudio_base_url = "http://localhost:1234/v1"  # Default LMStudio port
    
    print(f"📝 Using LMStudio engine with model: {model_name}")
    print(f"🔗 Connecting to LMStudio at: {lmstudio_base_url}")
    
    try:
        # Construct solver with LMStudio engine
        solver = construct_solver(
            llm_engine_name=llm_engine_name,
            enabled_tools=["Base_Generator_Tool"],  # Start with basic tools
            output_types="final,direct",  # Get final and direct outputs
            verbose=True,
            temperature=0.7,
            base_url=lmstudio_base_url  # Pass LMStudio URL to engine
        )
        
        # Test query
        question = "What is the capital of France?"
        print(f"\n❓ Question: {question}")
        
        # Solve the query using LMStudio
        result = solver.solve(question)
        
        # Display results
        if "direct_output" in result:
            print(f"\n💬 Direct Answer: {result['direct_output']}")
        
        if "final_output" in result:
            print(f"\n🧠 Detailed Solution: {result['final_output']}")
            
        print("\n✅ LMStudio integration successful!")
        
    except Exception as e:
        print(f"\n❌ Error connecting to LMStudio: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure LMStudio is running on http://localhost:1234")
        print("2. Check that your model is loaded in LMStudio")
        print("3. Verify the model name matches what's loaded in LMStudio")
        print("4. Ensure OpenAI-compatible server is enabled in LMStudio settings")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
