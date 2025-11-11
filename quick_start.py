# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name
llm_engine_name = "lmstudio-agentflow-planner-7b-mlx" # you can use "gpt-4o" as well
# llm_engine_name = "gpt-4o"

# Construct the solver
# Use "self" to make tools use the same LM Studio model as the main engine
# We need to specify which tools to enable and provide matching tool_engine parameters
solver = construct_solver(
    llm_engine_name=llm_engine_name,
    enabled_tools=["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool"],
    tool_engine=["self", "self", "self"]
)

# Solve the user query
output = solver.solve("what is the latest model, in 2025, that zhipu AI has released?")
print(output["direct_output"])