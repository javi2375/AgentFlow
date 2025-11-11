# AgentFlow MCP Server with Local Models

## 🎯 **Important: Model Selection**

AgentFlow MCP server uses **whatever model is currently loaded in LM Studio**. To use specific local models:

### Required Models:
- **Planner Agent**: `agentflow-planner-7b-mlx`
- **Generalist/Executor Agent**: `demyagent-4b-qx86-hi-mlx`

## 🚀 **Setup Instructions**

### 1. Load Models in LM Studio

Before using the MCP server, you must load the desired model in LM Studio:

1. Open LM Studio
2. Go to the **Home** tab
3. Search for and load one of the required models:
   - `agentflow-planner-7b-mlx` (for planning tasks)
   - `demyagent-4b-qx86-hi-mlx` (for general tasks)
4. Click **"Start Server"** to make the model available

### 2. Configure MCP Server

Add to LM Studio's MCP configuration:

```json
{
  "mcpServers": {
    "agentflow": {
      "command": "python3",
      "args": ["/Users/javi/dev/agent/AgentFlow/Untitled/mcp_agentflow_server.py"],
      "cwd": "/Users/javi/dev/agent/AgentFlow/Untitled",
      "env": {
        "PYTHONPATH": "/Users/javi/dev/agent/AgentFlow/Untitled"
      }
    }
  }
}
```

### 3. Use the AgentFlow Tools

Once configured, you'll have two tools available in LM Studio:

#### `check_agentflow_status`
- Checks if AgentFlow is ready
- Shows LM Studio connection status
- Lists available models

#### `solve_with_agentflow`
- Solves questions using the currently loaded model
- Supports complex reasoning and tool usage
- Returns detailed step-by-step solutions

## 🔧 **How Model Selection Works**

AgentFlow automatically uses the model that's currently loaded in LM Studio:

- If `agentflow-planner-7b-mlx` is loaded → AgentFlow uses it for planning
- If `demyagent-4b-qx86-hi-mlx` is loaded → AgentFlow uses it for execution
- You can switch models in LM Studio and AgentFlow will automatically use the new one

## 📋 **Example Usage**

### Step 1: Check Status
```
Use check_agentflow_status tool
```

### Step 2: Solve a Problem
```
Use solve_with_agentflow tool with question:
"What is the capital of France and what is its population?"
```

## 🐛 **Troubleshooting**

### "No models loaded" Error
**Problem**: LM Studio shows "No models loaded"
**Solution**: 
1. Open LM Studio
2. Load a model (agentflow-planner-7b-mlx or demyagent-4b-qx86-hi-mlx)
3. Start the server
4. Try again

### Connection Issues
**Problem**: "LM Studio not available"
**Solution**:
1. Make sure LM Studio is running
2. Check that server is started on localhost:1234
3. Verify model is loaded

### Model Performance
**For best results**:
- Use `agentflow-planner-7b-mlx` for complex planning tasks
- Use `demyagent-4b-qx86-hi-mlx` for general questions and quick responses
- Both models work with the same MCP server configuration

## ✅ **Verification**

To verify everything works:

1. Load `demyagent-4b-qx86-hi-mlx` in LM Studio
2. Start LM Studio server
3. Use `check_agentflow_status` - should show "connected"
4. Use `solve_with_agentflow` with "What is 2+2?" - should return "4"

The MCP server is now fully configured for local models!