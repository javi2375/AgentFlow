# AgentFlow MCP Server Integration with LM Studio

This guide explains how to integrate AgentFlow with LM Studio using the MCP (Model Context Protocol) server.

## Overview

The AgentFlow MCP server allows you to:
- Use AgentFlow agents directly through LM Studio's interface
- Ask questions and get responses from AgentFlow agents
- Check the status of AgentFlow and LM Studio connection
- Access AgentFlow's reasoning capabilities and tools

## Prerequisites

1. **LM Studio**: Download and install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. **AgentFlow**: Ensure AgentFlow is properly installed and accessible
3. **Python Environment**: Set up with required dependencies

## Setup Instructions

### 1. Start LM Studio

1. Open LM Studio
2. Load a model (e.g., any LLM model of your choice)
3. Enable the OpenAI-compatible server:
   - Go to the "Developer" tab
   - Ensure "OpenAI Compatible Server" is enabled
   - Note the server URL (default: `http://localhost:1234/v1`)

### 2. Start the AgentFlow MCP Server

```bash
# Activate the AgentFlow virtual environment
cd agentflow
source .venv/bin/activate

# Start the MCP server
python ../mcp_agentflow_server.py
```

The server will start and initialize AgentFlow with LM Studio connection.

### 3. Configure LM Studio to Use the MCP Server

In LM Studio, you can add the MCP server as a tool:

1. Go to Settings → MCP Servers
2. Add a new server with the following configuration:
   ```json
     "AgentFlow" {
     "command": "python3",
     "args": ["/Users/javi/dev/agent/AgentFlow/Untitled/mcp_agentflow_server.py"],
     "env": {
       "PYTHONPATH": "/Users/javi/dev/agent/AgentFlow/Untitled/agentflow",
        "AGENTFLOW_PATH": "/Users/javi/dev/agent/AgentFlow/Untitled",
        "LMSTUDIO_URL": "http://localhost:1234/v1",
        "LMSTUDIO_MODEL": "agentflow-planner-7b-mlx"
     }
   }
   ```

## Available Tools

The MCP server provides two main tools:

### 1. `solve_with_agentflow`

Solve a question using AgentFlow agent.

**Parameters:**
- `question` (string, required): The question or prompt to solve

**Example:**
```json
{
  "name": "solve_with_agentflow",
  "arguments": {
    "question": "What is the capital of France?"
  }
}
```

**Response:**
```json
{
  "question": "What is the capital of France?",
  "success": true,
  "final_output": "The capital of France is Paris.",
  "direct_output": "Paris",
  "steps": [...],
  "tools_used": [...],
  "execution_time": 2.5,
  "error": null
}
```

### 2. `check_agentflow_status`

Check the status of AgentFlow and LM Studio connection.

**Parameters:** None

**Response:**
```json
{
  "agentflow_status": "initialized",
  "lmstudio_status": "connected",
  "model_count": 28,
  "server_ready": true
}
```

## Usage Examples

### Basic Question Answering

1. Load a model in LM Studio
2. Start the MCP server
3. In LM Studio's chat interface, you can now use AgentFlow:
   - "Use AgentFlow to solve: What are the main causes of climate change?"
   - "Ask AgentFlow: Explain quantum computing in simple terms"

### Complex Problem Solving

AgentFlow can handle multi-step problems using its tools:

- Mathematical reasoning
- Code generation and execution
- Web search and information retrieval
- Document analysis
- Logical reasoning

## Troubleshooting

### Common Issues

1. **"No models loaded" error**
   - Solution: Load a model in LM Studio's interface
   - Go to the Home tab in LM Studio and select a model

2. **Connection timeout**
   - Ensure LM Studio's OpenAI server is enabled
   - Check that port 1234 is not blocked
   - Verify the server URL in AgentFlow configuration

3. **MCP server not found**
   - Ensure the MCP server script is in the correct location
   - Check Python path and dependencies

4. **Import errors**
   - Activate the correct virtual environment
   - Install missing dependencies with `pip install -r requirements.txt`

### Debug Mode

To enable debug logging:

```bash
export AGENTFLOW_DEBUG=1
python mcp_agentflow_server.py
```

## Configuration Options

The MCP server can be configured by modifying the `construct_solver` call in `mcp_agentflow_server.py`:

- `llm_engine_name`: Set to a specific LM Studio model name (e.g., "lmstudio-agentflow-planner-7b-mlx")
- `enabled_tools`: List of tools to enable (e.g., ["Base_Generator_Tool"])
- `tool_engine`: Set to ["self"] to use the same LM Studio model for tools
- `max_steps`: Maximum reasoning steps (default: 5)
- `max_time`: Maximum execution time in seconds (default: 60)
- `max_tokens`: Maximum tokens for responses (default: 1000)
- `temperature`: Sampling temperature (default: 0.7)

## Advanced Usage

### Custom Tools

You can extend AgentFlow with custom tools by:

1. Creating a new tool in `agentflow/agentflow/tools/`
2. Adding it to the `enabled_tools` list in the MCP server
3. Restarting the MCP server

### Multiple Models

The MCP server supports different LM Studio models. You can specify the model in the LM Studio interface, and AgentFlow will use the currently loaded model.

## Performance Tips

1. **Model Selection**: Use models appropriate for your task complexity
2. **Token Limits**: Adjust `max_tokens` based on expected response length
3. **Temperature**: Use lower temperatures (0.1-0.3) for factual answers, higher (0.7-1.0) for creative tasks
4. **Caching**: Enable caching for repeated queries

## Security Considerations

1. **Network Access**: AgentFlow tools may access external resources
2. **Code Execution**: Python code execution is sandboxed but be cautious with untrusted inputs
3. **API Keys**: Ensure LM Studio API keys are kept secure
4. **File Access**: Limit file system access as needed

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review AgentFlow logs for detailed error information
3. Ensure LM Studio is running and configured correctly
4. Verify network connectivity and firewall settings

## Updates

The MCP server is actively developed. Check for updates to:

- New tools and capabilities
- Performance improvements
- Bug fixes and security patches
- Enhanced LM Studio integration

---

**Note**: This MCP server provides a bridge between LM Studio's model serving capabilities and AgentFlow's advanced reasoning and tool-using abilities, creating a powerful AI assistant platform.
