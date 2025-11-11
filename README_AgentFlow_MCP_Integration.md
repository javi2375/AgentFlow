# AgentFlow MCP Integration with LM Studio

This document provides a comprehensive guide for integrating AgentFlow with LM Studio using the Model Context Protocol (MCP).

## Overview

The AgentFlow MCP server enables seamless integration between AgentFlow's agent capabilities and LM Studio's model serving interface. This allows you to:

- Use AgentFlow's advanced reasoning and tool execution directly through LM Studio
- Leverage AgentFlow's tools (search, coding, analysis) within LM Studio
- Configure models, temperature, and execution parameters
- Monitor system status and connections

## Quick Start

### Prerequisites

1. **LM Studio**: Installed and running with your preferred model loaded
2. **AgentFlow**: Installed and accessible at the configured path
3. **Node.js**: For running the MCP server (version 18+ recommended)

### Installation

1. **Build the MCP Server**:
   ```bash
   cd /Users/javi/Documents/Cline/MCP/agentflow-mcp-server
   npm install
   npm run build
   ```

2. **Configure LM Studio**:
   Add the MCP server to your LM Studio configuration:

   ```json
   {
     "mcpServers": {
       "agentflow": {
         "command": "node",
         "args": ["/Users/javi/Documents/Cline/MCP/agentflow-mcp-server/build/index.js"],
         "env": {
           "AGENTFLOW_PATH": "/Users/javi/dev/agent/AgentFlow/Untitled",
           "LMSTUDIO_URL": "http://localhost:1234/v1",
           "LMSTUDIO_MODEL": "Qwen2.5-7B-Instruct"
         }
       }
     }
   }
   ```

3. **Start Integration**:
   - Ensure LM Studio is running and serving on port 1234
   - Restart LM Studio to load the MCP server
   - Begin asking questions through LM Studio's interface

## Available Tools

### solve_with_agentflow

The main tool for solving questions and tasks using AgentFlow.

**Parameters:**
- `question` (string, required): Your question or task
- `model` (string, optional): LM Studio model to use
- `temperature` (number, optional): Generation temperature (0.0-1.0)
- `max_steps` (integer, optional): Maximum reasoning steps
- `max_time` (integer, optional): Maximum execution time in seconds
- `output_types` (string, optional): Output format (base,final,direct)
- `enabled_tools` (string, optional): Tools to enable
- `base_url` (string, optional): LM Studio server URL

**Example Usage:**
```
Ask: "What are the main differences between React and Vue.js?"
Parameters: temperature=0.7, max_steps=15, output_types=final,direct
```

### check_agentflow_status

Monitor the integration status and available capabilities.

**Parameters:**
- `check_lmstudio` (boolean): Check LM Studio connection
- `check_tools` (boolean): Check available tools

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|-----------|-------------|----------|
| `AGENTFLOW_PATH` | Path to AgentFlow installation | `/Users/javi/dev/agent/AgentFlow/Untitled` |
| `LMSTUDIO_URL` | LM Studio API endpoint | `http://localhost:1234/v1` |
| `LMSTUDIO_MODEL` | Default model to use | `local-model` |

### Tool Configuration

AgentFlow supports various tools that can be enabled/disabled:

- **Base_Generator_Tool**: General reasoning and response generation
- **Python_Coder_Tool**: Python code execution and analysis
- **Google_Search_Tool**: Web search via Google
- **Wikipedia_Search_Tool**: Wikipedia article retrieval

## Usage Examples

### Simple Question Answering
```
Question: "What is the population of Tokyo?"
Expected: Direct answer with current population data
```

### Complex Reasoning
```
Question: "Explain the process of photosynthesis and its importance to life on Earth"
Expected: Detailed explanation with multiple steps and scientific context
```

### Code Generation
```
Question: "Write a Python function that implements binary search"
Expected: Working Python code with explanations
```

### Research Tasks
```
Question: "What are the latest developments in quantum computing?"
Expected: Recent information with sources and analysis
```

## Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**
   - Ensure LM Studio is running
   - Check that the port (default: 1234) is correct
   - Verify the URL format: `http://localhost:1234/v1`

2. **AgentFlow Import Errors**
   - Verify `AGENTFLOW_PATH` is correct
   - Ensure AgentFlow is properly installed
   - Check Python path and dependencies

3. **Tool Execution Failures**
   - Confirm required tools are enabled
   - Check tool-specific configurations
   - Review error messages for specific issues

### Debug Mode

Enable verbose output to see detailed execution steps:
```json
{
  "question": "Your question here",
  "verbose": true
}
```

### Performance Optimization

- Adjust `max_steps` for complex vs simple questions
- Use appropriate `temperature` for creative vs factual tasks
- Limit `max_time` for time-sensitive applications

## Advanced Configuration

### Custom Tool Selection

Specify exactly which tools to use:
```json
{
  "enabled_tools": "Base_Generator_Tool,Python_Coder_Tool"
}
```

### Output Format Control

Choose different output formats based on your needs:
- `base`: Raw LLM response
- `final`: Detailed, formatted answer
- `direct`: Concise, direct answer
- `final,direct`: Both detailed and concise answers

### Model Selection

Use different models for different tasks:
```json
{
  "model": "Qwen2.5-7B-Instruct",
  "temperature": 0.1
}
```

## Development and Extension

### Modifying the MCP Server

1. Edit TypeScript files in `/Users/javi/Documents/Cline/MCP/agentflow-mcp-server/src/`
2. Rebuild with `npm run build`
3. Restart LM Studio to reload the server

### Adding New Tools

To extend functionality:
1. Implement the tool in AgentFlow
2. Add the tool to the MCP server's tool list
3. Update the solver configuration to include the new tool

## Support and Contributing

For issues, feature requests, or contributions:

1. Check existing documentation and troubleshooting guides
2. Review error logs for specific issues
3. Test with minimal configurations to isolate problems
4. Report issues with detailed system information

## License

This integration is provided under the MIT License. See the MCP server repository for full license details.
