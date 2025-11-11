# LM Studio + AgentFlow MCP Integration Guide

This guide shows how to integrate the AgentFlow MCP server with LM Studio, allowing you to use AgentFlow's advanced reasoning capabilities directly from LM Studio's interface.

## 🚀 Quick Integration

### Step 1: Install Dependencies

First, install the required MCP package for AgentFlow:

```bash
pip install mcp
```

### Step 2: Configure LM Studio

Add the AgentFlow MCP server to LM Studio's MCP settings:

1. Open LM Studio
2. Go to Settings → MCP Servers
3. Click "Add Server"
4. Configure with these settings:

| Setting | Value |
|---------|-------|
| **Name** | AgentFlow MCP |
| **Command** | `python3` |
| **Arguments** | `/Users/javi/dev/agent/AgentFlow/Untitled/mcp_agentflow_server_simple.py` |
| **Environment** | <details> |

Set these environment variables:
```
AGENTFLOW_PATH=/Users/javi/dev/agent/AgentFlow/Untitled
LMSTUDIO_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=agentflow-planner-7b-mlx
```

### Step 3: Start Using

Once configured, you can use AgentFlow directly in LM Studio:

## 📖 Usage Examples

### Simple Question
Type in LM Studio:
```
What is the capital of France?
```

### Complex Problem
```
Write a Python function that calculates the nth Fibonacci number efficiently
```

### Research Task
```
What are the latest developments in quantum computing?
```

### Coding Task
```
Create a web scraper for extracting product information from e-commerce sites
```

## 🛠️ Available Tools

The AgentFlow MCP server exposes these tools through LM Studio:

1. **Base_Generator_Tool** - Generate responses and analyze problems
2. **Python_Coder_Tool** - Execute Python code and scripts  
3. **Google_Search_Tool** - Search the web for current information
4. **Wikipedia_Search_Tool** - Access Wikipedia knowledge base

## 🔧 Configuration Options

You can customize AgentFlow behavior by modifying the tool call:

### Enable Specific Tools
```json
{
  "enabled_tools": "Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool"
}
```

### Output Types
```json
{
  "output_types": "final,direct"  // Get both reasoning and final answer
}
```

### Model Selection
```json
{
  "model": "your-preferred-model-name"
}
```

### Temperature Control
```json
{
  "temperature": 0.3  // More conservative responses
}
```

## 🔍 Verification

To test the integration:

1. Open LM Studio
2. Check that "AgentFlow MCP" server appears in the server list
3. Try a simple question like "What is 2+2?"
4. Verify you get a detailed response

## 🎯 Benefits

### For LM Studio Users
- **Advanced Reasoning**: Get AgentFlow's sophisticated multi-step problem solving
- **Tool Integration**: Access web search, coding, and research tools
- **Local Models**: Use your local LM Studio models with AgentFlow's capabilities
- **Memory**: Persistent context across conversations

### For AgentFlow Users
- **Standard Interface**: MCP protocol compatibility
- **Broader Reach**: Access from any MCP-compatible client
- **Unified Workflow**: Seamless integration between local models and advanced agent capabilities

## 📞 Troubleshooting

### Common Issues

1. **"AgentFlow packages not found"**
   - Ensure AgentFlow is installed: `/Users/javi/dev/agent/AgentFlow/Untitled`
   - Check AGENTFLOW_PATH environment variable

2. **"MCP packages not installed"**
   - Install: `pip install mcp`

3. **"LM Studio connection failed"**
   - Ensure LM Studio is running with OpenAI-compatible server
   - Check LMSTUDIO_URL environment variable

4. **"Tool execution failed"**
   - Verify AgentFlow tools are properly installed
   - Check enabled_tools configuration

5. **"Question argument required"**
   - Ensure your question is properly formatted in the tool call

### Debug Mode

Enable verbose logging by setting:
```json
{
  "verbose": true
}
```

## 📁 Files Created

- `mcp_agentflow_server_simple.py` - The main MCP server file
- `test_agentflow_mcp.py` - Test script for verification
- `README_AgentFlow_MCP_Integration.md` - Complete documentation
- `mcp_agentflow_requirements.txt` - Dependencies list

---

**This integration bridges AgentFlow's advanced agent capabilities with LM Studio's user-friendly interface, providing the best of both worlds!**
