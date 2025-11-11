# LM Studio Connection Troubleshooting Guide

This guide helps diagnose and resolve common LM Studio connection issues with AgentFlow.

## 🔍 Quick Diagnosis

Run the diagnostic tool first:
```bash
python agentflow/scripts/test_lmstudio_connectivity.py
```

## 🚨 Common Issues & Solutions

### 1. Connection Timeout Errors

**Symptoms:**
- `Connection timeout` errors
- `Failed to connect to LM Studio server`
- Long hangs during initialization

**Solutions:**

#### Check LM Studio Status
- ✅ Ensure LM Studio application is running
- ✅ Verify the OpenAI-compatible server is enabled
  - In LM Studio: Settings → Server → Enable "OpenAI Compatible Server"
- ✅ Check server status in LM Studio's Server tab

#### Verify Connection Details
- ✅ Default URL: `http://localhost:1234/v1`
- ✅ Custom URL: Ensure correct port and protocol
- ✅ Check for typos in URL configuration

#### Network Issues
- ✅ Firewall not blocking port 1234
- ✅ No other applications using port 1234
- ✅ Localhost resolution working (`ping localhost`)

### 2. Model Not Found Errors

**Symptoms:**
- `Model not found` errors
- Empty model list
- Invalid model name

**Solutions:**

#### Check Model Loading
- ✅ Model is loaded in LM Studio
- ✅ Model appears in LM Studio's "Loaded Models" list
- ✅ Model name matches exactly (case-sensitive)

#### Verify Model Name
```python
# List available models
python agentflow/scripts/test_lmstudio_connectivity.py --model

# Common model name formats:
"Qwen2.5-7B-Instruct"  # As shown in LM Studio
"qwen2.5-7b-instruct"   # Lowercase version
```

### 3. Authentication Issues

**Symptoms:**
- `401 Unauthorized` errors
- API key related errors

**Solutions:**

#### LM Studio Authentication
- ✅ LM Studio typically doesn't require API keys
- ✅ Default API key: `"lm-studio"` (automatically used)
- ✅ Custom API key: Set `LMSTUDIO_API_KEY` environment variable

#### Configuration
```bash
# Set custom API key (optional)
export LMSTUDIO_API_KEY="your-custom-key"

# Set custom server URL (if not default)
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
```

### 4. Performance Issues

**Symptoms:**
- Slow responses
- Frequent timeouts
- Intermittent connection failures

**Solutions:**

#### Optimize LM Studio Settings
- ✅ Increase context limit in LM Studio
- ✅ Adjust GPU memory allocation
- ✅ Use smaller models for testing

#### AgentFlow Configuration
```python
# Increase timeout for slower models
engine = create_llm_engine(
    "lmstudio-your-model",
    connection_timeout=30.0,  # 30 seconds
    max_retries=5           # More retries
)
```

## 🛠️ Advanced Troubleshooting

### Port Scanning
Find if LM Studio is running on a different port:

```bash
# Scan common ports for LM Studio
for port in 1234 1235 8080 8000; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "LM Studio found on port $port"
    fi
done
```

### Manual Connection Test
Test LM Studio API directly:

```bash
# Test models endpoint
curl -H "Authorization: Bearer lm-studio" \
     http://localhost:1234/v1/models

# Test chat completion
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer lm-studio" \
     -d '{"model":"your-model","messages":[{"role":"user","content":"test"}]}' \
     http://localhost:1234/v1/chat/completions
```

### Log Analysis
Check LM Studio logs for errors:

1. Open LM Studio
2. Go to Settings → Logs
3. Look for error messages related to:
   - Server startup failures
   - Model loading errors
   - Network binding issues

## 🔧 Configuration Examples

### Basic Usage
```python
from agentflow.engine.factory import create_llm_engine

# Default configuration
engine = create_llm_engine("lmstudio-your-model-name")
```

### Custom Configuration
```python
# Custom server and timeout
engine = create_llm_engine(
    "lmstudio-your-model-name",
    base_url="http://localhost:1234/v1",
    connection_timeout=15.0,
    max_retries=3
)
```

### Environment Variables
```bash
# .bashrc or .zshrc
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
export LMSTUDIO_API_KEY="lm-studio"
```

## 📋 Diagnostic Checklist

Before seeking help, run through this checklist:

- [ ] LM Studio application is running
- [ ] OpenAI-compatible server is enabled in LM Studio
- [ ] Model is loaded in LM Studio
- [ ] Correct model name is being used
- [ ] Default port 1234 is available
- [ ] Firewall allows localhost connections
- [ ] Diagnostic tool runs successfully
- [ ] Can access `http://localhost:1234/v1/models` in browser
- [ ] AgentFlow is up to date

## 🆘 Getting Help

If issues persist:

1. **Run diagnostic tool and save output:**
   ```bash
   python agentflow/scripts/test_lmstudio_connectivity.py > lmstudio_diag.txt
   ```

2. **Collect system information:**
   - LM Studio version
   - Model name and size
   - Operating system
   - Python version
   - AgentFlow version

3. **Report issues with:**
   - Diagnostic output
   - LM Studio logs
   - Steps to reproduce
   - Expected vs actual behavior

## 🔄 Alternative Solutions

If LM Studio continues to have issues:

### Use Ollama (Alternative Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Use with AgentFlow
engine = create_llm_engine("ollama-llama3.2")
```

### Use Cloud Providers
```python
# OpenAI
engine = create_llm_engine("gpt-4o")

# Anthropic
engine = create_llm_engine("claude-3-5-sonnet")

# Together AI
engine = create_llm_engine("together-meta-llama/Llama-3-8b-instruct")
```

---

*Last updated: 2025-10-30*