# Serving Qwen-2.5-7B-Instruct Locally with vLLM

This guide explains how to serve Qwen-2.5-7B-Instruct (or other models) locally using vLLM as an alternative to using DashScope or Together AI APIs.

> **🍎 macOS Users Important Notice:** vLLM requires NVIDIA GPUs and CUDA, which are not available on macOS. For macOS, we recommend using **[LM Studio](./llm_engine.md#using-lm-studio-recommended-for-macos)** instead. See the [LM Studio setup guide](./llm_engine.md#using-lm-studio-recommended-for-macos) for macOS-compatible local inference.

---

## Prerequisites

1. **GPU**: At least one NVIDIA GPU with sufficient VRAM (recommended: 16GB+ for 7B models) **❌ Not available on macOS**
2. **vLLM installed**: Should already be installed if you ran `bash setup.sh` **❌ Not compatible with macOS**
3. **Model access**: Ensure you have access to download models from HuggingFace
4. **CUDA**: NVIDIA CUDA toolkit installed and configured **❌ Not available on macOS**

### 🍎 Alternative for macOS Users

For macOS users, we recommend using **LM Studio** instead:
- ✅ Works on Apple Silicon (M1/M2/M3) and Intel Macs
- ✅ No CUDA or NVIDIA dependencies required
- ✅ Local inference with privacy benefits
- ✅ OpenAI-compatible API

See [LM Studio Setup Guide](./llm_engine.md#using-lm-studio-recommended-for-macos) for detailed instructions.

---

## Step 1: Create a Serving Script

We recommend using the format from `scripts/serve_vllm.sh`. Create a new script for serving the Qwen model in tmux:

**Create `scripts/serve_vllm_qwen.sh`:**

```bash
#!/bin/bash

# ===========================================================================
# Script: serve_vllm_qwen.sh
# Description:
#   Launch Qwen-2.5-7B-Instruct using vLLM in a tmux window
#   - Uses GPU 0
#   - tensor-parallel-size=1
#   - Port 8001 (different from planner model on port 8000)
# ===========================================================================

MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU="0"
PORT=8001
TMUX_SESSION="vllm_qwen"
TP=1

VENV_ACTIVATE="source .venv/bin/activate"

echo "Launching model: $MODEL"
echo "  Port: $PORT"
echo "  GPU: $GPU"
echo "  Tensor Parallel Size: $TP"

# Create tmux session and run vLLM
tmux new-session -d -s "$TMUX_SESSION"

CMD_START="
    $VENV_ACTIVATE;
    export CUDA_VISIBLE_DEVICES=$GPU;
    echo '--- Starting $MODEL on port $PORT with TP=$TP ---';
    echo 'CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP
"

tmux send-keys -t "${TMUX_SESSION}:0" "$CMD_START" C-m

echo ""
echo "=== Model launched in tmux session: '$TMUX_SESSION'"
echo "=== View logs:   tmux attach-session -t $TMUX_SESSION"
echo "=== Detach:      Ctrl+B, then D"
echo "=== Kill session: tmux kill-session -t $TMUX_SESSION"
```

**Make it executable:**
```bash
chmod +x scripts/serve_vllm_qwen.sh
```

---

## Step 2: Launch the vLLM Server

Run the script to start serving the model:

```bash
bash scripts/serve_vllm_qwen.sh
```

**Monitor the server:**
- View logs: `tmux attach-session -t vllm_qwen`
- Detach from tmux: Press `Ctrl+B`, then `D`
- Kill the server: `tmux kill-session -t vllm_qwen`

**Wait for the model to load** - you should see messages like:
```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
```

---

## Step 3: Test the vLLM Server
Example for Qwen-2.5-7B-Instruct served on port 8001
### Option 1: Quick Test with curl

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7
  }'
```

### Option 2: Test with AgentFlow's Test Script

Use the existing test script [`agentflow/scripts/test_vllm.py`](../../agentflow/scripts/test_vllm.py). First, edit the script to configure your local server:

```python
# In agentflow/scripts/test_vllm.py, modify lines 7-8:
llm = ChatVLLM(
    model_string="Qwen/Qwen2.5-7B-Instruct",  # Replace YOUR_LOCAL_MODEL_NAME
    base_url="http://localhost:8001/v1/",     # Replace YOUR_PORT with 8001
    use_cache=False,
    system_prompt="You are a helpful AI assistant."
)
```

**Run the test:**
```bash
cd agentflow
python scripts/test_vllm.py
```

**Expected output:**
```
Starting vLLM Connection and Generation Tests...

--- Testing Text Generation ---

--- Test Prompt ---
[The test prompt will be displayed here]

--- Generated Response ---
[Model response will be displayed here]

--- Test Passed ---

All selected tests completed.
```

If you see `--- Test Passed ---` and a valid response from the model, your vLLM setup is working correctly!

---

## Step 4: Configure AgentFlow to Use Local vLLM

### Method 1: Modify the LLM Engine in Code

Edit `agentflow/agentflow/models/planner.py` (line 19) to use your local vLLM server:

```python
self.llm_engine_fixed = create_llm_engine(
    model_string="vllm-Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8001/v1/",
    is_multimodal=False,
    temperature=temperature
)
```

### Method 2: Using Environment Variables (if supported)

Some configurations may support environment variables. Check your specific setup.

---

## Common Issues & Troubleshooting

### Issue 1: macOS Compatibility
**Problem:** vLLM doesn't work on macOS

**Solution:** Use **LM Studio** instead:
```bash
# Install LM Studio from https://lmstudio.ai/
# Start local server and load your model
# Use model string: lmstudio-Qwen2.5-7B-Instruct
```

### Issue 2: Port Already in Use
**Error:** `Address already in use`

**Solution:** Change the port in `serve_vllm_qwen.sh`:
```bash
PORT=8002  # Use a different port
```

### Issue 3: Out of Memory (OOM)
**Error:** `CUDA out of memory`

**Solutions:**
1. Use a smaller model (e.g., Qwen2.5-3B-Instruct)
2. Enable quantization (add `--quantization awq` or `--quantization gptq` to vllm serve)
3. Reduce max model length: `--max-model-len 4096`
4. **For macOS:** Use LM Studio with quantized GGUF models

### Issue 4: Model Download Fails
**Error:** Cannot download model from HuggingFace

**Solution:** Set HuggingFace cache and token:
```bash
export HF_HOME=/path/to/cache
export HF_TOKEN=your_huggingface_token  # If accessing gated models
```

### Issue 5: Connection Refused
**Error:** `Connection refused` when testing

**Solution:**
1. Check if server is running: `tmux ls`
2. View server logs: `tmux attach-session -t vllm_qwen`
3. Verify port: `curl http://localhost:8001/health`
4. **For macOS:** Ensure LM Studio server is running on port 1234

---

## Advanced Configuration

### Using Multiple GPUs

For larger models or faster inference, use tensor parallelism:

```bash
# In serve_vllm_qwen.sh
GPU="0,1"  # Use GPU 0 and 1
TP=2       # Tensor parallel size
```

### Serving Multiple Models Simultaneously

You can serve different models on different ports:
- Port 8000: AgentFlow-Planner-7B (planner agent)
- Port 8001: Qwen2.5-7B-Instruct (executor, verifier, generator agents)

Just make sure to configure the correct `base_url` for each agent.

---

## Resource Requirements

### NVIDIA GPU Systems (vLLM)
| Model Size | Minimum VRAM | Recommended VRAM | Tensor Parallel |
|------------|--------------|------------------|-----------------|
| 7B (FP16)  | 14 GB        | 16 GB+           | 1 GPU           |
| 7B (INT8)  | 8 GB         | 12 GB+           | 1 GPU           |
| 7B (INT4)  | 4 GB         | 8 GB+            | 1 GPU           |
| 72B (FP16) | 144 GB       | 160 GB+          | 2-4 GPUs        |

### 🍎 macOS Systems (LM Studio)
| Model Size | Minimum RAM | Recommended RAM | Notes |
|------------|--------------|------------------|-------|
| 7B (Q4_K_M) | 8 GB        | 16 GB+           | Apple Silicon recommended |
| 7B (Q8_0)   | 12 GB       | 20 GB+           | Better quality, more memory |
| 3B (Q4_K_M) | 4 GB        | 8 GB+            | Good for older Macs |

---

## Summary

1. **Create serving script** based on `scripts/serve_vllm.sh`
2. **Launch vLLM server** with `bash scripts/serve_vllm_qwen.sh`
3. **Test the server** using the provided test script
4. **Configure AgentFlow** to use `vllm-<model-name>` with local `base_url`

This setup gives you full control over the model and avoids API rate limits and costs!
