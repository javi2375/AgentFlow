#!/bin/bash

# ===========================================================================
# Script: serve_vllm.sh
# Description:
#   Launch model using vLLM in a tmux window
#   - Uses GPU 0
#   - tensor-parallel-size=1
#   - Port 8000
# ===========================================================================

# Check for macOS and CUDA availability
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "❌ vLLM requires NVIDIA GPU and CUDA, which are not available on macOS."
    echo ""
    echo "🍎 For macOS, we recommend using LM Studio instead:"
    echo "   1. Install LM Studio from https://lmstudio.ai/"
    echo "   2. Download a model like 'Qwen2.5-7B-Instruct'"
    echo "   3. Start LM Studio server (default port: 1234)"
    echo "   4. Use 'lmstudio-' prefix in your model configurations"
    echo ""
    echo "📚 See assets/doc/llm_engine.md for LM Studio setup instructions."
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install NVIDIA CUDA toolkit."
    echo "   Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

MODEL="AgentFlow/agentflow-planner-7b"
GPU="0"
PORT=8000
TMUX_SESSION="vllm_agentflow"
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
    echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP
"

tmux send-keys -t "${TMUX_SESSION}:0" "$CMD_START" C-m

echo ""
echo "✅ Model launched in tmux session: '$TMUX_SESSION'"
echo "💡 View logs:   tmux attach-session -t $TMUX_SESSION"
echo "💡 Detach:      Ctrl+B, then D"
echo "💡 Kill session: tmux kill-session -t $TMUX_SESSION"
