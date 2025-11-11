set -ex

# Check for macOS and provide alternative setup
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected: This script installs NVIDIA GPU-specific packages."
    echo "   For macOS, use CPU/Metal compatible setup instead:"
    echo ""
    echo "📦 Recommended macOS setup:"
    echo "   uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools"
    echo "   uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0"
    echo "   uv pip install --no-cache-dir transformers==4.53.3"
    echo "   uv pip install --no-cache-dir -e .[dev,agent]"
    echo ""
    echo "🚀 For local LLM inference, install LM Studio:"
    echo "   1. Download from https://lmstudio.ai/"
    echo "   2. Install models like 'Qwen2.5-7B-Instruct'"
    echo "   3. Start server on port 1234"
    echo ""
    echo "📚 See assets/doc/llm_engine.md for LM Studio configuration."
    exit 0
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install NVIDIA CUDA toolkit."
    echo "   Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# python -m uv pip install --upgrade uv pip

uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir transformers==4.53.3
uv pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
uv pip install --no-cache-dir vllm==0.9.2
uv pip install --no-cache-dir verl==0.5.0

uv pip install --no-cache-dir -e .[dev,agent]
