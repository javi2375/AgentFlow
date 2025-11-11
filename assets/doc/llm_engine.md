## Supported LLM Engines

We support a broad range of LLM engines for agents and tools in [`factory.py`](../../agentflow/agentflow/engine/factory.py), including LM Studio, GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and more.

### 🍎 macOS Users - Recommended Setup

**LM Studio is the recommended LLM engine for macOS users** as it provides local inference without requiring NVIDIA GPUs or CUDA:

- ✅ Works on Apple Silicon (M1/M2/M3) and Intel Macs
- ✅ No CUDA or NVIDIA dependencies required
- ✅ Local inference with privacy and cost benefits
- ✅ OpenAI-compatible API integration

See the [LM Studio section](#using-lm-studio) below for setup instructions.

> **⚠️ Note for macOS Users:** vLLM requires NVIDIA GPUs and CUDA, which are not available on macOS. Use LM Studio instead for local model serving.

| Model Family | Model_string Example | Supported Models | Official Model List |
|--------------|---------------------|------------------|---------------------|
| vLLM | `vllm-Qwen/Qwen2.5-7B-Instruct` | Various vLLM-supported models (e.g., `Qwen2.5-7B-Instruct`, `Qwen2.5-VL-3B-Instruct`). Supports local checkpoint models for customization and local inference. **⚠️ Requires NVIDIA GPU + CUDA (not available on macOS)** | [vLLM Models](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| DashScope (Qwen) | `dashscope-qwen2.5-7b-instruct` | Qwen models via Alibaba Cloud DashScope API | [DashScope Models](https://help.aliyun.com/zh/model-studio/getting-started/models) |
| OpenAI | `gpt-4o`, `o1-mini` | `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-3.5-turbo`, `gpt-4`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o1-pro`, `o4-mini` | [OpenAI Models](https://platform.openai.com/docs/models) |
| Azure OpenAI | `azure-gpt-4o` | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-3.5-turbo`, `gpt-4`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o1-pro`, `o4-mini` | [Azure OpenAI Models](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#models) |
| Anthropic | `claude-3-5-sonnet-20241022` | `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-opus-20240229`, `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219` | [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models/all-models) |
| TogetherAI | `together-meta-llama/Llama-3-70b-chat-hf` | Most models including `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `Qwen/QwQ-32B`, `Qwen/Qwen2-VL-72B-Instruct`, `meta-llama/Llama-3-70b-chat-hf`, `Qwen/Qwen2-72B-Instruct` | [TogetherAI Models](https://api.together.ai/models) |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | `deepseek-chat`, `deepseek-reasoner` | [DeepSeek Models](https://api-docs.deepseek.com/quick_start/pricing) |
| Gemini | `gemini-2.0-flash` | `gemini-1.5-pro`, `gemini-1.5-flash-8b`, `gemini-1.5-flash`, `gemini-2.0-flash-lite`, `gemini-2.0-flash`, `gemini-2.5-pro-preview-03-25` | [Gemini Models](https://ai.google.dev/gemini-api/docs/models) |
| Grok | `grok-3`, `grok-2-vision` | `grok-2-vision-1212`, `grok-2-vision`, `grok-2-vision-latest`, `grok-3-mini-fast-beta`, `grok-3-mini-fast`, `grok-3-mini-fast-latest`, `grok-3-mini-beta`, `grok-3-mini`, `grok-3-mini-latest`, `grok-3-fast-beta`, `grok-3-fast`, `grok-3-fast-latest`, `grok-3-beta`, `grok-3`, `grok-3-latest` | [Grok Models](https://docs.x.ai/docs/models#models-and-pricing) |
| LiteLLM | `litellm-gpt-4o` | Any model supported by LiteLLM, including models from OpenAI, Anthropic, Google, Gemini, Mistral, Cohere, and more. | [LiteLLM Models](https://docs.litellm.ai/docs/providers) |
| Ollama | `ollama-qwen2.5` | Any model supported by Ollama, such as `DeepSeek-R1`, `Qwen 3`, `Llama 3.3`, `Gemma 3`, `Qwen 2.5-VL`, and other models. | [Ollama Models](https://ollama.ai/library) |
| LM Studio | `lmstudio-Qwen2.5-7B-Instruct` | Any text model you run locally via LM Studio's OpenAI-compatible endpoint (default `http://localhost:1234/v1`). **🍎 Recommended for macOS users** | LM Studio (local OpenAI-compatible) |

### Using LM Studio

1. In LM Studio, start the local server (OpenAI-compatible) and load a model (e.g., **Qwen2.5-7B-Instruct**).  
2. Set the environment variables in your private `.env`:
   ```
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   LMSTUDIO_API_KEY=lm-studio
   ```
3. In your config or code, use a model string with the `lmstudio-` prefix, e.g.:
   ```
   lmstudio-agentflow-planner-7b-mlx