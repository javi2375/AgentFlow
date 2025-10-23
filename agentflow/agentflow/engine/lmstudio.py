try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "If you'd like to use LM Studio, please install the openai package by running `pip install openai`."
    )

import os
import json
import base64
import platformdirs
from typing import List, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import EngineLM, CachedEngine


class ChatLMStudio(EngineLM, CachedEngine):
    """
    LM Studio implementation of the EngineLM interface.

    LM Studio exposes an OpenAI-compatible REST API (chat completions) on a local
    server (default base_url http://localhost:1234/v1). This client uses the
    `openai` SDK pointed at that base_url.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "demyagent-4b-qx86-hi-mlx",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        use_cache: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        :param model_string: Name of the model as served by LM Studio.
        :param system_prompt: Default system prompt to prepend for conversations.
        :param is_multimodal: LM Studio proxy is typically text-only; leave False.
        :param use_cache: Enable diskcache-based response caching.
        :param base_url: Override LM Studio server base URL (OpenAI-compatible).
        :param api_key: Optional API key; LM Studio generally doesn't require one.
        """
        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_lmstudio_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)

        self.base_url = base_url or os.environ.get(
            "LMSTUDIO_BASE_URL", "http://localhost:1234/v1"
        )
        # Keep a default token for compatibility; LM Studio often ignores it.
        self.api_key = api_key or os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to connect to LM Studio server at {self.base_url}. "
                f"Please ensure LM Studio's OpenAI-compatible server is running. Error: {e}"
            )

    @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def generate(
        self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs
    ):
        """
        Generate a response using LM Studio's OpenAI-compatible /chat/completions.

        Supports:
          - str prompt
          - list[str] concatenated by newline

        Bytes (image) content is not supported by default; raise if provided.
        """
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)

            elif isinstance(content, list):
                # If list is all strings, join as a single prompt
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(full_text, system_prompt=system_prompt, **kwargs)

                # If any bytes are present, multimodal is not supported by default
                elif any(isinstance(item, bytes) for item in content):
                    if not self.is_multimodal:
                        raise NotImplementedError(
                            f"Multimodal generation is not supported for LM Studio models via the OpenAI proxy."
                        )
                    # If future LM Studio builds support images, adapt similarly to vLLM's _generate_multimodal
                    return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

                else:
                    raise ValueError("Unsupported content in list: only str or bytes are allowed.")
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, "args", None),
            }

    def _generate_text(
        self,
        prompt: str,
        system_prompt=None,
        max_tokens: int = 2048,
        top_p: float = 0.99,
        response_format=None,
        **kwargs,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Respect common sampling kwargs with sensible defaults
        temperature = kwargs.get("temperature", 0.7)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # Many LM Studio-routed models ignore penalties; harmless if unsupported.
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        response_text = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    # Placeholder for parity with other engines; raises until LM Studio gains image support.
    def _generate_multimodal(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0.0,
        max_tokens=2048,
        top_p=0.99,
        response_format=None,
    ):
        raise NotImplementedError(
            "LM Studio multimodal generation is not supported via the OpenAI-compatible proxy."
        )
