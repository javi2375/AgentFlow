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
import time
import requests
import socket
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
        connection_timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        :param model_string: Name of the model as served by LM Studio.
        :param system_prompt: Default system prompt to prepend for conversations.
        :param is_multimodal: LM Studio proxy is typically text-only; leave False.
        :param use_cache: Enable diskcache-based response caching.
        :param base_url: Override LM Studio server base URL (OpenAI-compatible).
        :param api_key: Optional API key; LM Studio generally doesn't require one.
        :param connection_timeout: Timeout in seconds for connection attempts.
        :param max_retries: Maximum number of connection retry attempts.
        """
        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries

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

        # Perform connection health check before creating client
        self._validate_connection()
        
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.connection_timeout,
            )
            # Test connection with a simple request
            self._test_connection()
        except Exception as e:
            raise ValueError(
                f"Failed to connect to LM Studio server at {self.base_url}. "
                f"Please ensure LM Studio's OpenAI-compatible server is running. Error: {e}"
            )

    def _validate_connection(self):
        """
        Validate that LM Studio server is reachable before creating client.
        Implements retry logic with exponential backoff.
        """
        for attempt in range(self.max_retries):
            try:
                # Extract host and port from base_url for socket connection test
                parsed_url = self.base_url.replace("http://", "").replace("https://", "").split("/")[0]
                host, port = parsed_url.split(":") if ":" in parsed_url else (parsed_url, "80")
                
                # Test socket connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.connection_timeout)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                
                if result == 0:
                    return  # Connection successful
                    
                raise ConnectionError(f"Socket connection failed with code {result}")
                
            except (socket.gaierror, socket.timeout, ConnectionError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    self._raise_connection_error(e, attempt + 1)
                
                wait_time = min(2 ** attempt, 5)  # Exponential backoff, max 5 seconds
                time.sleep(wait_time)
                
    def _test_connection(self):
        """
        Test LM Studio API endpoint with a simple request.
        """
        try:
            # Test with a simple models list request
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.connection_timeout
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API test failed: {e}")
            
    def _raise_connection_error(self, original_error, attempts):
        """
        Raise a detailed connection error with troubleshooting information.
        """
        error_details = {
            "base_url": self.base_url,
            "attempts": attempts,
            "timeout": self.connection_timeout,
            "original_error": str(original_error),
            "troubleshooting": [
                "1. Ensure LM Studio is running on your machine",
                "2. Check that LM Studio's OpenAI-compatible server is enabled",
                f"3. Verify server URL is correct (current: {self.base_url})",
                "4. Check if port is available and not blocked by firewall",
                "5. Try restarting LM Studio application"
            ]
        }
        
        raise ValueError(
            f"LM Studio connection failed after {attempts} attempts. "
            f"Details: {error_details}"
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
