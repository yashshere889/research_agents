from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from loguru import logger
from pydantic import BaseModel

from research_agents.config import settings
from research_agents.utils.retry import with_retry
from research_agents.utils.token_counter import count_tokens


class BaseAgent(ABC):
    """Base class for research agents backed by Gemini or Hugging Face."""

    MODEL = settings.agent_model
    _hf_cache: ClassVar[dict[str, tuple[Any, Any]]] = {}

    def __init__(self) -> None:
        self.total_tokens = 0
        self.revision_instructions: str | None = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's role/persona system prompt."""

    def _model(self) -> Any:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required to call Gemini agents.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run: python3 -m pip install google-genai"
            ) from exc

        return genai.Client(api_key=settings.gemini_api_key)

    def _hf_model(self) -> tuple[Any, Any]:
        cached = self._hf_cache.get(self.MODEL)
        if cached:
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face backend requires transformers, torch, accelerate, and "
                "optionally bitsandbytes. Run: python3 -m pip install -r "
                "research_agents/requirements.txt"
            ) from exc

        token = settings.hf_token or None
        processor = AutoProcessor.from_pretrained(self.MODEL, token=token)
        model_kwargs: dict[str, Any] = {"device_map": "auto", "token": token}

        if settings.hf_load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif settings.hf_torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, settings.hf_torch_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"

        if settings.hf_attn_implementation != "auto":
            model_kwargs["attn_implementation"] = settings.hf_attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.MODEL, **model_kwargs)
        model.eval()
        self._hf_cache[self.MODEL] = (processor, model)
        return processor, model

    def _extract_json_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            return cleaned

        object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if object_match:
            return object_match.group(0)
        array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if array_match:
            return array_match.group(0)
        raise ValueError(f"LLM did not return JSON. Response was: {text[:500]!r}")

    @with_retry(max_attempts=3, backoff_factor=2.0)
    def call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Call the configured LLM and optionally parse the response into a Pydantic model."""
        del tools
        prompt_parts = [m["content"] for m in messages]

        if self.revision_instructions:
            prompt_parts.append(
                "\nRevision instructions from review team:\n"
                f"{self.revision_instructions}"
            )

        if response_model:
            schema = json.dumps(response_model.model_json_schema(), indent=2)
            prompt_parts[-1] += (
                "\n\nRespond ONLY with a valid JSON object matching this schema:\n"
                f"{schema}\n"
                "No markdown fences, no extra text."
            )

        provider = settings.llm_provider.lower()
        if provider == "huggingface":
            text = self._call_huggingface(prompt_parts)
            self.total_tokens += count_tokens("".join(prompt_parts)) + count_tokens(text)
        elif provider == "gemini":
            text = self._call_gemini(prompt_parts)
        else:
            raise RuntimeError(
                f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
                "Use 'gemini' or 'huggingface'."
            )

        logger.debug(f"[{self.__class__.__name__}] tokens: {self.total_tokens}")

        if response_model:
            cleaned = self._extract_json_text(text)
            return response_model.model_validate_json(cleaned)
        return text

    def _call_gemini(self, prompt_parts: list[str]) -> str:
        model = self._model()
        try:
            from google.genai import types

            generation_config = types.GenerateContentConfig(
                max_output_tokens=settings.max_output_tokens,
                system_instruction=self.system_prompt,
            )
            response = model.models.generate_content(
                model=self.MODEL,
                contents="\n\n".join(prompt_parts),
                config=generation_config,
            )
        except Exception as exc:
            if "ResourceExhausted" in exc.__class__.__name__ or "quota" in str(exc).lower():
                raise RuntimeError(
                    f"Gemini quota was exhausted for model {self.MODEL!r}. "
                    "Either enable billing / quota for that model, or set "
                    "AGENT_MODEL=gemini-2.5-flash in your .env file."
                ) from exc
            raise

        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            self.total_tokens += prompt_tokens + output_tokens
        else:
            text = getattr(response, "text", "") or ""
            self.total_tokens += count_tokens("".join(prompt_parts)) + count_tokens(text)
        return getattr(response, "text", "") or ""

    def _call_huggingface(self, prompt_parts: list[str]) -> str:
        processor, model = self._hf_model()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(prompt_parts)},
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_output_tokens,
            do_sample=False,
        )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        if hasattr(processor, "parse_response"):
            parsed = processor.parse_response(response)
            if isinstance(parsed, dict):
                return parsed.get("answer") or parsed.get("content") or str(parsed)
            return str(parsed)
        return response.strip()

    @abstractmethod
    def run(self, state: Any) -> Any:
        """Execute the agent task and return a typed output model."""
