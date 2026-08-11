from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any, Protocol


class InferenceBackend(Protocol):
    model_name: str

    def generate_json(
        self,
        system: str,
        prompt: str,
        *,
        max_new_tokens: int = 1_024,
        enable_thinking: bool = False,
    ) -> Any: ...


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>|<reasoning>.*?</reasoning>",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_model_noise(text: str) -> str:
    """Remove thinking blocks / fences that Qwen often wraps around JSON."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    # Drop unclosed thinking prefix if the model never closed the tag.
    cleaned = re.sub(r"<think(?:ing)?>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = _FENCE_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    # Common prose prefixes before the object.
    cleaned = re.sub(
        r"^(?:here(?:'s| is)|returning|output|result|json)\s*(?:the\s+)?(?:json|response)?\s*[:\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_balanced_json_fragment(text: str) -> str | None:
    """Return the first balanced {...} or [...] slice, respecting string literals."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    # Truncated: return open fragment from start for repair.
    return text[start:] if depth > 0 else None


def _repair_truncated_json(fragment: str) -> str:
    """Best-effort close for truncated JSON objects/arrays from max_tokens cuts."""
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in fragment:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    repaired = fragment.rstrip()
    # Trailing comma before we close containers.
    repaired = re.sub(r",\s*$", "", repaired)
    if in_string:
        repaired += '"'
    # If we ended on a bare key like `"summary":` close with empty string.
    if re.search(r':\s*$', repaired):
        repaired += '""'
    while stack:
        repaired += stack.pop()
    return repaired


def extract_json(text: str) -> Any:
    """Parse model JSON robustly: strip thinking/fences, balance braces, repair truncations."""
    cleaned = _strip_model_noise(text)
    if not cleaned:
        raise ValueError("Model returned empty content where JSON was required")

    candidates: list[str] = [cleaned]
    fragment = _extract_balanced_json_fragment(cleaned)
    if fragment and fragment not in candidates:
        candidates.append(fragment)
    if fragment:
        repaired = _repair_truncated_json(fragment)
        if repaired not in candidates:
            candidates.append(repaired)

    errors: list[Exception] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(exc)
            continue

    # Last resort: original rfind heuristic for odd wrappers.
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if starts:
        start = min(starts)
        closing = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(closing)
        if end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                errors.append(exc)
            try:
                return json.loads(_repair_truncated_json(cleaned[start:]))
            except json.JSONDecodeError as exc:
                errors.append(exc)

    raise ValueError(
        "Could not parse JSON from model output: "
        + (str(errors[-1]) if errors else "no JSON object found")
    )


def _default_llm_concurrency() -> int:
    return 100


def _env_concurrency(*keys: str, default: int | None = None) -> int:
    fallback = _default_llm_concurrency() if default is None else default
    for key in keys:
        raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            continue
        try:
            return max(1, min(100, int(raw)))
        except ValueError:
            continue
    return fallback


# Shared across backends in this process.
_LLM_MAX_CONCURRENCY = _env_concurrency(
    "STRUCTURE_LLM_MAX_CONCURRENCY",
    "OPENROUTER_MAX_CONCURRENCY",
    "DS_STRUCTURE_WORKERS",
)
_LLM_SEMAPHORE = threading.Semaphore(_LLM_MAX_CONCURRENCY)


class QwenOpenRouterBackend:
    """OpenAI-compatible chat completions via OpenRouter."""

    DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3-32b"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_input_tokens: int = 24_000,
        temperature: float = 0.5,
        top_p: float = 0.8,
        top_k: int = 20,
        seed: int = 42,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 4,
        site_url: str = "https://rayonlabs.ai",
        app_title: str = "ds-structure",
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Missing OPENROUTER_API_KEY")

        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.api_key = key
        self.api_url = api_url or os.environ.get("OPENROUTER_API_URL", self.DEFAULT_URL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.site_url = site_url
        self.app_title = app_title

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        enable_thinking: bool = False,
    ) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_title,
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        # OpenRouter/Qwen3: effort=none alone still spends reasoning tokens. Pair with exclude.
        if enable_thinking:
            payload["reasoning"] = {"effort": "medium"}
        else:
            payload["reasoning"] = {"effort": "none", "exclude": True}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            response = None
            with _LLM_SEMAPHORE:
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    last_error = exc

            if response is None:
                if attempt >= self.max_retries:
                    raise last_error or RuntimeError("OpenRouter request failed")
                time.sleep(min(30.0, (2**attempt) + random.random()))
                continue

            if response.status_code == 429:
                last_error = RuntimeError(f"OpenRouter rate limited (429): {response.text[:200]}")
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(min(30.0, (2**attempt) + random.random()))
                continue

            if response.status_code >= 500:
                last_error = RuntimeError(f"OpenRouter server error ({response.status_code})")
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(min(30.0, (2**attempt) + random.random()))
                continue

            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = ""
            # Prefer content; if empty, some providers stash text under reasoning fields.
            if not content.strip():
                for key in ("reasoning", "reasoning_content", "reasoning_details"):
                    alt = message.get(key)
                    if isinstance(alt, str) and alt.strip():
                        content = alt
                        break
            return _strip_model_noise(content) or content

        raise last_error or RuntimeError("OpenRouter request failed")

    def generate_json(
        self,
        system: str,
        prompt: str,
        *,
        max_new_tokens: int = 1_024,
        enable_thinking: bool = False,
    ) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = self._completion(messages, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
        if not text.strip():
            raise ValueError("OpenRouter API returned an empty completion")
        return extract_json(text)


class QwenTransformersBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B",
        *,
        max_input_tokens: int = 24_000,
        temperature: float = 0.5,
        top_p: float = 0.8,
        top_k: int = 20,
        seed: int = 42,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate_json(
        self,
        system: str,
        prompt: str,
        *,
        max_new_tokens: int = 1_024,
        enable_thinking: bool = False,
    ) -> Any:
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(enable_thinking),
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs.input_ids.shape[1] :]
        return extract_json(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def build_default_backend(
    model_name: str,
    *,
    max_input_tokens: int = 16_000,
    temperature: float = 0.5,
    top_p: float = 0.8,
    top_k: int = 20,
    seed: int = 42,
) -> InferenceBackend:
    """Structure jobs always use OpenRouter (requires OPENROUTER_API_KEY)."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Missing OPENROUTER_API_KEY for ds-structure LLM calls")

    model = model_name or QwenOpenRouterBackend.DEFAULT_MODEL
    if "/" in model and not model.startswith("qwen/"):
        # Map legacy Chutes-style ids to the OpenRouter slug.
        if model.upper().startswith("QWEN/QWEN3-32B"):
            model = QwenOpenRouterBackend.DEFAULT_MODEL
    return QwenOpenRouterBackend(
        model,
        max_input_tokens=max_input_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
