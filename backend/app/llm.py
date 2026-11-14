
"""
Configurable LLM providers with automatic primary-to-backup failover.

Supports Anthropic, OpenAI-compatible APIs and Ollama.
A temporary failure on the primary provider triggers the configured backup.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai", "ollama")

RETRYABLE_STATUS_CODES = (408, 425, 429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    """The LLM call failed."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMConfigError(LLMError):
    """The LLM is not correctly configured."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...


def validate_llm_config(s: Settings) -> list[str]:
    problems: list[str] = []
    provider = s.llm_provider.strip().lower()

    if not provider:
        return [
            "LLM_PROVIDER is not set. Choose one of: "
            "anthropic, openai, ollama."
        ]

    if provider not in PROVIDERS:
        return [
            f"LLM_PROVIDER='{s.llm_provider}' is not supported. "
            f"Choose one of: {', '.join(PROVIDERS)}."
        ]

    if not s.llm_model.strip():
        problems.append(
            "LLM_MODEL is not set. Use a model name enabled for your account."
        )

    if provider in ("anthropic", "openai") and not s.llm_api_key.strip():
        problems.append(
            f"LLM_API_KEY is required for provider '{provider}'."
        )

    return problems


class AnthropicClient:
    def __init__(self, s: Settings):
        self.base = (
            s.llm_base_url or "https://api.anthropic.com"
        ).rstrip("/")
        self.model = s.llm_model
        self.key = s.llm_api_key
        self.timeout = s.llm_timeout_seconds

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0,
        }

        headers = {
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = _post(
            f"{self.base}/v1/messages",
            headers,
            body,
            self.timeout,
        )

        try:
            return "".join(
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            ).strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(
                "Unexpected response shape from the Anthropic API."
            ) from exc


class OpenAIClient:
    """Works with OpenAI-compatible APIs, including Gemini and Groq."""

    def __init__(self, s: Settings):
        self.base = (
            s.llm_base_url or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = s.llm_model
        self.key = s.llm_api_key
        self.timeout = s.llm_timeout_seconds

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.key}"
        }

        try:
            data = _post(url, headers, body, self.timeout)

        except LLMError as exc:
            msg = str(exc)

            if "HTTP 400" not in msg or not any(
                key in msg
                for key in (
                    "max_tokens",
                    "max_completion_tokens",
                    "temperature",
                )
            ):
                raise

            # Some newer OpenAI-compatible models require
            # max_completion_tokens instead of max_tokens.
            body = {
                key: value
                for key, value in body.items()
                if key not in ("max_tokens", "temperature")
            } | {
                "max_completion_tokens": max_tokens
            }

            data = _post(url, headers, body, self.timeout)

        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""

            usage = data.get("usage") or {}
            token_details = (
                usage.get("completion_tokens_details") or {}
            )

            # Log response metadata only.
            # Never log prompts, document contents, model output or keys.
            log.info(
                "llm_response_metadata",
                extra={
                    "ctx": {
                        "model": self.model,
                        "finish_reason": choice.get("finish_reason"),
                        "content_length": (
                            len(content)
                            if isinstance(content, str)
                            else -1
                        ),
                        "content_type": type(content).__name__,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": token_details.get(
                            "reasoning_tokens"
                        ),
                        "has_refusal": bool(message.get("refusal")),
                    }
                },
            )

            if not isinstance(content, str):
                raise LLMError(
                    "The LLM returned an unsupported content format."
                )

            return content.strip()

        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "Unexpected response shape from the OpenAI-compatible API."
            ) from exc


class OllamaClient:
    def __init__(self, s: Settings):
        self.base = (
            s.llm_base_url or s.ollama_base_url
        ).rstrip("/")
        self.model = s.llm_model
        self.timeout = s.llm_timeout_seconds

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        data = _post(
            f"{self.base}/api/chat",
            {},
            body,
            self.timeout,
        )

        try:
            return (data["message"]["content"] or "").strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(
                "Unexpected response shape from Ollama."
            ) from exc


def _provider_message(resp: httpx.Response) -> str:
    """Return shortened provider error text without exposing the request."""
    try:
        data = resp.json()

        if isinstance(data, dict):
            err = data.get("error", data)

            if isinstance(err, dict):
                msg = (
                    err.get("message")
                    or err.get("detail")
                    or err.get("msg")
                    or str(err)
                )
            elif isinstance(err, list):
                msg = "; ".join(
                    str(
                        item.get("message")
                        or item.get("detail")
                        or item.get("msg")
                        or item
                    )
                    if isinstance(item, dict)
                    else str(item)
                    for item in err
                )
            else:
                msg = str(err)

        elif isinstance(data, list):
            messages = []

            for item in data:
                if isinstance(item, dict):
                    messages.append(
                        str(
                            item.get("message")
                            or item.get("detail")
                            or item.get("msg")
                            or item
                        )
                    )
                else:
                    messages.append(str(item))

            msg = "; ".join(messages)

        else:
            msg = str(data)

        return str(msg)[:240]

    except (ValueError, TypeError):
        return resp.text[:160]


def _post(
    url: str,
    headers: dict,
    body: dict,
    timeout: float,
    *,
    retries: int = 0,
) -> dict:
    resp = None

    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json=body,
                timeout=timeout,
            )

        except httpx.HTTPError as exc:
            if attempt < retries:
                time.sleep(1.5)
                continue

            raise LLMError(
                f"Could not reach the LLM service ({type(exc).__name__}). "
                "Check the URL and your network.",
                retryable=True,
            ) from exc

        if (
            resp.status_code in RETRYABLE_STATUS_CODES
            and attempt < retries
        ):
            time.sleep(1.5)
            continue

        break

    assert resp is not None

    if resp.status_code in (401, 403):
        raise LLMError(
            "The LLM provider rejected the credentials. "
            "Check the API key."
        )

    if resp.status_code == 404:
        raise LLMError(
            "The LLM endpoint or model was not found (HTTP 404). "
            "Check the model name and base URL."
        )

    if resp.status_code == 429:
        detail = _provider_message(resp)

        raise LLMError(
            "The LLM provider rate-limited the request "
            f"(HTTP 429). Details: {detail or 'No details provided.'}",
            retryable=True,
        )

    if resp.status_code >= 400:
        detail = _provider_message(resp)

        raise LLMError(
            f"The LLM provider returned HTTP {resp.status_code}"
            + (f": {detail}" if detail else "."),
            retryable=resp.status_code in RETRYABLE_STATUS_CODES,
        )

    try:
        data = resp.json()

        if not isinstance(data, dict):
            raise LLMError(
                "The LLM provider returned an unexpected response format."
            )

        return data

    except ValueError as exc:
        raise LLMError(
            "The LLM provider returned a non-JSON response."
        ) from exc


class FallbackClient:
    """Try the primary provider, then Groq on temporary failures."""

    def __init__(self, primary: LLMClient, backup: LLMClient):
        self.primary = primary
        self.backup = backup

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            return self.primary.complete(system, user, max_tokens)

        except LLMError as primary_exc:
            if not primary_exc.retryable:
                raise

            log.warning(
                "Primary LLM provider failed temporarily. "
                "Trying the backup provider."
            )

            try:
                result = self.backup.complete(
                    system,
                    user,
                    max_tokens,
                )

                log.info("Backup LLM provider succeeded.")
                return result

            except LLMError as backup_exc:
                log.error(
                    "Both primary and backup LLM providers failed."
                )

                raise LLMError(
                    "Both LLM providers failed. "
                    f"Primary: {str(primary_exc)[:120]}. "
                    f"Backup: {str(backup_exc)[:120]}",
                    retryable=backup_exc.retryable,
                ) from backup_exc


def get_llm(s: Settings) -> LLMClient:
    problems = validate_llm_config(s)

    if problems:
        raise LLMConfigError(" ".join(problems))

    provider = s.llm_provider.strip().lower()

    primary = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "ollama": OllamaClient,
    }[provider](s)

    # If no Groq key is configured, retain the original single-provider mode.
    if not s.groq_api_key.strip():
        log.warning(
            "Groq backup is not configured. Using primary LLM only."
        )
        return primary

    # Build Groq's OpenAI-compatible client using separate settings.
    groq_settings = s.model_copy(
        update={
            "llm_provider": "openai",
            "llm_model": s.groq_model,
            "llm_api_key": s.groq_api_key,
            "llm_base_url": s.groq_base_url,
        }
    )

    backup = OpenAIClient(groq_settings)

    log.info(
        "LLM automatic fallback configured: primary=%s, backup=groq",
        provider,
    )

    return FallbackClient(primary, backup)
