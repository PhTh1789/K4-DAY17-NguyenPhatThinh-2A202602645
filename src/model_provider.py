from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)


@dataclass
class ProviderConfig:
    """Configuration shared by every supported chat-model provider."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return the canonical provider name for a user-facing alias."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "openai": "openai",
        "gpt": "openai",
        "chatgpt": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {supported}.") from exc


def _require_api_key(config: ProviderConfig) -> str:
    if config.api_key:
        return config.api_key
    raise ValueError(f"Provider {config.provider!r} requires an API key in live mode.")


def _model_kwargs(config: ProviderConfig) -> dict[str, object]:
    if not config.model_name.strip():
        raise ValueError("model_name must not be empty.")
    return {
        "model": config.model_name.strip(),
        "temperature": config.temperature,
    }


def build_chat_model(config: ProviderConfig):
    """Instantiate a LangChain chat model without making a network request.

    Imports stay local to each branch so deterministic offline mode does not
    require provider SDKs merely to import the agents.
    """

    provider = normalize_provider(config.provider)
    kwargs = _model_kwargs(config)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if config.base_url:
            kwargs["base_url"] = config.base_url
        kwargs["api_key"] = _require_api_key(config)
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL/base_url.")
        kwargs["base_url"] = config.base_url
        # Some local OpenAI-compatible servers ignore authentication, while
        # ChatOpenAI still expects a non-empty value.
        kwargs["api_key"] = config.api_key or "not-required"
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs["api_key"] = _require_api_key(config)
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs = {
            "model_name": kwargs["model"],
            "temperature": kwargs["temperature"],
            "api_key": _require_api_key(config),
        }
        if config.base_url:
            anthropic_kwargs["base_url"] = config.base_url
        return ChatAnthropic(**anthropic_kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        if config.base_url:
            kwargs["base_url"] = config.base_url
        kwargs["api_key"] = _require_api_key(config)
        return ChatOpenRouter(**kwargs)

    # normalize_provider currently makes this unreachable. Keeping an explicit
    # failure here protects the factory if provider support changes later.
    raise ValueError(f"Unsupported provider: {provider!r}.")
