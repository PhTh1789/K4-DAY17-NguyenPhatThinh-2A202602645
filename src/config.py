from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, memory settings, and model configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.8


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _positive_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}.")
    return value


def _non_negative_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    return value


def _probability(name: str, default: float) -> float:
    value = _non_negative_float(name, default)
    if value > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    return value


def _provider_api_key(provider: str) -> str | None:
    if provider == "openai":
        return _env("OPENAI_API_KEY")
    if provider == "custom":
        return _env("CUSTOM_API_KEY")
    if provider == "gemini":
        return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    if provider == "anthropic":
        return _env("ANTHROPIC_API_KEY")
    if provider == "openrouter":
        return _env("OPENROUTER_API_KEY")
    return None


def _provider_base_url(provider: str) -> str | None:
    if provider == "openai":
        return _env("OPENAI_BASE_URL")
    if provider == "custom":
        return _env("CUSTOM_BASE_URL")
    if provider == "anthropic":
        return _env("ANTHROPIC_BASE_URL")
    if provider == "ollama":
        return _env("OLLAMA_BASE_URL") or "http://localhost:11434"
    if provider == "openrouter":
        return _env("OPENROUTER_BASE_URL")
    return None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env`, create the state directory, and return lab settings.

    API keys remain optional here because the required offline benchmark must
    work without credentials. ``build_chat_model`` validates live-mode needs.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env", override=False)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(_env("LLM_PROVIDER") or "openai")
    model = ProviderConfig(
        provider=provider,
        model_name=_env("LLM_MODEL") or DEFAULT_MODELS[provider],
        temperature=_non_negative_float("LLM_TEMPERATURE", 0.0),
        api_key=_provider_api_key(provider),
        base_url=_provider_base_url(provider),
    )

    judge_provider = normalize_provider(_env("JUDGE_PROVIDER") or provider)
    judge_default_model = (
        model.model_name if judge_provider == provider else DEFAULT_MODELS[judge_provider]
    )
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=_env("JUDGE_MODEL") or judge_default_model,
        temperature=_non_negative_float("JUDGE_TEMPERATURE", 0.0),
        api_key=_env("JUDGE_API_KEY") or _provider_api_key(judge_provider),
        base_url=_env("JUDGE_BASE_URL") or _provider_base_url(judge_provider),
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=_positive_int("COMPACT_THRESHOLD_TOKENS", 1200),
        compact_keep_messages=_positive_int("COMPACT_KEEP_MESSAGES", 6),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=_probability(
            "PROFILE_CONFIDENCE_THRESHOLD", 0.8
        ),
    )
