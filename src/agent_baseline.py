from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: full within-thread history and no persistent memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using live mode when configured, otherwise deterministic offline mode."""

        del user_id  # Baseline intentionally has no user-scoped memory.
        normalized_thread_id = thread_id.strip()
        normalized_message = message.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")
        if not normalized_message:
            raise ValueError("message must not be empty.")
        if self.langchain_agent is not None:
            return self._reply_live(normalized_thread_id, normalized_message)
        return self._reply_offline(normalized_thread_id, normalized_message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id.strip())
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id.strip())
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Store and answer from only the selected thread."""

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message.strip()})
        prompt_tokens = self._context_tokens(session.messages)
        response = self._offline_response(session.messages, message)
        response_tokens = estimate_tokens(response)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message.strip()})
        prompt_tokens = self._context_tokens(session.messages)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message.strip()}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = self._result_text(result)
        response_tokens = estimate_tokens(response)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += response_tokens
        session.messages.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    @staticmethod
    def _context_tokens(messages: list[dict[str, str]]) -> int:
        return sum(
            estimate_tokens(f"{item['role']}: {item['content']}") for item in messages
        )

    @staticmethod
    def _offline_response(messages: list[dict[str, str]], current_message: str) -> str:
        facts: dict[str, str] = {}
        for item in messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        lower = current_message.casefold()
        is_query = "?" in current_message or any(
            marker in lower for marker in ("nhắc lại", "tóm tắt", "thử nhớ")
        )
        if not is_query:
            return "Mình đã ghi nhận thông tin trong thread hiện tại."
        return BaselineAgent._answer_from_facts(facts, lower)

    @staticmethod
    def _answer_from_facts(facts: dict[str, str], lower_question: str) -> str:
        if not facts:
            return "Mình chưa có thông tin từ thread hiện tại để trả lời."

        field_markers = {
            "name": ("tên", "là ai"),
            "location": ("ở đâu", "nơi ở", "huế", "đà nẵng", "hà nội"),
            "profession": ("nghề", "công việc", "làm gì", "engineer", "manager"),
            "response_style": ("style", "kiểu trả lời", "cách trả lời"),
            "favorite_drink": ("đồ uống", "uống gì"),
            "favorite_food": ("món ăn", "ăn gì"),
            "pet": ("nuôi con gì", "thú nuôi", "corgi"),
            "interests": ("mối quan tâm", "kỹ thuật", "quan tâm chính"),
        }
        labels = {
            "name": "Tên",
            "location": "Nơi ở",
            "profession": "Nghề nghiệp",
            "response_style": "Style trả lời",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú nuôi",
            "interests": "Mối quan tâm",
        }
        requested = [
            key
            for key, markers in field_markers.items()
            if key in facts and any(marker in lower_question for marker in markers)
        ]
        if not requested:
            requested = [key for key in labels if key in facts]
        return "; ".join(f"{labels[key]}: {facts[key]}" for key in requested) + "."

    @staticmethod
    def _result_text(result: Any) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise RuntimeError("Live agent returned no messages.")
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
        return str(content)

    def _maybe_build_langchain_agent(self):
        """Build the optional live agent; missing credentials select offline mode."""

        try:
            model = build_chat_model(self.config.model)
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except (ImportError, ValueError):
            return None
        return create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            system_prompt=(
                "Answer concisely. Use only messages from the current thread and "
                "never claim to remember another thread."
            ),
        )
