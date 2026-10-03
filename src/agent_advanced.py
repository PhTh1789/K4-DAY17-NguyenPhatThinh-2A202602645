from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: compact thread context plus persistent user profile memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route one validated turn through offline or optional live mode."""

        normalized_user_id = user_id.strip()
        normalized_thread_id = thread_id.strip()
        normalized_message = message.strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty.")
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")
        if not normalized_message:
            raise ValueError("message must not be empty.")
        if self.langchain_agent is not None:
            return self._reply_live(
                normalized_user_id, normalized_thread_id, normalized_message
            )
        return self._reply_offline(
            normalized_user_id, normalized_thread_id, normalized_message
        )

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id.strip(), 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id.strip(), 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Persist profile facts and answer deterministically from all memory layers."""

        updates = self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        if not self._is_query(message) and updates:
            rendered = ", ".join(f"{key}={value}" for key, value in updates.items())
            response = f"Mình đã cập nhật hồ sơ: {rendered}."
        response_tokens = self._record_response(thread_id, response, prompt_tokens)
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    def _reply_live(
        self, user_id: str, thread_id: str, message: str
    ) -> dict[str, Any]:
        self._persist_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        live_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": f"Persistent user profile:\n{self.profile_store.read_text(user_id)}",
            }
        ]
        summary = str(context["summary"])
        if summary:
            live_messages.append(
                {"role": "system", "content": f"Older thread summary:\n{summary}"}
            )
        recent_messages = context["messages"]
        if not isinstance(recent_messages, list):
            raise TypeError("Compact-memory messages must be a list.")
        live_messages.extend(dict(item) for item in recent_messages)
        result = self.langchain_agent.invoke({"messages": live_messages})
        response = self._result_text(result)
        response_tokens = self._record_response(thread_id, response, prompt_tokens)
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile, summary, and recent-message context for one turn."""

        context = self.compact_memory.context(thread_id)
        total = estimate_tokens(self.profile_store.read_text(user_id))
        total += estimate_tokens(str(context["summary"]))
        messages = context["messages"]
        if not isinstance(messages, list):
            raise TypeError("Compact-memory messages must be a list.")
        total += sum(
            estimate_tokens(f"{item['role']}: {item['content']}") for item in messages
        )
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from persisted facts without an LLM."""

        del thread_id
        if not self._is_query(message):
            return "Mình đã ghi nhận nội dung và cập nhật memory khi phù hợp."
        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin bền vững để trả lời."
        return self._answer_from_facts(facts, message.casefold())

    def _persist_updates(self, user_id: str, message: str) -> dict[str, str]:
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        return updates

    def _record_response(
        self, thread_id: str, response: str, prompt_tokens: int
    ) -> int:
        self.compact_memory.append(thread_id, "assistant", response)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return response_tokens

    @staticmethod
    def _is_query(message: str) -> bool:
        lower = message.casefold()
        return "?" in message or any(
            marker in lower for marker in ("nhắc lại", "tóm tắt", "thử nhớ")
        )

    @staticmethod
    def _answer_from_facts(facts: dict[str, str], lower_question: str) -> str:
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
        rendered = [f"{labels[key]}: {facts[key]}" for key in requested]
        style = facts.get("response_style", "").casefold()
        if "3 bullet" in style:
            buckets = [[], [], []]
            for index, item in enumerate(rendered):
                buckets[index % 3].append(item)
            return "\n".join(
                f"- {'; '.join(bucket) if bucket else 'Không có thêm thông tin liên quan'}."
                for bucket in buckets
            )
        if "bullet" in style:
            return "\n".join(f"- {item}." for item in rendered)
        return "; ".join(rendered) + "."

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
        """Build an optional live agent driven by the custom memory layer."""

        try:
            model = build_chat_model(self.config.model)
            from langchain.agents import create_agent
        except (ImportError, ValueError):
            return None
        return create_agent(
            model=model,
            system_prompt=(
                "Answer concisely using the supplied persistent profile, compact "
                "summary, and recent messages. Prefer newer corrected facts."
            ),
        )
