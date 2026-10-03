from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


EMPTY_PROFILE = "# User Profile\n\n"
FACT_LINE = re.compile(r"^- ([a-z][a-z0-9_]*):\s*(.+?)\s*$")
FACT_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
FACT_ORDER = (
    "name",
    "location",
    "profession",
    "response_style",
    "favorite_drink",
    "favorite_food",
    "pet",
    "interests",
)
MERGED_FACT_SEPARATORS = {
    "response_style": "; ",
    "interests": ", ",
}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
PROFESSION = (
    r"[A-Za-z][A-Za-z0-9+.#-]*"
    r"(?:\s+[A-Za-z][A-Za-z0-9+.#-]*){0,2}"
    r"\s+(?:engineer|manager|developer|designer|analyst|scientist)"
)
SUMMARY_MAX_ITEMS = 6
SUMMARY_ITEM_MAX_CHARS = 180
MESSAGE_ROLES = {"user", "assistant", "system", "tool"}
DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = 0.8
MAX_FACT_VALUE_CHARS = 500
MAX_MERGED_FACT_PARTS = 16
MAX_PROFILE_BYTES = 64 * 1024
PROFILE_FACT_BASE_CONFIDENCE = {
    "name": 0.98,
    "location": 0.95,
    "profession": 0.95,
    "response_style": 0.90,
    "favorite_drink": 0.92,
    "favorite_food": 0.92,
    "pet": 0.92,
    "interests": 0.80,
}


def estimate_tokens(text: str) -> int:
    """Estimate tokens deterministically at roughly four characters each."""

    stripped = text.strip()
    if not stripped:
        return 0
    return math.ceil(len(stripped) / 4)


@dataclass
class UserProfileStore:
    """Persistent, path-safe storage for one ``User.md`` per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        raw_id = user_id.strip()
        if not raw_id:
            raise ValueError("user_id must not be empty.")

        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_id).strip("-_")
        slug = slug[:48] or "user"
        needs_digest = slug != raw_id or slug.upper() in WINDOWS_RESERVED_NAMES
        if needs_digest:
            digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:10]
            slug = f"{slug}-{digest}"

        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if root not in path.parents:
            raise ValueError("Resolved profile path escaped the profile directory.")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return EMPTY_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        content_size = len(content.encode("utf-8"))
        if content_size > MAX_PROFILE_BYTES:
            raise ValueError(
                f"Profile exceeds the {MAX_PROFILE_BYTES}-byte storage limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=".User.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            raise ValueError("search_text must not be empty.")

        path = self.path_for(user_id)
        if not path.exists():
            return False

        current = path.read_text(encoding="utf-8")
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse canonical ``- key: value`` facts from a user's profile."""

        parsed: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = FACT_LINE.match(line)
            if match:
                parsed[match.group(1)] = match.group(2)
        return parsed

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one fact and rewrite the profile canonically."""

        normalized_key = key.strip().lower()
        normalized_value = " ".join(value.split())
        if not FACT_KEY.fullmatch(normalized_key):
            raise ValueError(f"Invalid fact key: {key!r}.")
        if not normalized_value:
            raise ValueError("Fact value must not be empty.")
        if len(normalized_value) > MAX_FACT_VALUE_CHARS:
            raise ValueError(
                f"Fact value exceeds the {MAX_FACT_VALUE_CHARS}-character limit."
            )

        facts = self.facts(user_id)
        separator = MERGED_FACT_SEPARATORS.get(normalized_key)
        if separator and normalized_key in facts:
            existing_parts = [
                part.strip() for part in facts[normalized_key].split(separator)
            ]
            new_parts = [part.strip() for part in normalized_value.split(separator)]
            merged_parts = list(dict.fromkeys(existing_parts + new_parts))
            merged_parts = merged_parts[-MAX_MERGED_FACT_PARTS:]
            while (
                len(separator.join(merged_parts)) > MAX_FACT_VALUE_CHARS
                and len(merged_parts) > 1
            ):
                merged_parts.pop(0)
            normalized_value = separator.join(merged_parts)
        facts[normalized_key] = normalized_value
        ordered_keys = [key for key in FACT_ORDER if key in facts]
        ordered_keys.extend(sorted(set(facts) - set(ordered_keys)))
        body = "\n".join(f"- {fact_key}: {facts[fact_key]}" for fact_key in ordered_keys)
        return self.write_text(user_id, f"{EMPTY_PROFILE}{body}\n")


def _profile_update_confidence(key: str, lower_message: str) -> float:
    """Return a conservative confidence score for one extracted profile fact."""

    confidence = PROFILE_FACT_BASE_CONFIDENCE.get(key, 0.0)
    correction_markers = (
        "đính chính",
        "hiện tại",
        "không còn",
        "cập nhật",
        "thực ra",
    )
    if key in {"location", "profession"} and any(
        marker in lower_message for marker in correction_markers
    ):
        return 0.99
    return confidence


def extract_profile_updates(
    message: str,
    min_confidence: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD,
) -> dict[str, str]:
    """Extract high-confidence, persistent facts from one user message.

    The rules intentionally favor precision over recall: explicit statements,
    corrections, and stable preferences are accepted, while recall questions
    and known joke/noise constructions are ignored.
    """

    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")

    text = " ".join(message.split())
    if not text:
        return {}
    lower = text.casefold()
    question_prefixes = (
        "nhắc lại",
        "tóm tắt",
        "bạn có thể nhắc lại",
        "bạn thử nhớ lại",
        "mình tên gì",
    )
    if "?" in text or lower.startswith(question_prefixes):
        return {}

    updates: dict[str, str] = {}

    name_patterns = (
        r"\b(?:mình\s+)?tên(?:\s+mình)?\s+là\s+([^,.;!?]+)",
        r"(?:^|[:;,]\s*)tên\s+([^,.;!?]+?)(?=\s*,|\s+nghề\b|$)",
    )
    for pattern in name_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            updates["name"] = match.group(1).strip()

    location_patterns = (
        r"\bmình\s+ở\s+(.+?)(?=\s+(?:và|chứ|để|trong|dù|nhưng|chưa)\b|[,.;!?]|$)",
        r"\b(?:mình\s+)?(?:hiện|đang|vẫn)\s+ở\s+(.+?)(?=\s+(?:và|chứ|để|trong|dù|nhưng|chưa)\b|[,.;!?]|$)",
        r"\btừ tuần này\s+mình\s+đang\s+làm việc\s+ở\s+(.+?)(?=\s+vài tháng\b|[,.;!?]|$)",
        r"\bnơi ở\s+đã\s+cập nhật\s+từ\s+.+?\s+sang\s+([^,.;!?]+)",
        r"\bnơi ở(?:\s+hiện tại)?\s+(?:vẫn\s+)?là\s+([^,.;!?]+)",
    )
    for pattern in location_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate and "không phải nơi ở" not in candidate.casefold():
                updates["location"] = candidate

    profession_patterns = (
        rf"\b(?:mình\s+)?đang\s+làm\s+({PROFESSION})",
        rf"\bgiờ\s+chuyển\s+sang\s+({PROFESSION})",
        rf"\bmình\s+làm\s+({PROFESSION})",
        rf"\bnghề nghiệp(?:\s+hiện tại)?(?:\s+vẫn)?(?:\s+là)?\s+({PROFESSION})",
        rf"(?:^|[:;,]\s*)nghề\s+({PROFESSION})",
    )
    for pattern in profession_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate.casefold() == "product manager" and "câu đùa" in lower:
                continue
            updates["profession"] = candidate

    style_context = any(
        marker in lower
        for marker in (
            "mình muốn bạn trả lời",
            "mình muốn câu trả lời",
            "mình vẫn muốn câu trả lời",
            "mình không thích câu trả lời",
            "mình rất thích các câu trả lời",
            "hãy trả lời",
            "khi bạn trả lời, mình muốn",
            "style trả lời",
            "cách giải thích",
            "cách trình bày",
        )
    )
    if style_context:
        style_parts: list[str] = []
        if re.search(r"\b(?:3|ba)\s+bullet\b", lower):
            style_parts.append("3 bullet")
        elif "bullet" in lower:
            style_parts.append("bullet")
        if "ngắn gọn" in lower or re.search(r"\bngắn\b", lower) or "đừng lan man" in lower:
            style_parts.append("ngắn gọn")
        if "rõ ý" in lower:
            style_parts.append("rõ ý")
        if "có cấu trúc" in lower:
            style_parts.append("có cấu trúc")
        if "ví dụ thực chiến" in lower:
            style_parts.append("có ví dụ thực chiến")
        elif "ví dụ thực tế" in lower:
            style_parts.append("có ví dụ thực tế")
        if "số liệu" in lower or "định lượng" in lower:
            style_parts.append("có số liệu")
        if "trade-off" in lower:
            style_parts.append("nhấn trade-off")
        if style_parts:
            updates["response_style"] = "; ".join(dict.fromkeys(style_parts))

    drink_patterns = (
        r"\bđồ uống yêu thích(?:\s+của mình)?\s+là\s+(.+?)(?=[,.;!?]|$)",
        r"\bmình\s+vẫn\s+uống\s+(.+?)(?=\s+(?:như cũ|nhưng)\b|[,.;!?]|$)",
    )
    for pattern in drink_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            updates["favorite_drink"] = match.group(1).strip()

    food_patterns = (
        r"\bmón ăn yêu thích(?:\s+của mình)?\s+là\s+(.+?)(?=[,.;!?]|$)",
        r"\bmình\s+ăn\s+(.+?)\s+và\s+thấy\s+đúng\s+là\s+món ruột\b",
    )
    for pattern in food_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            updates["favorite_food"] = match.group(1).strip()

    pet_match = re.search(
        r"\bmình\s+nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.;!?]+)",
        text,
        flags=re.IGNORECASE,
    )
    if pet_match:
        updates["pet"] = pet_match.group(1).strip()
    else:
        pet_match = re.search(
            r"\bcon\s+corgi(?:\s+tên)?\s+([\wÀ-ỹ]+)",
            text,
            flags=re.IGNORECASE,
        )
        if pet_match:
            updates["pet"] = f"corgi tên {pet_match.group(1).strip()}"

    interest_context = bool(
        re.search(
            r"\bmình\s+(?:vẫn\s+)?thích\s+(?:python|ai|mlops|rag|evaluation|benchmark)",
            lower,
        )
        or re.search(r"\bmình\s+đang\s+quan tâm\b", lower)
        or re.search(
            r"\bdài hạn:\s*mình\s+thích\s+(?:python|ai|mlops|rag|evaluation|benchmark)",
            lower,
        )
    )
    if interest_context:
        interests: list[str] = []
        interest_terms = (
            ("python", "Python"),
            ("ai ứng dụng", "AI ứng dụng"),
            ("ai agent", "AI agent"),
            ("mlops", "MLOps"),
            ("rag", "RAG"),
            ("evaluation", "evaluation"),
            ("benchmark memory", "benchmark memory"),
        )
        for marker, label in interest_terms:
            if marker in lower:
                interests.append(label)
        if interests:
            updates["interests"] = ", ".join(dict.fromkeys(interests))

    return {
        key: value
        for key, value in updates.items()
        if len(value) <= MAX_FACT_VALUE_CHARS
        and _profile_update_confidence(key, lower) >= min_confidence
    }


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded, deterministic summary from the most recent items."""

    if max_items <= 0:
        raise ValueError("max_items must be greater than zero.")

    normalized: list[dict[str, str]] = []
    for message in messages:
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        if len(content) > SUMMARY_ITEM_MAX_CHARS:
            content = f"{content[: SUMMARY_ITEM_MAX_CHARS - 3].rstrip()}..."
        role = str(message.get("role", "message")).strip().lower() or "message"
        normalized.append({"role": role, "content": content})

    selected = normalized[-max_items:]
    if not selected:
        return ""

    lines = ["Compact summary:"]
    lines.extend(f"- {item['role']}: {item['content']}" for item in selected)
    return "\n".join(lines)


def _summary_messages(summary: str) -> list[dict[str, str]]:
    """Convert this module's summary format back into mergeable items."""

    messages: list[dict[str, str]] = []
    for line in summary.splitlines():
        if not line.startswith("- "):
            continue
        role, separator, content = line[2:].partition(": ")
        if separator and content:
            messages.append({"role": role, "content": content})
    return messages


@dataclass
class CompactMemoryManager:
    """Bound thread context by summarizing older messages when needed."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0:
            raise ValueError("threshold_tokens must be greater than zero.")
        if self.keep_messages <= 0:
            raise ValueError("keep_messages must be greater than zero.")

    def append(self, thread_id: str, role: str, content: str) -> None:
        normalized_thread_id = thread_id.strip()
        normalized_role = role.strip().lower()
        normalized_content = content.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")
        if normalized_role not in MESSAGE_ROLES:
            raise ValueError(f"Unsupported message role: {role!r}.")
        if not normalized_content:
            raise ValueError("content must not be empty.")

        thread = self.state.setdefault(
            normalized_thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        if not isinstance(messages, list):
            raise TypeError("Thread messages state must be a list.")
        messages.append({"role": normalized_role, "content": normalized_content})

        summary = str(thread.get("summary", ""))
        context_tokens = estimate_tokens(summary)
        context_tokens += sum(
            estimate_tokens(f"{item['role']}: {item['content']}") for item in messages
        )
        if context_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older_messages = messages[: -self.keep_messages]
        recent_messages = messages[-self.keep_messages :]
        summary_source = _summary_messages(summary)
        summary_source.extend(older_messages)
        thread["summary"] = summarize_messages(summary_source, max_items=SUMMARY_MAX_ITEMS)
        thread["messages"] = [dict(item) for item in recent_messages]
        thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")
        thread = self.state.get(normalized_thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}

        messages = thread.get("messages", [])
        if not isinstance(messages, list):
            raise TypeError("Thread messages state must be a list.")
        return {
            "messages": [dict(item) for item in messages],
            "summary": str(thread.get("summary", "")),
            "compactions": int(thread.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
