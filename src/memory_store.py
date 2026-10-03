from __future__ import annotations

import hashlib
import math
import re
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
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
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
        path.write_text(updated, encoding="utf-8", newline="\n")
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

        facts = self.facts(user_id)
        facts[normalized_key] = normalized_value
        ordered_keys = [key for key in FACT_ORDER if key in facts]
        ordered_keys.extend(sorted(set(facts) - set(ordered_keys)))
        body = "\n".join(f"- {fact_key}: {facts[fact_key]}" for fact_key in ordered_keys)
        return self.write_text(user_id, f"{EMPTY_PROFILE}{body}\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    raise NotImplementedError


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    raise NotImplementedError


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        raise NotImplementedError

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        raise NotImplementedError

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        raise NotImplementedError
