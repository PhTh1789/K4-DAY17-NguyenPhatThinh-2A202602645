from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import (
    EMPTY_PROFILE,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    raise NotImplementedError


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify profile CRUD, canonical facts, token estimates, and path safety."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct"

    assert store.read_text(user_id) == EMPTY_PROFILE
    assert store.file_size(user_id) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("12345") == 2

    initial = "# User Profile\n\n- name: DũngCT\n"
    path = store.write_text(user_id, initial)
    assert path == tmp_path / "profiles" / user_id / "User.md"
    assert store.read_text(user_id) == initial
    assert store.file_size(user_id) == len(initial.encode("utf-8"))

    assert store.edit_text(user_id, "DũngCT", "Dũng CT") is True
    assert store.edit_text(user_id, "missing", "replacement") is False
    assert "Dũng CT" in store.read_text(user_id)

    store.upsert_fact(user_id, "location", "Huế")
    store.upsert_fact(user_id, "profession", "MLOps engineer")
    store.upsert_fact(user_id, "location", "Đà Nẵng")
    facts = store.facts(user_id)
    assert facts == {
        "name": "Dũng CT",
        "location": "Đà Nẵng",
        "profession": "MLOps engineer",
    }
    assert "Huế" not in store.read_text(user_id)

    unsafe_path = store.path_for("../../outside")
    assert (tmp_path / "profiles").resolve() in unsafe_path.parents
    assert unsafe_path.name == "User.md"

    with pytest.raises(ValueError, match="user_id"):
        store.path_for("   ")
    with pytest.raises(ValueError, match="search_text"):
        store.edit_text(user_id, "", "replacement")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Chào bạn, mình tên là DũngCT.", {"name": "DũngCT"}),
        (
            "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
            {"location": "Đà Nẵng", "profession": "backend engineer"},
        ),
        (
            "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
            {"location": "Huế"},
        ),
        (
            "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
            {"profession": "MLOps engineer"},
        ),
        (
            "Có lúc mình đùa là chuyển sang product manager, nhưng đó chỉ là câu đùa. "
            "Nghề nghiệp hiện tại vẫn là MLOps engineer. Hà Nội chỉ là nơi đi họp, "
            "không phải nơi ở hiện tại.",
            {"profession": "MLOps engineer"},
        ),
        (
            "Mình muốn bạn trả lời ngắn gọn thành 3 bullet, có ví dụ thực chiến và "
            "nhấn vào trade-off.",
            {
                "response_style": (
                    "3 bullet; ngắn gọn; có ví dụ thực chiến; nhấn trade-off"
                )
            },
        ),
        (
            "Mình thích Python, AI ứng dụng và cà phê sữa đá.",
            {"interests": "Python, AI ứng dụng"},
        ),
        ("Đồ uống yêu thích là cà phê sữa đá.", {"favorite_drink": "cà phê sữa đá"}),
        ("Món ăn yêu thích là mì Quảng.", {"favorite_food": "mì Quảng"}),
        ("Mình nuôi một bé corgi tên Bơ.", {"pet": "corgi tên Bơ"}),
        ("Mình tên gì và hiện tại đang ở đâu?", {}),
        ("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.", {}),
        (
            "Nếu sau này mình nhắc Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại.",
            {},
        ),
    ],
)
def test_extract_profile_updates_handles_corrections_and_noise(
    message: str, expected: dict[str, str]
) -> None:
    assert extract_profile_updates(message) == expected


def test_profile_updates_replace_corrected_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    messages = [
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    ]

    for message in messages:
        for key, value in extract_profile_updates(message).items():
            store.upsert_fact("dungct", key, value)

    assert store.facts("dungct") == {
        "location": "Huế",
        "profession": "MLOps engineer",
    }
    profile = store.read_text("dungct")
    assert "Đà Nẵng" not in profile
    assert "backend engineer" not in profile


def test_profile_updates_merge_cumulative_preferences(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    store.upsert_fact("dungct", "response_style", "ngắn gọn; có ví dụ thực tế")
    store.upsert_fact("dungct", "response_style", "có ví dụ thực tế; nhấn trade-off")
    store.upsert_fact("dungct", "interests", "Python, AI agent")
    store.upsert_fact("dungct", "interests", "Python, AI ứng dụng")

    assert store.facts("dungct") == {
        "response_style": "ngắn gọn; có ví dụ thực tế; nhấn trade-off",
        "interests": "Python, AI agent, AI ứng dụng",
    }


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify old messages become a bounded summary after the threshold."""

    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    for index in range(8):
        manager.append(
            "thread-1",
            "user",
            f"user-message-{index} " + "chi tiết dài " * 16,
        )
        manager.append("thread-1", "assistant", f"assistant-message-{index}")

    context = manager.context("thread-1")
    messages = context["messages"]
    summary = context["summary"]
    assert isinstance(messages, list)
    assert isinstance(summary, str)
    assert manager.compaction_count("thread-1") > 0
    assert context["compactions"] == manager.compaction_count("thread-1")
    assert 0 < len(messages) <= 2
    assert summary.startswith("Compact summary:\n")
    assert len(summary.splitlines()) <= 7
    assert estimate_tokens(summary) < 400

    snapshot = manager.context("thread-1")
    snapshot_messages = snapshot["messages"]
    assert isinstance(snapshot_messages, list)
    snapshot_messages.clear()
    assert manager.context("thread-1")["messages"]

    assert manager.context("unknown") == {
        "messages": [],
        "summary": "",
        "compactions": 0,
    }

    sample = [
        {"role": "user", "content": f"message-{index}"} for index in range(8)
    ]
    bounded_summary = summarize_messages(sample, max_items=3)
    assert "message-0" not in bounded_summary
    assert "message-5" in bounded_summary
    assert "message-7" in bounded_summary
    assert len(bounded_summary.splitlines()) == 4

    with pytest.raises(ValueError, match="threshold_tokens"):
        CompactMemoryManager(threshold_tokens=0, keep_messages=2)
    with pytest.raises(ValueError, match="keep_messages"):
        CompactMemoryManager(threshold_tokens=80, keep_messages=0)
    with pytest.raises(ValueError, match="role"):
        manager.append("thread-1", "invalid", "message")


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    raise NotImplementedError


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    raise NotImplementedError
