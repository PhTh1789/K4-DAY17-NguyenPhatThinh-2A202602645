from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import EMPTY_PROFILE, UserProfileStore, estimate_tokens


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


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    raise NotImplementedError


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    raise NotImplementedError


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    raise NotImplementedError
