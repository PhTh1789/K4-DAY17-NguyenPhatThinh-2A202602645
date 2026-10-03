from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load and validate a benchmark dataset."""

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in benchmark dataset {path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise ValueError(f"Benchmark dataset must be a JSON list: {path}")

    conversations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_data):
        label = f"{path.name}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object.")

        conversation_id = item.get("id")
        user_id = item.get("user_id")
        turns = item.get("turns")
        recall_questions = item.get("recall_questions")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError(f"{label}.id must be a non-empty string.")
        if conversation_id in seen_ids:
            raise ValueError(f"Duplicate conversation id: {conversation_id!r}.")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"{label}.user_id must be a non-empty string.")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{label}.turns must be a non-empty list.")
        if not all(isinstance(turn, str) and turn.strip() for turn in turns):
            raise ValueError(f"{label}.turns must contain non-empty strings.")
        if not isinstance(recall_questions, list):
            raise ValueError(f"{label}.recall_questions must be a list.")

        for question_index, question_item in enumerate(recall_questions):
            question_label = f"{label}.recall_questions[{question_index}]"
            if not isinstance(question_item, dict):
                raise ValueError(f"{question_label} must be an object.")
            question = question_item.get("question")
            expected = question_item.get("expected_contains")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{question_label}.question must be a non-empty string.")
            if not isinstance(expected, list) or not expected:
                raise ValueError(
                    f"{question_label}.expected_contains must be a non-empty list."
                )
            if not all(isinstance(value, str) and value.strip() for value in expected):
                raise ValueError(
                    f"{question_label}.expected_contains must contain non-empty strings."
                )

        seen_ids.add(conversation_id)
        conversations.append(item)

    return conversations


def _normalized(text: str) -> str:
    """Normalize Unicode and case so Vietnamese matching is deterministic."""

    return unicodedata.normalize("NFC", text).casefold()


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 based on expected facts found in an answer."""

    if not expected:
        return 1.0
    normalized_answer = _normalized(answer)
    matches = sum(_normalized(value) in normalized_answer for value in expected)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score factual coverage (80%) and concise readability (20%)."""

    normalized_answer = _normalized(answer)
    coverage = (
        sum(_normalized(value) in normalized_answer for value in expected) / len(expected)
        if expected
        else 1.0
    )
    stripped_answer = answer.strip()
    if not stripped_answer:
        readability = 0.0
    elif len(stripped_answer) <= 800:
        readability = 1.0
    else:
        readability = 0.5
    return round((0.8 * coverage) + (0.2 * readability), 3)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and fresh-thread recall questions."""

    if agent.config.state_dir != config.state_dir:
        raise ValueError("Agent and benchmark must use the same state directory.")

    user_ids = {str(item["user_id"]) for item in conversations}
    has_persistent_memory = callable(getattr(agent, "memory_file_size", None))
    initial_memory_bytes = (
        sum(agent.memory_file_size(user_id) for user_id in user_ids)
        if has_persistent_memory
        else 0
    )

    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: set[str] = set()

    for conversation in conversations:
        conversation_id = str(conversation["id"])
        user_id = str(conversation["user_id"])
        dialogue_thread = f"{conversation_id}:dialogue"
        thread_ids.add(dialogue_thread)

        for turn in conversation["turns"]:
            result = agent.reply(user_id, dialogue_thread, str(turn))
            agent_tokens += int(result["tokens"])
            prompt_tokens += int(result["prompt_tokens"])

        for question_index, recall_item in enumerate(conversation["recall_questions"]):
            recall_thread = f"{conversation_id}:recall:{question_index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall_item["question"]))
            answer = str(result["response"])
            expected = [str(value) for value in recall_item["expected_contains"]]
            agent_tokens += int(result["tokens"])
            prompt_tokens += int(result["prompt_tokens"])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_memory_bytes = (
        sum(agent.memory_file_size(user_id) for user_id in user_ids)
        if has_persistent_memory
        else 0
    )
    compactions = sum(agent.compaction_count(thread_id) for thread_id in thread_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=(sum(recall_scores) / len(recall_scores) if recall_scores else 0.0),
        response_quality=(
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        ),
        memory_growth_bytes=max(0, final_memory_bytes - initial_memory_bytes),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a GitHub-flavoured Markdown table."""

    table_rows = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            row.recall_score,
            row.response_quality,
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    return tabulate(
        table_rows,
        headers=[
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ],
        tablefmt="github",
        floatfmt=".3f",
    )


def _run_suite(
    title: str,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> None:
    """Run a suite in disposable state so repeated benchmarks stay comparable."""

    with TemporaryDirectory(prefix="day17-benchmark-") as temporary_directory:
        suite_config = replace(config, state_dir=Path(temporary_directory) / "state")
        rows = [
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(suite_config, force_offline=True),
                conversations,
                suite_config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(suite_config, force_offline=True),
                conversations,
                suite_config,
            ),
        ]
    print(f"## {title}")
    print(format_rows(rows))


def main() -> None:
    """Run the standard and long-context stress benchmark suites offline."""

    root = Path(__file__).resolve().parent.parent
    config = load_config(root)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    _run_suite("Standard Benchmark", standard, config)
    print()
    _run_suite("Long-Context Stress Benchmark", stress, config)


if __name__ == "__main__":
    main()
