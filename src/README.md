# Completed Implementation

This folder contains the completed Day 17 memory-system implementation:

- deterministic offline Baseline and Advanced agents;
- optional live models for OpenAI, custom OpenAI-compatible endpoints, Gemini,
  Anthropic, Ollama, and OpenRouter;
- persistent `User.md` profiles with structured extraction and conflict handling;
- compact memory for bounded long-context prompts;
- confidence and storage-growth guardrails;
- Standard and Long-Context Stress benchmarks; and
- 22 behavioral and regression tests.

Run from the repository root:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

See the root `STEP8.md` for benchmark results, trade-off analysis, limitations,
and production risks.
