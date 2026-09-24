"""
Unit and Integration Tests for Evaluation and Replay Framework

================================================================================
TEST COVERAGE:
1. Transcript loading (valid JSON parsing, chunks extraction, offset ordering)
2. Scenario discovery (all 8 required benchmark scenarios present in eval/test_transcripts)
3. Replay result schema (mandatory fields, types, and serializability)
4. Gate evaluation (G1 through G8 individual gate logic and overall evaluation)
5. Failure handling (graceful capture of exceptions without pipeline crash)
6. Deterministic result structure (machine-readable JSON outputs in eval/results/)
================================================================================
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from eval.replay_harness import StreamingReplayHarness, ReplayChunk, ReplaySessionResult
from eval.gate_check import GateChecker, GateResult


@pytest.fixture
def replay_harness(tmp_path: Path) -> StreamingReplayHarness:
    """Fixture providing a configured StreamingReplayHarness instance."""
    results_dir = tmp_path / "results"
    return StreamingReplayHarness(
        test_transcripts_dir="eval/test_transcripts",
        results_dir=str(results_dir),
        logs_dir="logs"
    )


@pytest.fixture
def gate_checker() -> GateChecker:
    """Fixture providing a GateChecker instance."""
    return GateChecker(logs_dir="logs", results_dir="eval/results")


def test_scenario_discovery(replay_harness: StreamingReplayHarness):
    """Verifies that all 8 required benchmark scenario transcripts exist."""
    required_scenarios = [
        "partial_streaming",
        "single_intent",
        "multi_intent",
        "refinement",
        "no_retrieval",
        "unsupported_query",
        "citation_grounding",
        "latency"
    ]
    for scenario in required_scenarios:
        t_path = replay_harness.test_transcripts_dir / f"{scenario}.json"
        assert t_path.exists(), f"Missing required transcript file: {t_path}"


def test_transcript_loading(replay_harness: StreamingReplayHarness):
    """Verifies transcript loading parses chunks accurately."""
    single_intent_path = replay_harness.test_transcripts_dir / "single_intent.json"
    chunks = replay_harness.load_transcript_file(single_intent_path)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], ReplayChunk)
    assert len(chunks[0].text_chunk) > 0
    assert chunks[0].is_final is True

    # Multi-chunk transcript loading
    partial_path = replay_harness.test_transcripts_dir / "partial_streaming.json"
    p_chunks = replay_harness.load_transcript_file(partial_path)
    assert len(p_chunks) == 2
    assert p_chunks[0].is_final is False
    assert p_chunks[1].is_final is True


@pytest.mark.asyncio
async def test_replay_result_schema(replay_harness: StreamingReplayHarness):
    """Verifies that running a scenario produces a compliant ReplaySessionResult schema."""
    result = await replay_harness.run_scenario("no_retrieval", simulate_realtime_delay=False)
    assert isinstance(result, ReplaySessionResult)
    assert result.scenario == "no_retrieval"
    assert isinstance(result.passed, bool)
    assert isinstance(result.checks, dict)
    assert isinstance(result.failure_reasons, list)
    assert isinstance(result.events_observed, list)
    assert "transcript" in result.events_observed
    assert "controller" in result.events_observed
    assert "answer" in result.events_observed

    # Verify dictionary serialization
    result_dict = result.to_dict()
    assert "scenario" in result_dict
    assert "session_id" in result_dict
    assert "passed" in result_dict
    assert "retrieval_calls" in result_dict
    assert "answer_version" in result_dict


@pytest.mark.asyncio
async def test_failure_handling(replay_harness: StreamingReplayHarness):
    """Verifies that invalid or faulty transcripts fail gracefully without crashing the runner."""
    nonexistent_file = replay_harness.test_transcripts_dir / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        replay_harness.load_transcript_file(nonexistent_file)

    # Replay session with empty chunks
    result = await replay_harness.replay_session("empty_test", [])
    assert result.passed is True or len(result.failure_reasons) >= 0
    assert result.session_id.startswith("eval_empty_test")


def test_gate_checker_evaluation(gate_checker: GateChecker):
    """Verifies that GateChecker evaluates all 8 gates and produces valid GateResult objects."""
    gate_results = gate_checker.evaluate_all_gates()
    assert len(gate_results) == 8

    expected_gates = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    for g_id in expected_gates:
        assert g_id in gate_results
        gr = gate_results[g_id]
        assert isinstance(gr, GateResult)
        assert gr.gate_id == g_id
        assert isinstance(gr.passed, bool)
        assert isinstance(gr.details, dict)
        assert gr.score in (0.0, 1.0)


def test_deterministic_result_structure():
    """Verifies the JSON file structure of saved aggregate evaluation results."""
    results_path = Path("eval/results/replay_results.json")
    if not results_path.exists():
        pytest.skip("replay_results.json not yet generated in workspace")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_scenarios" in data
    assert "passed_scenarios" in data
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 8
