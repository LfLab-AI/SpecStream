import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts/specstream/paper_eval/qwen3/analyze_grant_events.py"
)
spec = importlib.util.spec_from_file_location("specstream_grant_event_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _rows(*, state="DRAFT_CATCHUP", budget=4, tokens=4, terminal=None):
    grant = {
        "event": "issued",
        "request_id": "r",
        "spec_cnt": "3",
        "grant_epoch": "7",
        "grant_state": state,
        "grant_tokens": str(budget),
        "tpc_low": "0",
        "tpc_high": "4",
        "monotonic_ns": "1000000",
        "deadline_us": "1500",
    }
    ack = dict(
        grant,
        event="ack",
        grant_tokens=str(tokens),
        monotonic_ns="2000000",
        grant_state=state if terminal is None else terminal,
    )
    return [grant, ack]


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
def test_full_catchup_ack_exhausts_exact_issued_budget(budget):
    summary = audit.analyze_rows(_rows(budget=budget, tokens=budget), expected_tpcs=4)
    assert summary["status"] == "PASS"
    assert summary["issued_tokens"] == summary["acked_tokens"] == budget
    assert summary["returned_unused_tokens"] == 0
    assert summary["catchup_multi_token_grants"] == int(budget > 1)


@pytest.mark.parametrize("terminal", sorted(audit.PARTIAL_DISPOSITIONS))
@pytest.mark.parametrize("tokens", [0, 1, 3])
def test_terminal_catchup_ack_accounts_completed_prefix_and_unused_budget(
    terminal, tokens
):
    summary = audit.analyze_rows(
        _rows(tokens=tokens, terminal=terminal), expected_tpcs=4
    )
    assert summary["status"] == "PASS"
    assert summary["acked_tokens"] == tokens
    assert summary["returned_unused_tokens"] == 4 - tokens


@pytest.mark.parametrize("budget", [1, 4])
def test_prefill_completion_consumes_one_token_then_returns_unused_budget(budget):
    summary = audit.analyze_rows(
        _rows(budget=budget, tokens=1, terminal="PREFILL_COMPLETE"), expected_tpcs=4
    )
    assert summary["status"] == "PASS"
    assert summary["acked_tokens"] == 1
    assert summary["returned_unused_tokens"] == budget - 1


@pytest.mark.parametrize(
    "terminal",
    ["EXPIRED", "PREFILL_DEFERRED", "SUPERSEDED", "PAUSED", "FINISHED", "EARLY_FINISH"],
)
def test_disposed_single_token_slack_is_not_counted_as_success(terminal):
    rows = _rows(state="SLACK_FILL", budget=1, tokens=0, terminal=terminal)
    summary = audit.analyze_rows(rows, expected_tpcs=4)
    assert summary["status"] == "PASS"
    assert summary["slack_fill_success"] == 0
    assert (
        audit.analyze_rows(rows, expected_tpcs=4, require_slack_fill=True)["status"]
        == "FAIL"
    )


def test_ack_after_launch_deadline_is_valid_but_expired_issuance_is_not():
    rows = _rows(state="SLACK_FILL", budget=1, tokens=1)
    assert (
        audit.analyze_rows(rows, expected_tpcs=4, require_slack_fill=True)["status"]
        == "PASS"
    )
    rows[0]["monotonic_ns"] = "1500000"
    summary = audit.analyze_rows(rows, expected_tpcs=4)
    assert summary["status"] == "FAIL"
    assert any("issued after" in text for text in summary["violations"])


@pytest.mark.parametrize(
    "budget,tokens,terminal,state",
    [
        (4, 5, "DRAFT_CATCHUP", "DRAFT_CATCHUP"),
        (4, -1, "EXPIRED", "DRAFT_CATCHUP"),
        (4, 3, "DRAFT_CATCHUP", "DRAFT_CATCHUP"),
        (4, 0, "", "DRAFT_CATCHUP"),
        (4, 4, "EXPIRED", "DRAFT_CATCHUP"),
        (4, 4, "SUPERSEDED", "DRAFT_CATCHUP"),
        (9, 9, "DRAFT_CATCHUP", "DRAFT_CATCHUP"),
        (0, 0, "EXPIRED", "DRAFT_CATCHUP"),
        (2, 2, "SLACK_FILL", "SLACK_FILL"),
        (1, 1, "PREFILL_COMPLETE", "SLACK_FILL"),
        (4, 2, "PREFILL_COMPLETE", "DRAFT_CATCHUP"),
        (4, 0, "PREFILL_DEFERRED", "DRAFT_CATCHUP"),
        (4, 1, "UNKNOWN", "DRAFT_CATCHUP"),
        (4, 4, "TARGET_EXCLUSIVE", "TARGET_EXCLUSIVE"),
    ],
)
def test_budget_or_disposition_violations_remain_failures(
    budget, tokens, terminal, state
):
    summary = audit.analyze_rows(
        _rows(state=state, budget=budget, tokens=tokens, terminal=terminal),
        expected_tpcs=4,
    )
    assert summary["status"] == "FAIL"
    assert summary["violations"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ack",
        "duplicate_ack",
        "missing_issue",
        "duplicate_issue",
        "ack_before_issue",
    ],
)
def test_exactly_one_issue_and_one_terminal_ack_is_still_required(mutation):
    rows = _rows()
    if mutation == "missing_ack":
        rows.pop()
    elif mutation == "missing_issue":
        rows.pop(0)
    elif mutation == "duplicate_ack":
        rows.append(dict(rows[1]))
    elif mutation == "duplicate_issue":
        rows.insert(0, dict(rows[0]))
    else:
        rows.reverse()
    summary = audit.analyze_rows(rows, expected_tpcs=4)
    assert summary["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("grant_tokens", "1.5"),
        ("grant_tokens", "NaN"),
        ("grant_tokens", "Infinity"),
        ("tpc_high", "3"),
    ],
)
def test_invalid_ack_numbers_and_tpc_mismatch_are_rejected(field, value):
    rows = _rows()
    rows[1][field] = value
    assert audit.analyze_rows(rows, expected_tpcs=4)["status"] == "FAIL"


def test_integer_parsing_preserves_large_epoch_values_without_float_rounding():
    assert audit.integer({"epoch": "9007199254740993"}, "epoch") == 9007199254740993
    assert audit.integer({"budget": "4.0"}, "budget") == 4


@pytest.mark.parametrize("complete,expected_exit", [(True, 0), (False, 2)])
def test_cli_writes_a_report_and_preserves_failure_exit(
    tmp_path, complete, expected_exit
):
    events = tmp_path / "events.csv"
    output = tmp_path / "report.json"
    rows = _rows()
    with events.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows if complete else rows[:1])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--events",
            str(events),
            "--output",
            str(output),
            "--expected-tpcs",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == ("PASS" if complete else "FAIL")
    if not complete:
        assert "expected one ACK event, got 0" in summary["violations"][0]
