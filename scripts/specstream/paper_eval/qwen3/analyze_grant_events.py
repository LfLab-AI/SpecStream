#!/usr/bin/env python3
"""Validate one issue and one terminal ACK for each bounded Draft lease."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


PARTIAL_DISPOSITIONS = frozenset(
    {"EXPIRED", "SUPERSEDED", "PAUSED", "FINISHED", "EARLY_FINISH"}
)
ISSUED_STATES = frozenset({"SLACK_FILL", "DRAFT_CATCHUP"})


def integer(row: dict[str, str], key: str) -> int:
    """Parse integral CSV values exactly, including legacy integral decimals."""
    raw = str(row.get(key, "0") or "0").strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{key} is not an integer: {raw!r}") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"{key} is not a finite integer: {raw!r}")
    return int(value)


def analyze_rows(
    rows: list[dict[str, str]],
    *,
    expected_tpcs: int,
    require_slack_fill: bool = False,
) -> dict:
    if expected_tpcs < 1:
        raise ValueError("expected_tpcs must be positive")
    violations: list[str] = []
    keyed = defaultdict(lambda: defaultdict(list))
    if not rows:
        violations.append("grant event stream is empty")
    for index, row in enumerate(rows, start=2):
        try:
            key = (
                str(row.get("request_id", "")),
                integer(row, "spec_cnt"),
                integer(row, "grant_epoch"),
            )
        except ValueError as exc:
            violations.append(f"CSV row {index}: {exc}")
            continue
        if not key[0] or key[1] < 0 or key[2] < 1:
            violations.append(
                f"CSV row {index}: invalid request/spec_cnt/epoch key {key}"
            )
        event = str(row.get("event", ""))
        if event not in {"issued", "ack"}:
            violations.append(f"CSV row {index}: unsupported event {event!r}")
        keyed[key][event].append((index, row))

    states = Counter()
    terminal_states = Counter()
    slack_issued = slack_success = expired_acks = 0
    granted_total = acked_total = unused_total = 0
    catchup_multi = catchup_tokens = 0
    for key, events in keyed.items():
        issued = events.get("issued", [])
        acks = events.get("ack", [])
        if len(issued) != 1:
            violations.append(f"{key}: expected one issued event, got {len(issued)}")
        if len(acks) != 1:
            violations.append(f"{key}: expected one ACK event, got {len(acks)}")
        if len(issued) != 1 or len(acks) != 1:
            continue
        issue_index, grant = issued[0]
        ack_index, ack = acks[0]
        if ack_index <= issue_index:
            violations.append(f"{key}: ACK was logged before its issued event")
        state = str(grant.get("grant_state", ""))
        ack_state = str(ack.get("grant_state", ""))
        states[state] += 1
        terminal_states[ack_state or state] += 1
        try:
            issued_tokens = integer(grant, "grant_tokens")
            ack_tokens = integer(ack, "grant_tokens")
            issue_monotonic_ns = integer(grant, "monotonic_ns")
            ack_monotonic_ns = integer(ack, "monotonic_ns")
            deadline_us = integer(grant, "deadline_us")
            issue_low, issue_high = integer(grant, "tpc_low"), integer(
                grant, "tpc_high"
            )
            ack_low, ack_high = integer(ack, "tpc_low"), integer(ack, "tpc_high")
        except ValueError as exc:
            violations.append(f"{key}: {exc}")
            continue
        valid_budget = state in ISSUED_STATES and 1 <= issued_tokens <= 8
        if state not in ISSUED_STATES:
            violations.append(f"{key}: unsupported issued state {state!r}")
        if not 1 <= issued_tokens <= 8:
            violations.append(
                f"{key}: issued token budget {issued_tokens} is outside 1..8"
            )
        if state == "SLACK_FILL" and issued_tokens != 1:
            violations.append(f"{key}: SLACK_FILL issued budget must equal one token")
            valid_budget = False
        if not 0 <= ack_tokens <= issued_tokens:
            violations.append(
                f"{key}: ACK prefix {ack_tokens} exceeds issued budget {issued_tokens} or is negative"
            )
            valid_budget = False

        # Normal completion exhausts the epoch. A partial disposition returns
        # its unused budget; accepting a partial normal ACK would hide a lost
        # suffix or an incorrect Target/Drafter token counter.
        valid_disposition = True
        if ack_state in {"", state}:
            if ack_tokens != issued_tokens:
                violations.append(
                    f"{key}: normal completion ACK did not exhaust its issued budget"
                )
                valid_disposition = False
        elif ack_state in PARTIAL_DISPOSITIONS:
            if not 0 <= ack_tokens < issued_tokens:
                violations.append(
                    f"{key}: {ack_state} ACK must describe an unexhausted prefix"
                )
                valid_disposition = False
        elif ack_state == "PREFILL_DEFERRED":
            if state != "SLACK_FILL" or ack_tokens != 0:
                violations.append(
                    f"{key}: PREFILL_DEFERRED requires a zero-token SLACK_FILL ACK"
                )
                valid_disposition = False
        elif ack_state == "PREFILL_COMPLETE":
            if state != "DRAFT_CATCHUP" or ack_tokens != 1:
                violations.append(
                    f"{key}: PREFILL_COMPLETE requires one completed catchup token"
                )
                valid_disposition = False
        else:
            violations.append(f"{key}: ACK state does not correspond to issued state")
            valid_disposition = False

        if issue_low < 0 or issue_high <= issue_low:
            violations.append(f"{key}: issued TPC range is invalid")
        if (ack_low, ack_high) != (issue_low, issue_high):
            violations.append(f"{key}: ACK TPC range differs from issued range")
        if min(issue_monotonic_ns, ack_monotonic_ns, deadline_us) < 0:
            violations.append(f"{key}: monotonic timestamp/deadline cannot be negative")
        if deadline_us:
            if not issue_monotonic_ns:
                violations.append(
                    f"{key}: grant deadline cannot be checked without monotonic_ns"
                )
            elif issue_monotonic_ns // 1000 >= deadline_us:
                violations.append(f"{key}: grant was issued after its launch deadline")
        if (
            issue_monotonic_ns
            and ack_monotonic_ns
            and ack_monotonic_ns < issue_monotonic_ns
        ):
            violations.append(f"{key}: ACK monotonic timestamp precedes issuance")
        if valid_budget:
            granted_total += issued_tokens
        if valid_budget and valid_disposition:
            acked_total += ack_tokens
            unused_total += issued_tokens - ack_tokens
        if state == "SLACK_FILL":
            slack_issued += 1
            width = issue_high - issue_low
            if width != expected_tpcs:
                violations.append(
                    f"{key}: SLACK_FILL used {width} TPCs, expected {expected_tpcs}"
                )
            if valid_budget and valid_disposition:
                if ack_tokens == 1:
                    slack_success += 1
                elif ack_state in {"EXPIRED", "PREFILL_DEFERRED"}:
                    expired_acks += 1
        elif state == "DRAFT_CATCHUP" and valid_budget:
            catchup_multi += int(issued_tokens > 1)
            if valid_disposition:
                catchup_tokens += ack_tokens

    if require_slack_fill and (slack_issued == 0 or slack_success == 0):
        violations.append(
            "no successfully ACKed one-token SLACK_FILL grant was observed"
        )
    return {
        "status": "FAIL" if violations else "PASS",
        "event_rows": len(rows),
        "grant_keys": len(keyed),
        "states": dict(states),
        "slack_fill_issued": slack_issued,
        "slack_fill_success": slack_success,
        "slack_fill_expired_or_deferred": expired_acks,
        "issued_tokens": granted_total,
        "acked_tokens": acked_total,
        "returned_unused_tokens": unused_total,
        "catchup_multi_token_grants": catchup_multi,
        "catchup_completed_tokens": catchup_tokens,
        "terminal_ack_states": dict(terminal_states),
        "violations": violations,
        "deadline_semantics": (
            "The recorded monotonic deadline is a latest-launch deadline. This analyzer rejects "
            "already-expired issuance; Drafter-side pre-launch enforcement remains covered "
            "by strict runtime invariants and unit tests. ACK completion may occur later."
        ),
        "ack_semantics": (
            "Exactly one issued event and one terminal ACK per request/spec_cnt/epoch; "
            "SLACK_FILL budget is one token; DRAFT_CATCHUP budget is 1..8 tokens. "
            "ACK tokens count the completed prefix and terminal dispositions return unused budget."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tpcs", type=int, required=True)
    parser.add_argument("--require-slack-fill", action="store_true")
    args = parser.parse_args()
    with args.events.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = analyze_rows(
        rows,
        expected_tpcs=args.expected_tpcs,
        require_slack_fill=args.require_slack_fill,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
