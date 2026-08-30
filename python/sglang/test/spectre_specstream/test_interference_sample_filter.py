import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts/specstream/smctrl/extract_interference_sample.py"
SPEC = importlib.util.spec_from_file_location("extract_interference_sample", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_row_matches_exact_runtime_shape():
    row = {
        "q": "4",
        "batch_size": "8",
        "context_tokens": "15390",
        "rid": "ignored",
        "draft_tpc_low": "0",
        "draft_tpc_high": "4",
    }

    assert MODULE.row_matches_shape(
        row,
        target_shape="verify_bs8_q4_ctx16k",
        draft_bs=8,
        draft_ctx_bucket="16k",
        draft_tpcs=4,
    )
    assert not MODULE.row_matches_shape(
        row,
        target_shape="verify_bs8_q8_ctx16k",
        draft_bs=8,
        draft_ctx_bucket="16k",
        draft_tpcs=4,
    )
    assert not MODULE.row_matches_shape(
        row,
        target_shape="verify_bs4_q4_ctx16k",
        draft_bs=4,
        draft_ctx_bucket="16k",
        draft_tpcs=4,
    )
    assert not MODULE.row_matches_shape(
        row,
        target_shape="verify_bs8_q4_ctx16k",
        draft_bs=8,
        draft_ctx_bucket="16k",
        draft_tpcs=8,
    )


def test_old_profile_can_infer_batch_size_from_request_ids():
    row = {
        "q": "2",
        "context_tokens": "8193",
        "rid": "a|b|c",
    }

    assert MODULE.row_matches_shape(
        row,
        target_shape="verify_bs3_q2_ctx16k",
        draft_bs=3,
        draft_ctx_bucket="16k",
    )


def test_history_h2d_calibration_filters_out_other_target_phases():
    row = {
        "q": "4",
        "batch_size": "1",
        "context_tokens": "8192",
        "target_phase": "history_h2d",
        "history_len": "4096",
        "h2d_ops": "8",
        "exposed_copy_ms": "1.25",
        "draft_tpc_low": "0",
        "draft_tpc_high": "4",
    }
    kwargs = {
        "target_shape": "verify_bs1_q4_ctx8k",
        "draft_bs": 1,
        "draft_ctx_bucket": "8k",
        "draft_tpcs": 4,
        "target_phase": "history_h2d",
    }
    assert MODULE.row_matches_shape(row, **kwargs)
    row["target_phase"] = "target_forward"
    assert not MODULE.row_matches_shape(row, **kwargs)

    row["target_phase"] = "history_h2d"
    row["exposed_copy_ms"] = "0"
    assert not MODULE.row_matches_shape(row, **kwargs)
