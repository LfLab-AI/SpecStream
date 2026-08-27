import pytest

from sglang.srt.speculative.spectre.specstream.resource_profile import ResourceProfile


def _profile():
    return ResourceProfile.from_dict(
        {
            "gpu": "A100",
            "draft_model": "draft",
            "target_model": "target",
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "verify_bs8_q5_ctx16k",
                    "draft_bs": 8,
                    "draft_ctx_bucket": "16k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 1.2,
                    "target_slowdown": 0.03,
                },
                {
                    "target_shape": "verify_bs8_q5_ctx16k",
                    "draft_bs": 8,
                    "draft_ctx_bucket": "16k",
                    "draft_tpcs": 8,
                    "draft_step_ms": 0.9,
                    "target_slowdown": 0.08,
                },
            ],
        }
    )


def test_only_measured_safe_entry_is_selected():
    selected = _profile().select_safe(
        target_shape="verify_bs8_q5_ctx16k",
        draft_bs=8,
        draft_ctx_bucket="16k",
        slowdown_budget=0.05,
        slack_us=1500,
        guard_us=200,
    )
    assert selected is not None
    assert selected.draft_tpcs == 4


def test_unobserved_shape_never_interpolates():
    assert (
        _profile().select_safe(
            target_shape="verify_bs4_q5_ctx16k",
            draft_bs=4,
            draft_ctx_bucket="16k",
            slowdown_budget=0.05,
            slack_us=5000,
            guard_us=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("draft_bs", "draft_ctx_bucket"),
    ((4, "16k"), (8, "32k")),
)
def test_exact_shape_metadata_must_be_consistent(draft_bs, draft_ctx_bucket):
    with pytest.raises(ValueError, match="target_shape"):
        ResourceProfile.from_dict(
            {
                "gpu": "A800",
                "draft_model": "draft",
                "target_model": "target",
                "total_tpcs": 54,
                "entries": [
                    {
                        "target_shape": "verify_bs8_q4_ctx16k",
                        "draft_bs": draft_bs,
                        "draft_ctx_bucket": draft_ctx_bucket,
                        "draft_tpcs": 4,
                        "draft_step_ms": 1.0,
                        "target_slowdown": 0.02,
                    }
                ],
            }
        )
