from types import SimpleNamespace

from sglang.srt.speculative.spectre.draft_delivery import (
    finish_normal_decode_bookkeeping,
)


def _req(*, spec_cnt: int, output_ids: list[int]):
    return SimpleNamespace(
        spec_cnt=spec_cnt,
        output_ids=output_ids,
        len_output_ids=0,
        cur_drafts=[99],
        draft_tokens_and_logits={"draft_tokens": [99]},
    )


def test_remote_drafter_keeps_target_round_id_across_q_tokens():
    req = _req(spec_cnt=7, output_ids=[])

    for token in range(5):
        req.output_ids.append(token)
        finish_normal_decode_bookkeeping(
            req=req,
            server_role="draft",
            verified_token=token,
            drafts=None,
        )

    assert req.spec_cnt == 7
    assert req.len_output_ids == 5


def test_target_normal_decode_advances_pipeline_round():
    req = _req(spec_cnt=7, output_ids=[42])

    def apply_drafts(req, *, verified_token, drafts, skip_d0):
        req.cur_drafts = []

    finish_normal_decode_bookkeeping(
        req=req,
        server_role="target",
        verified_token=42,
        drafts=None,
        apply_drafts=apply_drafts,
    )

    assert req.spec_cnt == 8
    assert req.len_output_ids == 1
    assert req.cur_drafts == []
