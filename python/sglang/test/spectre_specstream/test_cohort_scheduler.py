from types import SimpleNamespace

from sglang.srt.speculative.spectre.specstream.cohort_scheduler import (
    StreamWorkItem,
    build_cohort_plans,
)


class _Shape:
    dtype = "bf16"
    shape = (2048, 2, 8, 128)


def _item(rid, *, layer=0, q=4, deadline=10_000):
    return StreamWorkItem(
        rid,
        1,
        layer,
        q,
        0,
        0,
        2048,
        1024,
        deadline,
        SimpleNamespace(tensor=_Shape()),
    )


def test_only_compatible_work_is_cohorted():
    plans = build_cohort_plans(
        [_item("a"), _item("b"), _item("c", q=8)],
        max_cohort_size=8,
        num_staging_slots=2,
        max_cohort_delay_us=200,
        now_ns=1,
    )
    assert sorted(len(plan.items) for plan in plans) == [1, 2]


def test_expired_work_bypasses_cohort_wait():
    plans = build_cohort_plans(
        [_item("a", deadline=1), _item("b", deadline=1)],
        max_cohort_size=8,
        num_staging_slots=2,
        max_cohort_delay_us=200,
        now_ns=2,
    )
    assert [len(plan.items) for plan in plans] == [1, 1]


def test_adjacent_chunk_groups_share_one_cohort_plan():
    first = _item("a")
    second = _item("b")
    first = StreamWorkItem(
        **{
            **first.__dict__,
            "token_end": 4096,
            "cpu_descriptor": (first.cpu_descriptor, first.cpu_descriptor),
        }
    )
    second = StreamWorkItem(
        **{
            **second.__dict__,
            "token_end": 4096,
            "cpu_descriptor": (second.cpu_descriptor, second.cpu_descriptor),
        }
    )
    plans = build_cohort_plans(
        [first, second],
        max_cohort_size=8,
        num_staging_slots=2,
        max_cohort_delay_us=200,
        now_ns=1,
    )
    assert len(plans) == 1
    assert len(plans[0].items) == 2
    assert plans[0].key.chunk_tokens == 4096
