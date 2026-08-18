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
