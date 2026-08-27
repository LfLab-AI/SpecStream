from sglang.srt.speculative.spectre.specstream.tpc_partition import (
    build_complementary_tpc_partition,
)


def test_single_prefix_partition():
    p = build_complementary_tpc_partition(54, [(0, 4)])
    assert p is not None
    assert (p.draft_low, p.draft_high) == (0, 4)
    assert (p.target_low, p.target_high) == (4, 54)
    assert p.draft_tpcs == 4
    assert p.target_tpcs == 50


def test_multiple_grants_use_conservative_union():
    p = build_complementary_tpc_partition(54, [(0, 4), (0, 8), (0, 6)])
    assert p is not None
    assert (p.draft_low, p.draft_high) == (0, 8)
    assert (p.target_low, p.target_high) == (8, 54)


def test_no_active_grant_means_no_partition():
    assert build_complementary_tpc_partition(54, []) is None


def test_non_prefix_grant_is_rejected():
    try:
        build_complementary_tpc_partition(54, [(4, 8)])
    except ValueError as exc:
        assert "prefix" in str(exc)
    else:
        raise AssertionError("non-prefix Draft grant should fail closed")


def test_draft_cannot_consume_all_tpcs():
    try:
        build_complementary_tpc_partition(54, [(0, 54)])
    except ValueError as exc:
        assert "all physical TPCs" in str(exc)
    else:
        raise AssertionError("all-TPC Draft grant should fail closed")
