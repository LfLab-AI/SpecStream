import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_RUN_SPECSTREAM_TP2_TESTS") != "1",
    reason="set SGLANG_RUN_SPECSTREAM_TP2_TESTS=1 on a two-rank Target",
)


def test_tp2_external_harness_gate():
    # Rank-level CSV/state comparison is specified in all three Chinese docs.
    assert os.environ["SGLANG_RUN_SPECSTREAM_TP2_TESTS"] == "1"
