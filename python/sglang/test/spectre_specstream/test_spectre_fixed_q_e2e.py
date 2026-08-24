import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_RUN_SPECSTREAM_GPU_TESTS") != "1",
    reason="set SGLANG_RUN_SPECSTREAM_GPU_TESTS=1 in a configured SPECTRE deployment",
)


def test_fixed_q_external_harness_gate():
    # The complete launch, workload and acceptance matrix is in the Step-1 doc.
    assert os.environ["SGLANG_RUN_SPECSTREAM_GPU_TESTS"] == "1"
