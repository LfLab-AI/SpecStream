from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
)


def test_reject_and_high_overhead_override_cost_choice():
    controller = IOAwareController((1, 4, 8), switch_threshold=0)
    acceptance = AcceptanceTracker().snapshot((1, 4, 8))
    for override in ("rejected", "high_overhead"):
        values = dict(
            batch_size=8,
            context_tokens=65536,
            history_tokens=60000,
            history_bytes=8_000_000_000,
            num_chunks=32,
        )
        values[override] = True
        decision = controller.choose(
            SpecStreamBatchState(**values), SpecStreamCostProfile(), acceptance
        )
        assert decision.q == 1
        assert decision.mode == "ordinary"
        assert decision.reason == "safety_fallback"
