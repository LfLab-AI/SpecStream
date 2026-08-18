import json

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.diagnostics import (  # noqa: E402
    SpecStreamDiagnostics,
)


def test_shadow_and_logit_margin_are_persisted(tmp_path):
    diagnostics = SpecStreamDiagnostics(str(tmp_path / "profile.csv"), enabled=True)
    diagnostics.record_shadow(
        rid="r",
        round_id=3,
        layer_id=7,
        candidate=torch.tensor([1.0, 2.0]),
        reference=torch.tensor([1.0, 2.1]),
    )
    diagnostics.record_logit_margin(round_id=3, logits=torch.tensor([[0.0, 2.0, 1.0]]))
    rows = [json.loads(line) for line in diagnostics.path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["attention_shadow", "logit_margin"]
    assert rows[0]["round_id"] == 3 and rows[0]["layer_id"] == 7
