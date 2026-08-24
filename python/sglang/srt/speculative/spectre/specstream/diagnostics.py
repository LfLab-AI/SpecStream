from __future__ import annotations

import json
from pathlib import Path
import threading
import time


class SpecStreamDiagnostics:
    def __init__(self, profile_path: str, enabled: bool) -> None:
        path = Path(profile_path)
        self.path = path.with_name(f"{path.stem}.diagnostics.jsonl")
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self.max_shadow_abs = 0.0

    def record_shadow(
        self, *, rid: str, round_id: int, layer_id: int, candidate, reference
    ) -> float:
        if not self.enabled:
            return 0.0
        delta = (candidate.float() - reference.float()).abs()
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        denom = float(reference.float().norm().item())
        relative_l2 = float(delta.norm().item()) / max(denom, 1e-12)
        self.max_shadow_abs = max(self.max_shadow_abs, max_abs)
        self._write(
            {
                "timestamp": time.time(),
                "kind": "attention_shadow",
                "rid": rid,
                "round_id": round_id,
                "layer_id": layer_id,
                "max_abs": max_abs,
                "relative_l2": relative_l2,
            }
        )
        return max_abs

    def record_logit_margin(self, *, round_id: int, logits) -> float:
        if not self.enabled or logits is None or logits.numel() == 0:
            return 0.0
        top2 = logits.float().topk(k=2, dim=-1).values
        margin = float((top2[..., 0] - top2[..., 1]).min().item())
        self._write(
            {
                "timestamp": time.time(),
                "kind": "logit_margin",
                "round_id": round_id,
                "min_top1_top2_margin": margin,
            }
        )
        return margin

    def _write(self, payload: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
