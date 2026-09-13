from types import SimpleNamespace

from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler


def test_staging_source_and_host_metrics_do_not_change_gpu_elapsed_metrics(tmp_path):
    profiler = SpecStreamProfiler(str(tmp_path / "profile.csv"), 0, 1)
    meta = SimpleNamespace(
        round_id=1, mode="ordinary", q_len=4, context_tokens=0, items=(),
    )
    profiler.begin_round(meta)
    transfer = SimpleNamespace(
        source_nbytes=1024, padding_nbytes=0, dma_count=2,
        source_count=8, host_wait_ms=0.0, host_pack_ms=0.0,
        metadata_cache_hit=True,
    )
    profiler.record_h2d(1, 1024, 7.0)
    profiler.record_staging(1, transfer)
    profiler.record_staging(1, SimpleNamespace(
        source_nbytes=512, padding_nbytes=64, dma_count=3,
        source_count=3, host_wait_ms=0.25, host_pack_ms=0.5,
    ))
    row = profiler._active[1]
    assert row.h2d_source_bytes == 1536
    assert row.h2d_padding_bytes == 64
    assert row.h2d_dma_ops == 5 and row.h2d_source_slabs == 11
    assert row.host_slot_wait_ms == 0.25 and row.host_pack_ms == 0.5
    assert row.metadata_cache_hits == 1
    assert row.h2d_bytes == 1024 and row.h2d_ops == 1 and row.h2d_ms == 7.0
    assert row.h2d_event_ms == row.staging_wait_ms == 0.0
    profiler.record_staging(999, transfer)
