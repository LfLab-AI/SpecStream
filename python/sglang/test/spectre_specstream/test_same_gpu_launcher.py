from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "scripts/specstream/run_qwen25_7b_05b_same_gpu.sh"


def test_qwen25_launcher_places_both_full_models_on_one_physical_gpu_uuid():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in text
    assert text.count('CUDA_VISIBLE_DEVICES="$GPU_UUID"') == 2
    assert text.count("--tp-size 1") == 2
    assert "--spectre-fixed-q-mode parallel" in text
    assert "--specstream-pcie-slack-coexec" in text
    assert "--specstream-coexec-require-mps" in text
    assert 'entry.get("slack_source") in ("history_h2d", "*")' in text
