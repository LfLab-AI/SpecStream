import pytest

from sglang.srt.speculative.spectre.specstream.mps_env import read_mps_environment


def test_mps_environment_is_read_without_mutation():
    env = read_mps_environment(
        {
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "80",
            "CUDA_MPS_CLIENT_PRIORITY": "0",
            "CUDA_MPS_SM_PARTITION": "target-partition",
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/specstream-mps",
        }
    )
    assert env.configured
    assert env.active_thread_percentage == 80
    assert env.client_priority == 0
    assert env.sm_partition == "target-partition"


def test_invalid_mps_percentage_is_rejected():
    with pytest.raises(ValueError, match="must be in"):
        read_mps_environment({"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "0"})
