import pytest

pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.sm_controller import SMController


def _controller(total_tpcs=54):
    controller = object.__new__(SMController)
    controller.total_tpcs = total_tpcs
    controller.mask_scope = "stream"
    return controller


def test_tpc_range_is_half_open_and_bounded_by_device():
    controller = _controller()
    assert controller.validate_range(0, 4) == (0, 4)
    assert controller.validate_range(53, 54) == (53, 54)


@pytest.mark.parametrize("low,high", [(-1, 4), (4, 4), (5, 4), (0, 55)])
def test_invalid_tpc_range_fails_closed(low, high):
    with pytest.raises(ValueError, match="invalid TPC range"):
        _controller().validate_range(low, high)


def test_global_scope_rejects_devices_over_64_tpcs():
    controller = _controller(total_tpcs=65)
    controller.mask_scope = "global"
    with pytest.raises(RuntimeError, match="at most 64 TPCs"):
        controller.set_stream_mask(object(), 0, 4)


def test_global_scope_uses_callback_backend_without_reading_stream_handle():
    class FakeLib:
        installed = None

        @staticmethod
        def libsmctrl_make_mask(result, low, high):
            result._obj.value = 0x1234
            return 0

        def libsmctrl_set_global_mask(self, mask):
            self.installed = int(mask.value)

    controller = _controller()
    controller.mask_scope = "global"
    controller._lib = FakeLib()
    assert controller.set_stream_mask(object(), 0, 4) == 0x1234
    assert controller._lib.installed == 0x1234
