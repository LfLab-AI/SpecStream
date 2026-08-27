#!/usr/bin/env python3
from __future__ import annotations

import ast
import shutil
import textwrap
from pathlib import Path

REPO = Path.cwd()
MIXIN = REPO / "python/sglang/srt/speculative/spectre/drafter/spectre_draft_scheduler_mixin.py"
TP_WORKER = REPO / "python/sglang/srt/managers/tp_worker.py"
RUNTIME_DST = REPO / "python/sglang/srt/speculative/spectre/specstream/draft_grant_runtime.py"
RUNTIME_SRC = Path(__file__).resolve().with_name("draft_grant_runtime.py")


def die(msg: str) -> None:
    raise SystemExit("ERROR: " + msg)


def backup(path: Path) -> None:
    dst = path.with_suffix(path.suffix + ".first_grant.bak")
    if not dst.exists():
        shutil.copy2(path, dst)
        print("backup:", dst)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    die(f"class {name} not found in current branch")


def find_method(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    die(f"method {cls.name}.{name} not found in current branch")


def replace_span(lines: list[str], node: ast.AST, new_text: str) -> list[str]:
    start = node.lineno - 1
    end = node.end_lineno
    replacement = textwrap.dedent(new_text).rstrip().splitlines()
    indent = " " * node.col_offset
    replacement = [indent + x if x.strip() else "" for x in replacement]
    return lines[:start] + replacement + lines[end:]


def insert_before_method(lines: list[str], node: ast.AST, text: str) -> list[str]:
    at = node.lineno - 1
    block = textwrap.dedent(text).rstrip().splitlines()
    indent = " " * node.col_offset
    block = [indent + x if x.strip() else "" for x in block]
    return lines[:at] + block + [""] + lines[at:]


def patch_tp_worker() -> None:
    src = TP_WORKER.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(TP_WORKER))
    cls = find_class(tree, "TpModelWorker")

    if "def specstream_apply_draft_tpc_range(" not in src:
        fwd = find_method(cls, "forward_batch_generation")
        lines = src.splitlines()
        helpers = '''
def specstream_begin_controlled_draft_forward(self) -> None:
    self._specstream_controlled_draft_forward = True

def specstream_end_controlled_draft_forward(self) -> None:
    self._specstream_controlled_draft_forward = False

def specstream_draft_mask_active(self) -> bool:
    return getattr(self, "_specstream_active_draft_tpc_range", None) is not None

def specstream_total_tpcs(self):
    for name in (
        "_specstream_draft_smctrl",
        "specstream_draft_smctrl",
        "_specstream_smctrl",
        "specstream_smctrl",
    ):
        ctrl = getattr(self, name, None)
        if ctrl is not None and hasattr(ctrl, "total_tpcs"):
            return int(ctrl.total_tpcs)
    return None

def specstream_apply_draft_tpc_range(self, tpc_low: int, tpc_high: int) -> None:
    low, high = int(tpc_low), int(tpc_high)
    if low < 0 or high <= low:
        raise ValueError(f"invalid Draft TPC range [{low}, {high})")

    ctrl = None
    for name in (
        "_specstream_draft_smctrl",
        "specstream_draft_smctrl",
        "_specstream_smctrl",
        "specstream_smctrl",
    ):
        candidate = getattr(self, name, None)
        if candidate is not None:
            ctrl = candidate
            break
    if ctrl is None:
        raise RuntimeError(
            "SpecStream Draft SMController was not initialized in TpModelWorker"
        )

    scope = str(
        getattr(self.server_args, "specstream_smctrl_mask_scope", "global")
        or "global"
    ).lower()

    applied = False
    errors = []

    if scope == "global":
        for name in (
            "set_global_tpc_range",
            "set_global_tpc_mask",
            "set_global_mask",
            "set_tpc_range",
            "set_mask",
        ):
            fn = getattr(ctrl, name, None)
            if fn is None:
                continue
            for args in ((low, high), (range(low, high),)):
                try:
                    fn(*args)
                    applied = True
                    break
                except TypeError as exc:
                    errors.append(f"{name}{args}: {exc}")
            if applied:
                break
    else:
        stream = None
        for name in (
            "specstream_draft_stream",
            "_specstream_draft_stream",
            "specstream_forward_stream",
            "_specstream_forward_stream",
        ):
            candidate = getattr(self, name, None)
            if candidate is not None:
                stream = candidate
                break

        fn = getattr(ctrl, "set_stream_mask", None)
        if fn is not None and stream is not None:
            for args in (
                (stream, low, high),
                (stream, range(low, high)),
            ):
                try:
                    fn(*args)
                    applied = True
                    break
                except TypeError as exc:
                    errors.append(f"set_stream_mask{args}: {exc}")

    if not applied:
        public = [
            x for x in dir(ctrl)
            if "mask" in x.lower() or "tpc" in x.lower()
        ]
        raise RuntimeError(
            "Cannot apply SpecStream Draft TPC mask. "
            f"scope={scope}, controller={type(ctrl).__name__}, "
            f"candidate_methods={public}, attempts={errors}"
        )

    self._specstream_active_draft_tpc_range = (low, high)

def specstream_bootstrap_calibration_mask_if_needed(self) -> None:
    if self.specstream_draft_mask_active():
        return
    if not bool(getattr(self.server_args, "specstream_smctrl_enabled", False)):
        return
    tpcs = int(
        getattr(self.server_args, "specstream_smctrl_calibration_tpcs", 0) or 0
    )
    if tpcs <= 0:
        return
    self.specstream_apply_draft_tpc_range(0, tpcs)
'''
        lines = insert_before_method(lines, fwd, helpers)
        TP_WORKER.write_text("\n".join(lines) + "\n", encoding="utf-8")
        src = TP_WORKER.read_text(encoding="utf-8")

    # Locate the existing safety raise and replace only its enclosing if.
    tree = ast.parse(src, filename=str(TP_WORKER))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    target_raise = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            seg = ast.get_source_segment(src, node) or ""
            if "SpecStream refused an ungranted Draft forward" in seg:
                target_raise = node
                break
    if target_raise is None:
        die(
            "existing 'ungranted Draft forward' guard not found; "
            "refusing to guess current tp_worker layout"
        )

    enclosing = parents.get(target_raise)
    while enclosing is not None and not isinstance(enclosing, ast.If):
        enclosing = parents.get(enclosing)
    if enclosing is None:
        die("cannot locate enclosing if for current tp_worker safety guard")

    lines = src.splitlines()
    new_guard = '''
# Only a scheduler-marked remote Draft forward is grant-controlled.
# Ordinary/local/warmup forwards keep the native SGLang path.
if getattr(self, "_specstream_controlled_draft_forward", False):
    # Fixed-TPC calibration must establish a real spatial mask before the
    # first controlled forward.
    self.specstream_bootstrap_calibration_mask_if_needed()
    if not self.specstream_draft_mask_active():
        raise RuntimeError(
            "SpecStream refused an ungranted Draft forward: "
            "no TPC mask is active"
        )
'''
    lines = replace_span(lines, enclosing, new_guard)
    TP_WORKER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("patched:", TP_WORKER)


def patch_mixin() -> None:
    src = MIXIN.read_text(encoding="utf-8")
    import_line = (
        "from sglang.srt.speculative.spectre.specstream.draft_grant_runtime import (\n"
        "    DraftGrantGate,\n"
        "    call_runtime_first_available,\n"
        "    normalize_external_grant,\n"
        ")\n"
    )
    if "specstream.draft_grant_runtime import" not in src:
        marker = "from sglang.srt.utils import DynamicGradMode, broadcast_pyobj\n"
        if marker not in src:
            die("mixin import anchor not found")
        src = src.replace(marker, marker + import_line, 1)
        MIXIN.write_text(src, encoding="utf-8")

    tree = parse(MIXIN)
    cls = find_class(tree, "SpectreDraftSchedulerMixin")

    if "_specstream_init_first_grant_gate" not in MIXIN.read_text(encoding="utf-8"):
        anchor = find_method(cls, "_get_draft_state")
        lines = MIXIN.read_text(encoding="utf-8").splitlines()
        helpers = '''
def _specstream_smctrl_enabled_for_draft(self) -> bool:
    return bool(getattr(self.server_args, "specstream_smctrl_enabled", False))

def _specstream_calibration_tpcs(self) -> int:
    return int(
        getattr(self.server_args, "specstream_smctrl_calibration_tpcs", 0) or 0
    )

def _specstream_underlying_tp_worker(self):
    worker = getattr(self, "model_worker", None)
    if worker is None:
        raise RuntimeError("SpecStream cannot find Scheduler.model_worker")
    return getattr(worker, "target_worker", worker)

def _specstream_init_first_grant_gate(self) -> None:
    if not self._specstream_smctrl_enabled_for_draft():
        self._specstream_draft_grant_gate = None
        return

    if not bool(getattr(self.server_args, "spectre_draft_priority", False)):
        raise RuntimeError(
            "SpecStream SM control requires --spectre-draft-priority so "
            "one-token grant boundaries cannot be bypassed"
        )

    worker = self._specstream_underlying_tp_worker()
    total_tpcs = None
    fn = getattr(worker, "specstream_total_tpcs", None)
    if callable(fn):
        total_tpcs = fn()

    self._specstream_draft_grant_gate = DraftGrantGate(
        total_tpcs=total_tpcs
    )

    calibration_tpcs = self._specstream_calibration_tpcs()
    if calibration_tpcs > 0:
        grant = self._specstream_draft_grant_gate.install_calibration_grant(
            calibration_tpcs
        )
        self._specstream_apply_grant_to_worker(grant)
        if getattr(self, "tp_rank", 0) == 0:
            logger.info(
                "SpecStream calibration bootstrap grant active before first "
                "Draft forward: epoch=%d tpc=[%d,%d)",
                grant.epoch,
                grant.tpc_low,
                grant.tpc_high,
            )

def _specstream_req_ids(self, batch) -> tuple[str, ...]:
    if batch is None:
        return ()
    if isinstance(batch, (list, tuple)):
        reqs = batch
    else:
        reqs = getattr(batch, "reqs", ())
    return tuple(
        str(r.rid)
        for r in reqs
        if hasattr(r, "rid")
    )

def _specstream_existing_coexec_runtime(self):
    names = (
        "specstream_coexec_runtime",
        "_specstream_coexec_runtime",
        "coexec_runtime",
        "target_grant_runtime",
        "grant_runtime",
    )
    for owner in (self, getattr(self, "model_worker", None)):
        if owner is None:
            continue
        for name in names:
            obj = getattr(owner, name, None)
            if obj is not None:
                return obj
    return None

def _specstream_poll_online_grant(self, batch):
    gate = getattr(self, "_specstream_draft_grant_gate", None)
    if gate is None:
        return None

    if self._specstream_calibration_tpcs() > 0:
        return gate.try_acquire(self._specstream_req_ids(batch))

    runtime = self._specstream_existing_coexec_runtime()
    req_ids = self._specstream_req_ids(batch)
    external = call_runtime_first_available(
        runtime,
        (
            "try_acquire_one_token",
            "try_acquire_draft_grant",
            "acquire_draft_grant",
            "get_active_draft_grant",
            "current_draft_grant",
        ),
        batch=batch,
        request_ids=req_ids,
    )
    grant = normalize_external_grant(external)
    if grant is not None:
        gate.install_online_grant(grant)

    return gate.try_acquire(req_ids)

def _specstream_apply_grant_to_worker(self, grant) -> None:
    worker = self._specstream_underlying_tp_worker()
    fn = getattr(worker, "specstream_apply_draft_tpc_range", None)
    if fn is None:
        raise RuntimeError(
            "TpModelWorker lacks specstream_apply_draft_tpc_range(); "
            "tp_worker patch was not applied"
        )
    fn(int(grant.tpc_low), int(grant.tpc_high))

def _specstream_begin_controlled_forward(self) -> None:
    worker = self._specstream_underlying_tp_worker()
    fn = getattr(worker, "specstream_begin_controlled_draft_forward", None)
    if fn is not None:
        fn()

def _specstream_end_controlled_forward(self) -> None:
    worker = self._specstream_underlying_tp_worker()
    fn = getattr(worker, "specstream_end_controlled_draft_forward", None)
    if fn is not None:
        fn()

def _specstream_finish_one_granted_step(
    self, batch, *, success: bool
) -> None:
    gate = getattr(self, "_specstream_draft_grant_gate", None)
    if gate is None:
        return
    gate.complete_one_step(success=success)

    if self._specstream_calibration_tpcs() > 0:
        return

    runtime = self._specstream_existing_coexec_runtime()
    call_runtime_first_available(
        runtime,
        (
            "complete_draft_step",
            "ack_draft_grant",
            "complete_one_token",
            "mark_draft_step_complete",
        ),
        batch=batch,
        request_ids=self._specstream_req_ids(batch),
    )
'''
        lines = insert_before_method(lines, anchor, helpers)
        MIXIN.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Make _init_draft_components initialize the gate.
    tree = parse(MIXIN)
    cls = find_class(tree, "SpectreDraftSchedulerMixin")
    node = find_method(cls, "_init_draft_components")
    lines = MIXIN.read_text(encoding="utf-8").splitlines()
    current = "\n".join(lines[node.lineno - 1:node.end_lineno])
    if "self._specstream_init_first_grant_gate()" not in current:
        lines.insert(
            node.end_lineno,
            " " * (node.col_offset + 4)
            + "self._specstream_init_first_grant_gate()",
        )
        MIXIN.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Replace priority phase.
    tree = parse(MIXIN)
    cls = find_class(tree, "SpectreDraftSchedulerMixin")
    node = find_method(cls, "_run_draft_priority_phase")
    lines = MIXIN.read_text(encoding="utf-8").splitlines()

    replacement = '''
def _run_draft_priority_phase(self) -> None:
    saved_last_batch = self.last_batch
    self._filter_draft_batch()

    if self.draft_batch.is_empty():
        self.last_batch = saved_last_batch
        return

    if not self._specstream_smctrl_enabled_for_draft():
        remaining_steps = max(
            (
                r.draft_tokens_target
                - (len(r.output_ids) - r.draft_generation_start_len)
            )
            for r in self.draft_batch.reqs
            if not getattr(r, "draft_is_paused", False)
        )
        max_steps = self.server_args.spectre_max_draft_priority_steps
        if max_steps <= 0:
            max_steps = remaining_steps
        steps_taken = max(1, min(remaining_steps, max_steps))

        for _step in range(steps_taken):
            self._filter_draft_batch()
            if self.draft_batch.is_empty():
                break
            if not self.draft_batch.check_decode_mem():
                self._handle_draft_batch_oom()
                break
            self.draft_batch.prepare_for_decode()
            result = self.run_batch(self.draft_batch)
            self._process_draft_decode_result(
                self.draft_batch, result
            )
            self._update_draft_batch_after_decode()

        self.last_batch = saved_last_batch
        return

    # I2: no grant -> no GPU forward.
    grant = self._specstream_poll_online_grant(self.draft_batch)
    if grant is None:
        self.last_batch = saved_last_batch
        return

    if not self.draft_batch.check_decode_mem():
        gate = self._specstream_draft_grant_gate
        if gate is not None and gate.inflight:
            gate.complete_one_step(success=False)
        self._handle_draft_batch_oom()
        self.last_batch = saved_last_batch
        return

    self._specstream_apply_grant_to_worker(grant)
    self.draft_batch.prepare_for_decode()

    success = False
    self._specstream_begin_controlled_forward()
    try:
        result = self.run_batch(self.draft_batch)
        self._process_draft_decode_result(
            self.draft_batch, result
        )
        self._update_draft_batch_after_decode()
        self.draft_forward_cycle += 1
        success = True
    finally:
        self._specstream_end_controlled_forward()
        self._specstream_finish_one_granted_step(
            self.draft_batch, success=success
        )

    self.last_batch = saved_last_batch
'''
    lines = replace_span(lines, node, replacement)
    MIXIN.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Remote Draft prefill/repair is also GPU work. It must not bypass the
    # grant/mask boundary in online I2 mode. Gate it before ScheduleBatch
    # allocation and mark only its run_batch() as controlled.
    tree = parse(MIXIN)
    cls = find_class(tree, "SpectreDraftSchedulerMixin")
    prefill = find_method(cls, "_prefill_draft_reqs")
    all_lines = MIXIN.read_text(encoding="utf-8").splitlines()
    start, end = prefill.lineno - 1, prefill.end_lineno
    block = all_lines[start:end]

    if "_specstream_prefill_grant" not in "\n".join(block):
        init_idx = next((i for i, line in enumerate(block) if "draft_prefill_batch = ScheduleBatch.init_new(" in line), None)
        run_idx = next((i for i, line in enumerate(block) if "result = self.run_batch(draft_prefill_batch)" in line), None)
        if init_idx is None or run_idx is None:
            die("_prefill_draft_reqs anchors changed; refusing unsafe patch")
        base_indent = len(block[init_idx]) - len(block[init_idx].lstrip())
        gate_lines = [
            " " * base_indent + "if self._specstream_smctrl_enabled_for_draft():",
            " " * (base_indent + 4) + "_specstream_prefill_grant = self._specstream_poll_online_grant(admitted)",
            " " * (base_indent + 4) + "if _specstream_prefill_grant is None:",
            " " * (base_indent + 8) + "self.draft_waiting_queue = admitted + self.draft_waiting_queue",
            " " * (base_indent + 8) + "return",
            " " * (base_indent + 4) + "self._specstream_apply_grant_to_worker(_specstream_prefill_grant)",
            " " * base_indent + "else:",
            " " * (base_indent + 4) + "_specstream_prefill_grant = None",
            "",
        ]
        block[init_idx:init_idx] = gate_lines
        run_idx = next((i for i, line in enumerate(block) if "result = self.run_batch(draft_prefill_batch)" in line), None)
        run_indent = len(block[run_idx]) - len(block[run_idx].lstrip())
        original_run = block[run_idx].strip()
        wrapped = [
            " " * run_indent + "if _specstream_prefill_grant is None:",
            " " * (run_indent + 4) + original_run,
            " " * run_indent + "else:",
            " " * (run_indent + 4) + "_specstream_prefill_success = False",
            " " * (run_indent + 4) + "self._specstream_begin_controlled_forward()",
            " " * (run_indent + 4) + "try:",
            " " * (run_indent + 8) + original_run,
            " " * (run_indent + 8) + "_specstream_prefill_success = True",
            " " * (run_indent + 4) + "finally:",
            " " * (run_indent + 8) + "self._specstream_end_controlled_forward()",
            " " * (run_indent + 8) + "self._specstream_finish_one_granted_step(",
            " " * (run_indent + 12) + "draft_prefill_batch, success=_specstream_prefill_success",
            " " * (run_indent + 8) + ")",
        ]
        block[run_idx:run_idx + 1] = wrapped
        all_lines[start:end] = block
        MIXIN.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    print("patched:", MIXIN)


def main() -> None:
    for p in (MIXIN, TP_WORKER):
        if not p.exists():
            die(f"missing {p}")
    if not RUNTIME_SRC.exists():
        die(f"missing {RUNTIME_SRC}")

    backup(MIXIN)
    backup(TP_WORKER)

    RUNTIME_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME_SRC, RUNTIME_DST)
    print("installed:", RUNTIME_DST)

    patch_tp_worker()
    patch_mixin()

    for p in (MIXIN, TP_WORKER, RUNTIME_DST):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
        print("syntax OK:", p)

    print()
    print("Patch applied. Run editable install + tests, then only Gate-3 smoke.")


if __name__ == "__main__":
    main()
