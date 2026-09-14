"""Exercise dataset scoring and successful/failed HTTP evaluation runs."""

import importlib.util
import json
import threading
from argparse import Namespace
from collections import UserDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("specstream_evaluate", ROOT / "scripts/specstream/evaluate.py")
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_scores_and_prompt_formats():
    assert evaluate.score_answer("gsm8k", "We add 20 and 30. #### 50", "50")
    assert evaluate.score_answer("gsm8k", "#### 1,234.00", "1234")
    assert not evaluate.score_answer("gsm8k", "No answer", "0")
    assert evaluate.score_answer("longbench-v2", "Answer: (C).", "C")
    assert not evaluate.score_answer("longbench-v2", "A or C", "C")
    assert not evaluate.score_answer("longbench-v2", "B", "C")
    for output in ([1, 2], UserDict(input_ids=[1, 2])):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return output
        assert evaluate.prompt_ids(Tokenizer(), "text") == [1, 2]


def test_raw_jsonl(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"2+2?","answer":"#### 4"}\n\n', encoding="utf-8")
    rows = evaluate.load_rows("gsm8k", str(path))
    prompt, answer = evaluate.make_prompt("gsm8k", rows[0])
    assert "2+2?" in prompt and answer == "4"


@pytest.mark.parametrize("fail_request", [False, True])
def test_run_counts_errors_and_completion(tmp_path, fail_request):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"model_path":"fixture"}')

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            if fail_request and request["input_ids"] == [3, 4]:
                self.send_error(500)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"text": "#### 4", "meta_info": {
                "finish_reason": {"type": "stop"}, "prompt_tokens": 2,
                "completion_tokens": 3}}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workload = tmp_path / "workload.json"
    evaluate.write_json(workload, {"kind": "gsm8k", "output_tokens": 10, "ignore_eos": False,
                        "samples": [{"id": "a", "input_ids": [1, 2], "reference": "4"},
                                    {"id": "b", "input_ids": [3, 4], "reference": "4"}]})
    output = tmp_path / "run"
    args = Namespace(workload=str(workload), output_dir=str(output),
                     base_url=f"http://127.0.0.1:{server.server_port}",
                     warmup=1, concurrency=2, timeout=10)
    try:
        if fail_request:
            with pytest.raises(SystemExit) as exc:
                evaluate.run(args)
            assert exc.value.code == 1
        else:
            evaluate.run(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    summary = json.loads((output / "summary.json").read_text())
    assert len(calls) == 3
    assert summary["completed"] == (1 if fail_request else 2)
    assert summary["errors"] == int(fail_request)
    assert summary["output_tokens"] == (3 if fail_request else 6)
    assert summary["accuracy"] == (0.5 if fail_request else 1.0)
    assert (output / "complete.marker").exists() == (not fail_request)
