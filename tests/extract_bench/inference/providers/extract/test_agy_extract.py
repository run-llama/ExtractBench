"""Contract between the provider and the agy CLI's stream-json stdin/stdout."""

import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("pypdf")

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.extract.agy_extract import AgyExtractProvider
from extract_bench.inference.providers.extract.gemini_cli_trace import GeminiCliTrace

# Mimics agy: one stream-json user message on stdin, then init/step_update/result events.
FAKE_AGY = r"""
import json, os, sys
message = json.loads(sys.stdin.readline())
assert message["event"] == "user" and message["message"]["role"] == "user", message
tool = {"step_type": "tool", "tool_name": "run_command",
        "tool_info": {"parameters": {"CommandLine": "pdftotext input.pdf -"}}}
print(json.dumps({"event": "init", "env": {k: os.environ.get(k) for k in ("HOME", "GEMINI_API_KEY")},
                  "prompt": message["message"]["content"]}))
print(json.dumps({"event": "step_update", "step_update": {**tool, "state": "ACTIVE"}}))
print(json.dumps({"event": "step_update", "step_update": {**tool, "state": "DONE"}}))
print(json.dumps({"event": "result", "result": {"status": "SUCCESS", "num_turns": 1, "usage": {}}}))
"""


def test_agy_prompt_arrives_as_one_stream_json_user_message_and_success_result_passes(
    tmp_path: Path, monkeypatch
) -> None:
    """agy silently ignores an unrecognized stdin event (the run does nothing and exits 0), reports
    status in upper case, and emits each tool step twice; the real Gemini key must stay in the recorder."""
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "user-secret")
    provider = AgyExtractProvider("agy_extract")
    home = tmp_path / "home"
    provider._prepare_home(home)
    lines: deque[str] = deque(maxlen=20)
    tool_usage = {"tool_counts": {}, "shell_commands": [], "failed_tool_calls": 0}
    prompt = 'Extract "fields"\nfrom ./input.pdf'
    code, _ = provider._run_cli(
        [sys.executable, "-c", FAKE_AGY],
        tmp_path,
        home,
        "doc",
        lines,
        prompt,
        base_url="http://127.0.0.1:1",
        trace=GeminiCliTrace(tmp_path / "trace", {"example_id": "doc"}),
        tool_usage=tool_usage,
    )
    events = provider._events(lines)
    init = events[0]
    assert code == 0
    assert init["prompt"] == prompt
    assert init["env"] == {"HOME": str(home), "GEMINI_API_KEY": "local-recorder"}
    provider._raise_for_status(code, provider._result_event(events), "")
    assert tool_usage == {
        "tool_counts": {"run_command": 1},
        "shell_commands": ["pdftotext input.pdf -"],
        "failed_tool_calls": 0,
    }
    assert "user-secret" not in json.dumps(list(lines))


def test_agy_pipeline_uses_api_key_mode_and_the_thinking_level_model_id(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "user-secret")
    pipeline = get_pipeline("agy_extract_gemini_4_hydrogen_high_thinking_vision_evidence")
    provider = AgyExtractProvider("agy_extract", dict(pipeline.config))
    cmd = provider._build_cmd()
    assert cmd[cmd.index("--model") + 1] == "gemini-4-hydrogen-high"
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {"modelProvider": "gemini"}
    assert provider._harness.name == "agy_extract" and provider._harness.executable == "agy"


def test_agy_home_installs_a_hook_that_denies_web_tools(tmp_path: Path, monkeypatch) -> None:
    """agy cannot drop a tool and ignores permissions.deny with --dangerously-skip-permissions, so a
    global PreToolUse hook denies web search, matching Gemini CLI's excluded web tools."""
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "user-secret")
    home = tmp_path / "home"
    AgyExtractProvider("agy_extract")._prepare_home(home)
    (group,) = json.loads((home / ".gemini" / "config" / "hooks.json").read_text())["no-web"]["PreToolUse"]
    matcher = re.compile(group["matcher"])
    assert all(matcher.fullmatch(name) for name in ("search_web", "browser_subagent"))
    assert not any(matcher.fullmatch(name) for name in ("run_command", "read_file", "write_file"))
    (hook,) = group["hooks"]
    output = subprocess.run(["sh", "-c", hook["command"]], capture_output=True, text=True, check=True).stdout
    assert json.loads(output)["decision"] == "deny"
