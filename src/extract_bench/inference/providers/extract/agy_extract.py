"""Agentic extraction through Google Antigravity CLI (`agy`) in Gemini API-key mode.

Same harness shape, prompt and loopback recorder as ``gemini_cli_extract``: the
document is staged in a temp workdir, the agent reads it with its own tools and
writes ``./output.json``. Only the CLI differs:

- ``modelProvider: "gemini"`` (passed with ``--settings``) selects API-key auth,
  so no Google sign-in is needed. ``HOME`` points at a private directory because
  agy keeps its state under ``~/.gemini``.
- The prompt goes over stdin as one ``--input-format stream-json`` message; text
  print mode only accepts the prompt on argv, where a large schema would not fit.
- agy's model IDs carry the thinking level (``gemini-4-hydrogen-high``); its
  requests reach the recorder as the base model, which prices them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.extract.cli_harness import HARNESS_VERSION, CliHarness
from extract_bench.inference.providers.extract.gemini_cli_extract import (
    _MAX_SHELL_CMD_CHARS,
    GeminiCliExtractProvider,
)
from extract_bench.inference.providers.registry import register_provider

_AGY_SETTINGS = {"modelProvider": "gemini"}
_SHELL_TOOL = "run_command"
_DONE_STATE = "DONE"


@register_provider("agy_extract")
class AgyExtractProvider(GeminiCliExtractProvider):
    """Agentic extraction by shelling out to Antigravity CLI (`agy`)."""

    DEFAULT_MODEL = "gemini-4-hydrogen-high"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._gemini_bin = self.base_config.get("agy_bin", "agy")
        self._harness = CliHarness("agy_extract", self._gemini_bin, self.base_config.get(HARNESS_VERSION))

    def _build_settings(self) -> dict[str, Any]:
        return dict(_AGY_SETTINGS)

    def _build_cmd(self) -> list[str]:
        # `--print=` (empty) with stream-json input: the prompt arrives on stdin.
        return [
            self._gemini_bin,
            "--model",
            self._model,
            "--settings",
            json.dumps(self._build_settings()),
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--print=",
        ]

    def _cli_env(self, home: Path, base_url: str) -> dict[str, str]:
        return {
            "HOME": str(home),
            "TERM": "dumb",
            "GEMINI_API_KEY": "local-recorder",
            "GOOGLE_GEMINI_BASE_URL": base_url,
        }

    def _stdin_payload(self, prompt: str) -> str:
        message = {"event": "user", "message": {"role": "user", "content": prompt}}
        return json.dumps(message) + "\n"

    def _prepare_home(self, home: Path) -> None:
        home.mkdir(parents=True)

    @staticmethod
    def _result_event(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        """agy's ``{"event": "result", "result": {"status": "SUCCESS", ...}}`` in Gemini CLI's shape."""
        final = next((e for e in reversed(events) if e.get("event") == "result"), None)
        if final is None or not isinstance(final.get("result"), dict):
            return None
        result = final["result"]
        return {
            "type": "result",
            "status": str(result.get("status") or "").lower(),
            "error": result.get("error") or result.get("finished_reason"),
            "stats": {"usage": result.get("usage"), "num_turns": result.get("num_turns")},
        }

    @staticmethod
    def _tool_usage(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Finished tool steps (never their outputs) and every shell command, capped in length."""
        tool_counts: dict[str, int] = {}
        shell_commands: list[str] = []
        failed = 0
        for event in events:
            step = event.get("step_update")
            # Each tool step is emitted once while ACTIVE and again when it ends; count the end.
            if not isinstance(step, dict) or step.get("step_type") != "tool" or step.get("state") == "ACTIVE":
                continue
            if step.get("state") != _DONE_STATE:
                failed += 1
            name = str(step.get("tool_name") or "")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if name == _SHELL_TOOL:
                command = str(((step.get("tool_info") or {}).get("parameters") or {}).get("CommandLine") or "")
                if len(command) > _MAX_SHELL_CMD_CHARS:
                    command = command[:_MAX_SHELL_CMD_CHARS] + f"...[+{len(command) - _MAX_SHELL_CMD_CHARS} chars]"
                shell_commands.append(command)
        return {"tool_counts": tool_counts, "shell_commands": shell_commands, "failed_tool_calls": failed}
