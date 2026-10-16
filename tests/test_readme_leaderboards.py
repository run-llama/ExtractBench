"""The checked-in tables must reflect comparable overall scores and CSV edits."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_readme.py"
spec = importlib.util.spec_from_file_location("update_readme", SCRIPT)
assert spec is not None and spec.loader is not None
leaderboards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaderboards)


def pro_row(provider, perfect, value, cost, *, challenge="all", box="", page=""):
    return {
        "Provider": provider,
        "Category": "Coding Agents",
        "Challenge": challenge,
        "Perfect_Documents": perfect,
        "Value_F1": value,
        "Word_Grounding": box,
        "Page_Grounding": page,
        "Cost_Per_Page": cost,
    }


def test_challenge_scores_cannot_enter_overall_ranking_and_ties_use_value_then_cost():
    rows = [
        pro_row("Unpriced", "50", "80", ""),
        pro_row("Priced", "50", "80", "0.02"),
        pro_row("Lower value", "50", "79", "0.001"),
        pro_row("Fewer perfect", "49", "99", "0.001"),
        pro_row("Challenge only", "100", "100", "0.001", challenge="house_vocabulary"),
    ]
    rendered = leaderboards.pro_table(rows)
    assert len({line.count("|") for line in rendered.splitlines()}) == 1
    names = [line.split(" | ")[1] for line in rendered.splitlines()[2:]]
    assert names == ["Priced", "Unpriced", "Lower value", "Fewer perfect"]
    assert "2.00¢" in rendered


def test_unreported_grounding_stays_missing_and_does_not_rank_above_reported_zero():
    rendered = leaderboards.pro_table(
        [
            pro_row("Unknown", "0", "70", "", box="", page=""),
            pro_row("Zero", "0", "60", "0", box="0", page="0"),
        ]
    )
    unknown, zero = rendered.splitlines()[2:]
    assert "| — | — | — |" in unknown
    assert "| **0** | **0** | 0.00¢ |" in zero


def test_check_detects_stale_tables_without_rewriting_then_passes_after_generation(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    skeleton = "\n".join(
        f"<!-- {name}:START -->\nstale\n<!-- {name}:END -->" for name in ("LEADERBOARD", "GROUNDING", "PRO_LEADERBOARD")
    )
    readme.write_text(skeleton)
    monkeypatch.setattr(leaderboards, "README_PATH", readme)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--check"])
    with pytest.raises(SystemExit, match="tables are stale"):
        leaderboards.main()
    assert readme.read_text() == skeleton

    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    leaderboards.main()
    generated = readme.read_text()
    assert generated != skeleton
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--check"])
    leaderboards.main()
    assert readme.read_text() == generated
