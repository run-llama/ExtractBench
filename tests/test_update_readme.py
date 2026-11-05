import csv
import runpy
import sys
from pathlib import Path

import pytest


def test_unranked_winner_cannot_take_rank_or_best_score_emphasis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/update_readme.py"))
    main = script["main"]
    score_columns = (
        "Overall",
        "Short",
        "Medium",
        "Long",
        "Perfect_Documents",
        "Value_F1",
        *(
            f"{prefix}{suffix}"
            for prefix in ("Word_Grounding", "Page_Grounding")
            for suffix in ("", "_Short", "_Medium", "_Long")
        ),
    )
    rows = [
        {
            "Provider": name,
            "Category": "Coding Agents",
            "include_in_rankings": included,
            "Challenge": "all",
            "Cost_Per_Page": "0.01",
            **dict.fromkeys(score_columns, score),
        }
        for name, included, score in (("Excluded winner", "false", "100"), ("Ranked system", "true", "90"))
    ]
    for variable, filename in (("CSV_PATH", "leaderboard.csv"), ("PRO_CSV_PATH", "leaderboard-pro.csv")):
        path = tmp_path / filename
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        monkeypatch.setitem(main.__globals__, variable, path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            f"<!-- {name}:START -->\n<!-- {name}:END -->" for name in ("LEADERBOARD", "GROUNDING", "PRO_LEADERBOARD")
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(main.__globals__, "README_PATH", readme)
    monkeypatch.setattr(sys, "argv", ["update_readme.py"])
    main()
    rendered = readme.read_text(encoding="utf-8")
    assert "Excluded winner" not in rendered
    assert rendered.count("Top 1 of 1 ranked systems") == 3
    assert "| 1 | Ranked system | Coding Agents | **90**" in rendered
    assert '<td align="right">1</td><td>Ranked system</td><td align="right"><strong>90</strong>' in rendered
    monkeypatch.setattr(sys, "argv", ["update_readme.py", "--check"])
    main()


@pytest.mark.parametrize("included", ["", "yes"])
def test_malformed_ranking_flag_cannot_silently_include_a_row(tmp_path: Path, included: str) -> None:
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/update_readme.py"))
    path = tmp_path / "leaderboard.csv"
    path.write_text(f"Provider,include_in_rankings\nSystem,{included}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be true or false"):
        script["read_ranked_rows"](path)
