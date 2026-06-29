#!/usr/bin/env python3
"""Prepare the packaged KASUMI mechanism-portfolio idea for AI Scientist v1.

The executed KASUMI run used a fixed mechanism portfolio rather than an open-ended
idea search.  This script makes that explicit by writing `ideas.json` and
`idea_selection.json` from the checked-in `seed_ideas.json` without a provider
call.  The subsequent AI Scientist development stage can then generate and run
candidate policies from this fixed idea.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template_dir")
    args = parser.parse_args()
    base = Path(args.template_dir)
    seeds_path = base / "seed_ideas.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit(f"No packaged seed idea found at {seeds_path}")
    selected = dict(seeds[0])
    selected["novel"] = True
    selected["selection_rationale"] = (
        "Packaged mechanism portfolio used for KASUMI E2E reproduction; "
        "no open-ended idea-generation provider call was made."
    )
    (base / "ideas.json").write_text(json.dumps([selected], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (base / "ideas_all.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (base / "idea_selection.json").write_text(
        json.dumps(
            {
                "mode": "packaged_seed_no_live_idea_generation",
                "selected": selected,
                "archive_count": len(seeds),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared packaged KASUMI idea at {base / 'ideas.json'}")


if __name__ == "__main__":
    main()
