"""Audit final-paper prose for unverified numerical outcome claims.

current public release deliberately distinguishes two classes of numbers:

1. Allowed identifiers and design constants, such as Gemini 2.5 Flash,
   current public release, run_1, seed_20260701, year 2026, 120 slots, 48 months,
   endpoint weights 0.75/0.25, and table/figure references.
2. Blocked unverified outcome values in prose, such as
   "the selected policy improved welfare by 12.3%" or
   "mean strain was 0.42".

Exact outcome values belong in machine-generated verified_results.tex and the
machine-readable verified_claims.json/claim_verification.json path, not in
free-form manuscript prose.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

# Candidate decimal or percentage tokens. Integers are generally allowed because
# the paper needs to state design constants such as 120 slots, 48 months, four
# interventions, and years. Integer percentages remain candidates.
CANDIDATE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+%?|\d+%)(?![A-Za-z0-9_])"
)

# Numeric contexts that are normally identifiers, design constants, or paper
# mechanics rather than result claims.
ALLOW_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:gemini|gpt|claude|sonnet|llama|deepseek|flash|pro|model|models)\b",
        r"\bV\s*\d+(?:\.\d+)+\b|\bv\s*\d+(?:\.\d+)+\b|\bversion\b|\brelease\b",
        r"\brun[_\s-]*\d+\b|\bseed[_\s-]*\d+\b|\bscenario\b|\bcampaign\b",
        r"\b(?:year|month|months|warm[- ]?up|intervention start|employee slots|slots|arms|runs|interventions|docket|budget|points|policy budget)\b",
        r"\b(?:temperature|max tokens|max concurrency|retry|timeout|rounds|citation rounds)\b",
        r"\b(?:endpoint weight|sealed survey weight|mechanical weight|survey weight|0\.75|0\.25|primary endpoint definition|predeclared|preregistered|design constant)\b",
        r"\b(?:figure|fig\.|table|tab\.|section|appendix|equation|eq\.)\b",
        r"\\(?:ref|cref|citep|citet|cite|label)\b",
        r"\b(?:http|https|doi|arxiv)\b",
        r"\b(?:confidence interval|knowledge cutoff|acceptance rate)\b",
    )
)

# Contexts where a candidate number is likely an unverified result claim.
RESULT_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:improved|improvement|improves|increase[sd]?|decrease[sd]?|reduced|reduction|lowered|higher|worse|better|outperform|benefit|harm|changed|delta|selected[- ]minus[- ]reference)\b",
        r"\b(?:welfare|strain|fatigue|turnover|retention|departure|exit|service harm|service-harm|critical overdue|terminal liability|backlog|completion|quality error|rework|cost[- ]effectiveness|guardrail|primary endpoint|survey composite|mechanical anchor)\b",
        r"\b(?:mean|median|average|rate|ratio|share|score|points|units|percent|percentage|pp|effect size|result|outcome|baseline delta)\b",
    )
)


def _strip_nonprose(tex: str) -> str:
    """Remove blocks where numerical values are not free-form prose."""
    # Bibliography payloads, verified result tables, and common LaTeX mechanics
    # should not be audited as prose.
    tex = re.sub(
        r"\\begin\{filecontents\}\{references\.bib\}.*?\\end\{filecontents\}",
        "",
        tex,
        flags=re.DOTALL,
    )
    tex = re.sub(
        r"\\begin\{(?:table|table\*|tabular|tabularx|figure|figure\*)\}.*?\\end\{(?:table|table\*|tabular|tabularx|figure|figure\*)\}",
        "",
        tex,
        flags=re.DOTALL,
    )
    tex = re.sub(r"\\input\{[^}]*verified_results\.tex\}", "", tex)
    cleaned_lines: list[str] = []
    for line in tex.splitlines():
        line = re.sub(r"(?<!\\)%.*$", "", line)
        if any(
            token in line
            for token in (
                r"\includegraphics",
                r"\parbox",
                r"\usepackage",
                r"\documentclass",
                r"\bibliographystyle",
                r"\bibliography",
                r"\graphicspath",
                r"\label",
            )
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _window(text: str, start: int, end: int, radius: int = 120) -> str:
    return " ".join(text[max(0, start - radius) : min(len(text), end + radius)].split())


def _matches_any(patterns: Iterable[re.Pattern[str]], context: str) -> bool:
    return any(pattern.search(context) for pattern in patterns)


def _is_allowed_design_or_identifier(token: str, context: str) -> bool:
    # Percentages are allowed only in clearly non-result external/reference or
    # design-mechanics contexts. Outcome-like percentages remain blocked below.
    if _matches_any(ALLOW_CONTEXT_PATTERNS, context) and not _matches_any(
        RESULT_CONTEXT_PATTERNS, context
    ):
        return True
    # Version-like substrings can be matched partially (e.g. current public release may yield
    # 4.1 depending on regex position), so look at the context rather than token.
    if re.search(r"\b[Vv]\s*\d+(?:\.\d+)+\b", context):
        return True
    # Run/seed identifiers are mostly integers, but keep this explicit for
    # future-proofing.
    if re.search(r"\b(?:run|seed)[_\s-]*\d+\b", context, flags=re.I):
        return True
    return False


def _is_blocked_outcome_number(token: str, context: str) -> bool:
    # Percentage outcome claims are high risk; block them whenever the local
    # context contains an outcome/result word, even if some design words are also
    # present.
    if token.endswith("%") and _matches_any(RESULT_CONTEXT_PATTERNS, context):
        return True
    # Decimal result claims: require both a candidate result metric/topic and an
    # outcome/comparison/statistical word to avoid blocking identifiers.
    result_hits = sum(int(pattern.search(context) is not None) for pattern in RESULT_CONTEXT_PATTERNS)
    return result_hits >= 2


def audit(tex_path: Path) -> dict:
    text = _strip_nonprose(tex_path.read_text(encoding="utf-8"))
    findings = []
    for match in CANDIDATE_NUMBER.finditer(text):
        token = match.group(0)
        context = _window(text, match.start(), match.end())
        if _is_allowed_design_or_identifier(token, context):
            continue
        if not _is_blocked_outcome_number(token, context):
            continue
        findings.append(
            {
                "token": token,
                "context": context,
            }
        )
    return {
        "passed": not findings,
        "tex_path": str(tex_path.resolve()),
        "finding_count": len(findings),
        "findings": findings,
        "rule": (
            "Unverified decimal or percentage outcome claims in free-form prose are forbidden. "
            "Model names, version/run/seed identifiers, years, table/figure references, and "
            "preregistered design constants are allowed. Exact outcome values must appear in "
            "machine-generated verified_results.tex / verified_claims.json artifacts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.tex)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
