# Full E2E reproduction with provider calls

This document describes the live, provider-backed KASUMI pipeline.  It is
separate from the deterministic public-evidence replay in the README.

The replay path rebuilds reports and figures from checked-in JSON/CSV files.  The
E2E path reruns the research workflow from the task template and Shachi/AI
Scientist code surfaces.

## What the E2E path does

The public E2E pipeline is:

1. clone pinned public upstream revisions of Shachi and The AI Scientist;
2. apply the KASUMI Shachi environment/agent overlay and AI Scientist task template;
3. run a no-API preflight;
4. run the stressed-reference Shachi baseline, using Gemini 2.5 Flash-Lite by default;
5. prepare the packaged KASUMI mechanism-portfolio idea without live idea generation;
6. run the AI Scientist development stage with Gemini 2.5 Pro by default to produce `run_1` through `run_4`;
7. freeze the development-stage selection;
8. run frozen multiseed holdout on seeds `20260631`, `20260637`, and `20260641`;
9. generate the evidence-bound paper; and
10. run automated domain review.

This mirrors the experiment lineage without publishing private challenge wording,
raw workspace paths, response caches, failed attempts, API keys, or provider usage
ledgers.

## Non-goals

The live E2E path is not expected to be bit-identical across time, because LLM
providers can change serving details and stochastic behaviors.  The checked-in
paper/evidence bundle is the final artifact snapshot.  The live path is the
runnable reproduction harness for the methodology and execution contract.

## Requirements

Ubuntu/WSL or Linux with:

- `git`
- `curl`
- `patch`
- `pdflatex` and `bibtex` for final paper generation
- enough provider quota for Shachi calls, AI Scientist development, paper writing,
  and review
- a Gemini API key exported as `GEMINI_API_KEY`

## Commands

```bash
cd /path/to/KASUMI

WORKSPACE="$HOME/kasumi-e2e-workspace"

scripts/e2e/bootstrap_workspace.sh "$WORKSPACE"
scripts/e2e/run_preflight_no_api.sh "$WORKSPACE"

cp configs/operator_budget_plan.template.json "$WORKSPACE/operator_budget_plan.json"
# Edit the copied file locally. Do not commit it.

export GEMINI_API_KEY="..."
export KASUMI_E2E_ALLOW_PROVIDER_CALLS=1

scripts/e2e/run_baseline.sh "$WORKSPACE"
scripts/e2e/prepare_packaged_idea.sh "$WORKSPACE"
scripts/e2e/run_ai_scientist_development.sh "$WORKSPACE"
scripts/e2e/run_selection_and_holdout.sh "$WORKSPACE"
scripts/e2e/run_final_paper.sh "$WORKSPACE"
scripts/e2e/run_domain_review.sh "$WORKSPACE"
```

A single convenience runner is also provided:

```bash
export GEMINI_API_KEY="..."
export KASUMI_E2E_ALLOW_PROVIDER_CALLS=1
scripts/e2e/run_full_e2e.sh "$HOME/kasumi-e2e-workspace"
```

## Default model routing

The defaults are intentionally explicit:

```bash
POLICYLAB_SHACHI_MODEL="gemini/gemini-2.5-flash-lite"
POLICYLAB_AI_SCIENTIST_MODEL="gemini-2.5-pro"
KASUMI_FINAL_PAPER_MODEL="gemini-2.5-pro"
KASUMI_SCIENTIFIC_REVIEW_MODEL="gemini/gemini-2.5-pro"
KASUMI_PUBLIC_ADMIN_REVIEW_MODEL="gemini/gemini-2.5-flash"
```

Override these environment variables before running the relevant stage if needed.

## Why the upstream code is not fully vendored

KASUMI includes the task-specific overlay required for this experiment.  It does
not vendor the full upstream AI Scientist or Shachi repositories.  The bootstrap
script clones pinned public upstream revisions and applies the local overlay.
This keeps the public repository small while preserving the exact integration
surface needed for a live rerun.

## Output locations

The main result manifest is written under:

```text
$WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json
```

The final generated paper and review artifacts are written under the run root
reported in that manifest.
