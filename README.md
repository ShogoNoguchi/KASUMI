<p align="center">
  <img alt="Project status" src="https://img.shields.io/badge/status-research%20prototype-informational">
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="code style" src="https://img.shields.io/badge/code%20style-ruff-informational">
  <img alt="type check" src="https://img.shields.io/badge/type%20check-mypy-informational">
  <img alt="dep manager" src="https://img.shields.io/badge/dep_manager-uv-success">
  <img alt="config" src="https://img.shields.io/badge/config-YAML-informational">
</p>

# KASUMI: An Autonomous AI Scientist for Bureaucratic Personnel Policy

**KASUMI** is a synthetic policy-science workflow for studying bureaucratic
personnel policy in a simulated public-service organization. A foundation-model
research agent proposes staffing, transfer, training, and digital-support
programs; a Shachi-style agent-based simulation evaluates organizational
recovery under stress; a frozen multiseed holdout checks the selected policy;
and an AI Scientist-style writing and review stage turns the verified evidence
into an anonymous paper.

> Claim boundary: this repository reports synthetic simulation evidence only. It
> is not a policy recommendation, a real-world causal estimate, a personnel
> system, or a digital twin of any actual ministry or organization.

<!-- HERO IMAGE PLACEHOLDER
Insert a generated hero image here.
Prompt: "A clean scientific dashboard for bureaucratic personnel policy discovery: translucent layers labelled Staffing Ideas, Transfer Design, Agent-Based Bureaucracy Simulation, Multiseed Holdout, Verified Claims, and Automated Review; calm blue-white palette; subtle Kasumigaseki-style civic architecture silhouettes; no company logos; title text 'KASUMI' centered; modern research-lab aesthetic; high resolution 16:9."
-->

## What KASUMI demonstrates

1. **Personnel-policy discovery.** Four staffing, transfer, training, and support interventions are compared against a stressed reference organization.
2. **Guardrail-aware selection.** The selected candidate must improve a preregistered staff-welfare endpoint while passing service and fairness guardrails.
3. **Frozen multiseed holdout.** The selected policy is evaluated against the reference condition on new random seeds without reselection.
4. **Evidence-bound writing.** The final paper is supported by machine-readable claim verification, numeric audit, and automated review artifacts.
5. **Reproducible replay.** The public evidence bundle can be replayed deterministically without making provider calls.

## Headline result

The development comparison selected `capital_deepening_pathway` (`run_3`). It
improved the primary welfare endpoint by `0.005479431872` over the stressed
reference while satisfying all preregistered guardrails. In three frozen holdout
cells, the selected policy improved the primary endpoint in every cell, with a
mean holdout delta of `0.007192530420`; all holdout guardrails passed.

## Repository map

```text
.
├── artifacts/
│   ├── evidence/                 # sanitized public summaries and CSV tables
│   └── figures/                  # selected generated figures
├── docs/                         # GitHub Pages site
├── integrations/
│   ├── ai_scientist_template/     # task-template surface for a future template PR
│   └── shachi_extension/          # notes on the simulation extension surface
├── paper/FINAL_POLICY_PAPER.pdf   # generated preprint
├── scripts/                       # replay, audit, and PR-readiness utilities
├── src/
│   ├── civic_policy_scientist/     # deterministic replay/verification utilities
│   └── shachi/                    # compact task-extension compatibility surface
└── tests/                         # public release tests
```

## Quickstart: replay the public evidence bundle

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/run_replay_pipeline.py --evidence-dir artifacts/evidence --out-dir outputs/replay
python scripts/audit_public_release.py .
pytest -q
```

The replay command rebuilds a small report and two summary figures from the
public evidence bundle. It does not call any LLM provider.

## Full workflow concept

The full research workflow has five logical stages:

```text
1. Baseline simulation
2. Candidate policy generation
3. Development evaluation of run_1..run_4 against run_0
4. Frozen multiseed holdout of the selected policy
5. Evidence-bound paper writing and automated review
```

This repository packages the custom public-service environment, the task-template
surface, and the final evidence needed to inspect the workflow. The included
public replay is deterministic; live full runs require a compatible LLM provider
and a full agent framework installation.

## Main paper and evidence

The generated paper is available at [`paper/FINAL_POLICY_PAPER.pdf`](paper/FINAL_POLICY_PAPER.pdf).
The most compact evidence files are:

- [`artifacts/evidence/development_selection_summary.json`](artifacts/evidence/development_selection_summary.json)
- [`artifacts/evidence/multiseed_holdout_summary.json`](artifacts/evidence/multiseed_holdout_summary.json)
- [`artifacts/evidence/verification_summary.json`](artifacts/evidence/verification_summary.json)
- [`artifacts/evidence/automated_reviews_public.json`](artifacts/evidence/automated_reviews_public.json)

## Publishing to GitHub

See [`PUSH_INSTRUCTIONS.md`](PUSH_INSTRUCTIONS.md) for fresh-unzip validation, initial commit, remote setup, push, and GitHub Pages configuration.

## Preparing future upstream contributions

This repo is organized so that future work can split into two narrow contribution surfaces:

- `integrations/ai_scientist_template/public_service_policy_lab/`: a template-style task directory for automated research workflows.
- `src/shachi/env/japan_policy_scientist/` and `src/shachi/agent/`: an agent/environment extension surface for LLM-driven ABM.

Run the PR-readiness check:

```bash
python scripts/check_pr_readiness.py
```

The current public repository is intentionally focused on the standalone project
release. It does not attempt to open an upstream PR automatically.

## Image slots for the public page

The README and Pages site are designed with one hero image slot. Use this prompt:

> A clean scientific dashboard for bureaucratic personnel policy discovery:
> translucent layers labelled Staffing Ideas, Transfer Design, Agent-Based
> Bureaucracy Simulation, Multiseed Holdout, Verified Claims, and Automated
> Review; calm blue-white palette; subtle Kasumigaseki-style civic architecture
> silhouettes; no company logos; title text "KASUMI" centered; modern
> research-lab aesthetic; high resolution 16:9.

## Limitations

The simulation is synthetic. Agent self-reports are model outputs. Parameters are
not calibrated to administrative microdata. The holdout checks robustness inside
the same model family, not external validity. The results are useful as a
methodological demonstration and mechanism exploration tool, not as operational
policy advice.
