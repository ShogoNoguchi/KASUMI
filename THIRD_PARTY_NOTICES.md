# Third-party notices

This repository is released under Apache-2.0. It depends on several external
projects at runtime or development time. The repository does not vendor large
upstream research-framework source trees; it packages a project-specific task
extension, compact evidence, and replay utilities.

## Runtime dependencies

- Python: PSF License Agreement.
- Pydantic: MIT License.
- Matplotlib: Matplotlib License.
- PyYAML: MIT License.
- NetworkX: BSD License.

## Development dependencies

- pytest: MIT License.
- ruff: MIT License.
- mypy: MIT License.

## Research frameworks and papers

The workflow design is compatible with AI Scientist-style automated research and
Shachi-style LLM-driven agent-based modeling. This repository includes only the
project-specific task surface and compact compatibility shims needed for public
inspection and artifact replay. Users who perform live LLM experiments should
review and comply with the licenses of their selected full agent and scientist
framework installations.

## Generated artifacts

The PDF, figures, CSVs, and JSON summaries in `paper/` and `artifacts/` are
synthetic research artifacts produced by the project workflow. They are provided
for inspection and reproducible replay under the repository license, subject to
the claim boundary stated in README.md.

## Upstream research-framework integration code

KASUMI can be run in two modes: deterministic public-evidence replay and live
provider-backed E2E reproduction.  The E2E path uses public upstream projects
rather than publishing a private workspace snapshot.

### The AI Scientist v1

The live E2E bootstrap script clones the public AI Scientist repository at a
pinned revision and applies a small KASUMI-specific task-template integration.
This repository includes only the necessary integration surface:

- `integrations/ai_scientist_template/public_service_policy_lab/`: the KASUMI task template;
- `third_party/ai_scientist/policy_context.py`: protected-template context hooks used by the task;
- `third_party/ai_scientist/ai_scientist_v1_policy.patch`: a minimal patch allowing protected policy templates to enrich idea prompts, validate structured ideas, and expose source cards to the AI Scientist loop;
- `scripts/e2e/run_ai_scientist_development.sh`: the public E2E stage that launches candidate development from the packaged KASUMI mechanism portfolio.

The reason for including these files is reproducibility: without the task hooks
and patch, an external user could inspect the final evidence but could not rerun
the AI-Scientist-style candidate-development stage from the public repository.
The full upstream AI Scientist source tree is not vendored; it is cloned by
`scripts/e2e/bootstrap_workspace.sh` from the public repository at the pinned
revision configured by `AI_SCIENTIST_REF`.

### Shachi

The live E2E bootstrap script also clones the public Shachi repository at a
pinned revision and applies the KASUMI public-administration ABM overlay.  This
repository includes the necessary KASUMI-specific Shachi surface:

- `src/shachi/env/japan_policy_scientist/`: synthetic public-service environment, dynamics, metrics, transfer planning, task queues, gates, and validation;
- `src/shachi/agent/japan_policy_bureaucrat.py`: the bounded bureaucratic-agent interface;
- `scripts/e2e/shachi_run_japan_policy_scientist.py` and related gate runners: command-line entrypoints copied into an E2E Shachi workspace.

The reason for including these files is that KASUMI's results depend on a custom
Shachi environment and agent contract.  The full upstream Shachi repository is
not vendored; it is cloned by `scripts/e2e/bootstrap_workspace.sh` from the
public repository at the pinned revision configured by `SHACHI_REF`, and then
this overlay is copied into the workspace.

### What is intentionally not included

The repository does not include API keys, filled-in provider budget files, raw
response caches, failed-attempt logs, private workspace paths, or private
challenge/evaluation wording.  Public E2E scripts require users to provide their
own API credentials and explicitly set `KASUMI_E2E_ALLOW_PROVIDER_CALLS=1` before
any provider-backed stage can run.

