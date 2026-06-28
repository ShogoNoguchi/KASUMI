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
