# AI Scientist-style task template

This directory contains the project-specific task template used for synthetic
public-service policy discovery. It is intentionally packaged as a standalone
extension surface: `experiment.py`, `plot.py`, `selection_and_holdout.py`,
`claim_verifier.py`, `verified_results.py`, `paper_numeric_audit.py`, and
`review_policy_paper.py` are the files to inspect when preparing a minimal
upstream template contribution.

A compatible scientist runner should copy `public_service_policy_lab/` into its
`templates/` directory and make the packaged `src/shachi` extension importable.
