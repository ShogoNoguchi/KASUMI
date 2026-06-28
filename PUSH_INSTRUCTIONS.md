# Push instructions

Replace `<YOUR_USER_OR_ORG>` and `<REPO_NAME>` with your GitHub account and repository name.

```bash
unzip public_service_policy_lab.zip
cd public_service_policy_lab

python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/audit_public_release.py .
python scripts/check_pr_readiness.py
python scripts/run_replay_pipeline.py --evidence-dir artifacts/evidence --out-dir outputs/replay
pytest -q

git init
git add .
git commit -m "Initial public release of KASUMI"
git branch -M main
git remote add origin git@github.com:<YOUR_USER_OR_ORG>/<REPO_NAME>.git
git push -u origin main
```

Enable GitHub Pages from repository settings:

```text
Settings -> Pages -> Build and deployment -> Deploy from a branch -> main / docs
```

The landing page entry point is `docs/index.html`.
