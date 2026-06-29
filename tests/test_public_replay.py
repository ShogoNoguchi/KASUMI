from pathlib import Path
from civic_policy_scientist.pipeline import run
from civic_policy_scientist.verify import verify_public_bundle


def test_public_bundle_verifies():
    result = verify_public_bundle(Path('artifacts/evidence'))
    assert result['selected_run'] == 'run_3'
    assert result['holdout_cells'] == 3


def test_replay_pipeline(tmp_path):
    result = run(Path('artifacts/evidence'), tmp_path)
    assert Path(result['report']).is_file()
    assert (tmp_path / 'figures' / 'development_candidate_primary_delta.png').is_file()
    assert (tmp_path / 'figures' / 'holdout_primary_delta_by_seed.png').is_file()
    assert (tmp_path / 'figures' / 'primary_welfare_vs_service_loss.png').is_file()
