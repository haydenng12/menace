from dataclasses import replace

from menace.experiments import BASELINE, _convergence, _stability, run_trial


def test_convergence_uses_final_performance_floor():
    assert _convergence([100, 200, 300], [0.60, 0.74, 0.76]) == 200


def test_policy_stability_uses_shared_states():
    assert _stability({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == 0.5


def test_seeded_trial_is_reproducible():
    condition = replace(BASELINE, name="test")
    first = run_trial(condition, 0, 7, 40, 10, 20)
    second = run_trial(condition, 0, 7, 40, 10, 20)
    assert first == second


def test_symmetry_ablation_stores_at_least_as_many_states():
    with_symmetry = run_trial(BASELINE, 0, 11, 100, 10, 50)
    without = run_trial(replace(BASELINE, symmetry=False), 0, 11, 100, 10, 50)
    assert without.state_space_size >= with_symmetry.state_space_size
