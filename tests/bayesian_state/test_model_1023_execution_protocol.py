"""Equal model-comparison budgets retain each arm's own candidates."""
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_1023_execution_study import confirmation_bank, KAPPA


def test_equal_confirmation_budget_keeps_sticky_estimate_and_shared_anchor():
    modes = ['MIX', 'MAP', 'STICKY']
    screen, refined, anchors = {}, {}, {}
    for index, mode in enumerate(modes):
        key = 129, mode
        screen[key] = [candidate({'base': i + index * 20, KAPPA: 3. if mode == 'STICKY' else 0.},
                                 'screen') for i in range(8)]
        refined[key] = [candidate({'base': i + index * 20 + 10, KAPPA: 6. if mode == 'STICKY' else 0.},
                                  'refine') for i in range(8)]
        anchors[key] = candidate({'base': 100, KAPPA: 0.}, 'anchor')
    banks = confirmation_bank(screen, refined, anchors, [129], modes, quota=12, leaders=3)
    for key, bank in banks.items():
        ids = {r['id'] for r in bank}
        assert len(bank) == len(ids) == 12
        assert {screen[key][0]['id'], refined[key][0]['id'], anchors[key]['id']} <= ids
