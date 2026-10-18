"""Benchmark refactor equivalence and interruption tests, without paper edits."""
import json
from fractions import Fraction
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark_api import benchmark
from pht_experiment import solve_instance, replay_menu
from timed_solver.cores import BudgetExpired, optimize_joint, optimize_capacity


def make_instance(q=4, seed=1):
    rng = random.Random(seed)
    alpha = sorted(rng.sample(range(1, 31), q))
    pairs = rng.sample([(cap, theta) for cap in range(q + 1) for theta in range(1, 7)], min(8, (q + 1) * 6))
    return {'schema_version': 'pht-experiment-v1', 'instance_id': f'timing-test-{seed}',
            'products': [{'q': u + 1, 'alpha': str(alpha[u]),
                          'activation_cost': str(rng.randint(0, 12)),
                          'unit_service_cost': str(Fraction(rng.randint(0, 9), 4))} for u in range(q)],
            'buyers': [{'type_id': str(k), 'access_cap': cap, 'theta': str(theta), 'count': rng.randint(1, 5)}
                       for k, (cap, theta) in enumerate(pairs)], 'metadata': {}}


class BenchmarkTests(unittest.TestCase):
    def test_all_methods_exactly_match_original_interface(self):
        for seed in range(10):
            payload = make_instance(2 + seed % 4, seed)
            q = len(payload['products'])
            for method in ('joint', 'G0', 'G1', 'G2', 'fixed', 'capacity', 'contiguous'):
                options = {'fixed_products': list(range(1, q + 1))} if method == 'fixed' else {}
                original = solve_instance(payload, method, **options)
                actual = benchmark(payload, method, budget_seconds=None, **options)
                self.assertEqual(actual['menu'], original['menu'], (seed, method))
                self.assertEqual(actual['accounting'], original['accounting'])
                self.assertEqual(actual['buyer_choices'], original['buyer_choices'])
                self.assertTrue(actual['replay_verified'])
                if method == 'capacity':
                    self.assertEqual(actual['frontier'], original['frontier'])
                phase = actual['phase_seconds']
                self.assertAlmostEqual(sum(phase[k] for k in ('parse_validation', 'preprocess_optimization', 'recovery', 'replay')), phase['total'], places=10)
                self.assertLessEqual(actual['finish_optimization_seconds'], phase['preprocess_optimization'])
                self.assertEqual(actual['budget_exceeded_overhead_seconds'], 0)
                json.dumps(actual, allow_nan=False)

    def test_timeout_returns_feasible_replayed_incumbent(self):
        payload = make_instance(6, 44)
        for method in ('joint', 'G0', 'G1', 'G2', 'fixed', 'capacity', 'contiguous'):
            options = {'fixed_products': [1, 3, 6]} if method == 'fixed' else {}
            actual = benchmark(payload, method, budget_seconds=0, **options)
            self.assertEqual(actual['status'], 'timeout_with_incumbent')
            replay = replay_menu(payload, actual['menu']['products'], actual['menu']['prices'])
            self.assertEqual(actual['accounting'], replay['accounting'])
            self.assertTrue(actual['has_incumbent'])
            self.assertTrue(actual['budget_exhausted'])
            if method == 'fixed':
                self.assertEqual(actual['menu']['products'], [1, 3, 6])
            else:
                self.assertGreaterEqual(Fraction(actual['net_profit']), 0)

    def test_interrupted_threshold_layer_recovery(self):
        from pht_experiment import validate_instance
        data = validate_instance(make_instance(6, 15))
        args = (data['alpha'], data['activation'], data['service'], data['buyers'])
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            if calls > 30:
                raise BudgetExpired()
        state = optimize_joint(*args, recover=True, checkpoint=stop)
        self.assertFalse(state['complete'])
        menu = state['recover']()
        check = replay_menu(make_instance(6, 15), menu['selected'], {str(k): str(v) for k, v in menu['prices'].items()})
        self.assertEqual(Fraction(check['accounting']['net_profit']), state['optimal_profit'])
        calls = 0
        state = optimize_capacity(*args, L=4, checkpoint=stop)
        self.assertFalse(state['complete'])
        _, menus = state['recover']()
        for h, (value, menu) in enumerate(zip(state['frontier'], menus)):
            self.assertLessEqual(len(menu['selected']), h)
            check = replay_menu(make_instance(6, 15), menu['selected'], {str(k): str(v) for k, v in menu['prices'].items()})
            self.assertEqual(Fraction(check['accounting']['net_profit']), value)

    def test_capacity_zero_and_fixed_zero_demand(self):
        payload = make_instance(3, 55)
        for buyer in payload['buyers']:
            buyer['access_cap'] = 0
        # Aggregating types remains the input builder's job.
        aggregate = {}
        for buyer in payload['buyers']:
            aggregate[buyer['theta']] = aggregate.get(buyer['theta'], 0) + buyer['count']
        payload['buyers'] = [{'type_id': str(i), 'access_cap': 0, 'theta': theta, 'count': n}
                             for i, (theta, n) in enumerate(aggregate.items())]
        for method in ('joint', 'G0', 'G1', 'G2', 'capacity'):
            actual = benchmark(payload, method, budget_seconds=None)
            self.assertEqual(actual['net_profit'], '0')
        fixed = benchmark(payload, 'fixed', budget_seconds=None, fixed_products=[1, 2, 3])
        self.assertEqual(Fraction(fixed['net_profit']), -sum(Fraction(p['activation_cost']) for p in payload['products']))
        cap = benchmark(payload, 'capacity', budget_seconds=None, capacity=0)
        self.assertEqual(len(cap['frontier']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
