"""Independent small-instance checks for the final experiment interface.

The reference below does not import PHT state definitions, recurrences, price
recovery or menu replay. It enumerates rational intersections of utility
indifference/participation and price-domain hyperplanes, and directly evaluates
the paper's utility rule for every catalog. The tests are finite-case regression
checks, not a new general proof of continuous-price optimality.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark_api import benchmark
import milp_baseline as mb
from pht_experiment import replay_menu, validate_instance


def payload(alpha=(1, 2), costs=(0, 0), service=(0, 0), buyers=((2, 1, 1),)):
    return {
        'schema_version': 'pht-experiment-v1', 'instance_id': 'independent-final-check',
        'products': [dict(q=i+1, alpha=str(a), activation_cost=str(c), unit_service_cost=str(s))
                     for i, (a, c, s) in enumerate(zip(alpha, costs, service))],
        'buyers': [dict(type_id=str(k), access_cap=cap, theta=str(t), count=n)
                   for k, (cap, t, n) in enumerate(buyers)], 'metadata': {}}


def direct_utility_replay(p, catalog, prices):
    """Standalone specification: compare every eligible utility with exit."""
    products = {int(q['q']): q for q in p['products']}
    prices = {int(q): F(v) for q, v in prices.items()}
    if set(prices) != set(catalog) or any(v < 0 for v in prices.values()):
        raise ValueError('Invalid public price vector')
    chosen = []
    profit = -sum((F(products[q]['activation_cost']) for q in catalog), F(0))
    for buyer in p['buyers']:
        candidates = [(F(0), 0)]
        for q in catalog:
            if q <= buyer['access_cap']:
                utility = F(buyer['theta'])*F(products[q]['alpha'])-prices[q]
                candidates.append((utility, q))
        _, q = max(candidates)
        chosen.append(q)
        if q:
            profit += buyer['count']*(prices[q]-F(products[q]['unit_service_cost']))
    return profit, chosen


def rational_intersection(equations, dimension):
    """Fraction Gaussian elimination; return None for dependent hyperplanes."""
    rows = [[F(x) for x in coeff] + [F(rhs)] for coeff, rhs in equations]
    for col in range(dimension):
        pivot = next((j for j in range(col, dimension) if rows[j][col]), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x/divisor for x in rows[col]]
        for j in range(dimension):
            if j != col and rows[j][col]:
                factor = rows[j][col]
                rows[j] = [x-factor*y for x, y in zip(rows[j], rows[col])]
    return tuple(row[-1] for row in rows)


def independent_catalog_values(p):
    """Enumerate menu candidates for small test cases without a PHT oracle."""
    q = len(p['products'])
    alpha = {i['q']: F(i['alpha']) for i in p['products']}
    upper = 1+max((F(k['theta'])*alpha[u] for k in p['buyers']
                   for u in alpha if u <= k['access_cap']), default=F(0))
    values = {(): F(0)}
    for size in range(1, q+1):
        for catalog in combinations(range(1, q+1), size):
            equations = set()
            for i, u in enumerate(catalog):
                basis = tuple(int(j == i) for j in range(size))
                equations.add((basis, F(0)))
                equations.add((basis, upper))
                for buyer in p['buyers']:
                    if u <= buyer['access_cap']:
                        equations.add((basis, F(buyer['theta'])*alpha[u]))
            for i, u in enumerate(catalog):
                for j in range(i+1, size):
                    v = catalog[j]
                    normal = tuple(int(h == j)-int(h == i) for h in range(size))
                    for buyer in p['buyers']:
                        if v <= buyer['access_cap']:
                            equations.add((normal, F(buyer['theta'])*(alpha[v]-alpha[u])))
            seen, best = set(), None
            for selected in combinations(sorted(equations), size):
                point = rational_intersection(selected, size)
                if point is None or point in seen or any(x < 0 or x > upper for x in point):
                    continue
                seen.add(point)
                profit, _ = direct_utility_replay(p, catalog, dict(zip(catalog, point)))
                best = profit if best is None else max(best, profit)
            if best is None:
                raise AssertionError('The price cube must have vertices')
            values[catalog] = best
    return values


class FinalIndependentValidation(unittest.TestCase):
    def test_small_rational_instances_against_utility_hyperplanes(self):
        rng = random.Random(2026101801)
        for case in range(18):
            q = case % 3+1
            alpha = [F(x, 2) for x in sorted(rng.sample(range(1, 10), q))]
            costs = [F(rng.randrange(7), 3) for _ in alpha]
            service = [F(rng.randrange(6), 4) for _ in alpha]
            types = rng.sample([(cap, F(t, 2)) for cap in range(q+1) for t in range(1, 5)], 4)
            buyers = [(cap, t, rng.randrange(1, 4)) for cap, t in types]
            p = payload(alpha, costs, service, buyers)
            independent = independent_catalog_values(p)
            actual = benchmark(p, 'joint', budget_seconds=None)
            self.assertEqual(F(actual['net_profit']), max(independent.values()), case)
            restored, choices = direct_utility_replay(p, actual['menu']['products'], actual['menu']['prices'])
            self.assertEqual(restored, F(actual['net_profit']))
            self.assertEqual(choices, [x['product'] for x in actual['buyer_choices']])
            capacity = benchmark(p, 'capacity', budget_seconds=None)
            for frontier in capacity['frontier']:
                h = frontier['capacity']
                self.assertEqual(F(frontier['net_profit']), max(v for s, v in independent.items() if len(s) <= h))
            full = tuple(range(1, q+1))
            fixed = benchmark(p, 'fixed', fixed_products=full, budget_seconds=None)
            self.assertEqual(F(fixed['net_profit']), independent[full], case)

    def test_highest_quality_and_zero_utility_purchase(self):
        p = payload((1, 2), (3, 7), (0, 1), ((2, 1, 2), (1, 1, 3), (0, 5, 4)))
        menu, prices = [1, 2], {1: '1', 2: '2'}
        value, choices = direct_utility_replay(p, menu, prices)
        self.assertEqual(choices, [2, 1, 0])
        self.assertEqual(value, -5)
        check = replay_menu(p, menu, prices)
        self.assertEqual(value, F(check['accounting']['net_profit']))
        self.assertEqual(choices, [x['product'] for x in check['buyer_choices']])

    def test_empty_and_fixed_zero_demand_costs(self):
        p = payload((1, 2), (3, 7), (1, 1), ((0, 1, 5),))
        self.assertEqual(direct_utility_replay(p, [], {})[0], 0)
        fixed = benchmark(p, 'fixed', fixed_products=[1, 2], budget_seconds=None)
        self.assertEqual(fixed['menu']['products'], [1, 2])
        self.assertEqual(F(fixed['net_profit']), -10)
        self.assertEqual(direct_utility_replay(p, fixed['menu']['products'], fixed['menu']['prices'])[0], -10)
        self.assertEqual(benchmark(p, 'joint', budget_seconds=None)['net_profit'], '0')

    def test_assignment_mismatch_menu_remains_quality_eligible(self):
        p = payload((1, 2), (0, 0), (0, 100), ((2, 1, 1),))
        model = mb._build_model(validate_instance(p), 2)
        vector = np.zeros(len(model['lower']))
        for j in model['y']:
            vector[j] = 1
        vector[model['p'][0]], vector[model['p'][1]] = 1, 2
        vector[model['x'][0, 0]] = vector[model['payment'][0, 0]] = 1
        fake = SimpleNamespace(x=vector, fun=-1, status=0, message='mock weak tie',
                               mip_dual_bound=-1, mip_gap=0, mip_node_count=0)
        with patch.object(mb, 'milp', return_value=fake):
            result = mb.solve_milp_instance(p)
        self.assertTrue(result['quality_comparison_eligible'])
        self.assertTrue(result['has_valid_menu'])
        self.assertFalse(result['assignment_objective_agreement'])
        self.assertFalse(result['original_problem_certificate']['certified_optimal'])
        self.assertTrue(result['solver_certificate']['numerical_optimal_status'])
        self.assertEqual(result['raw_min_incumbent_objective'], -1)
        self.assertEqual(F(result['net_profit']), -98)
        self.assertEqual(direct_utility_replay(p, result['menu']['products'], result['menu']['prices'])[0], -98)

    def test_numeric_assignment_failure_retains_valid_replayed_menu(self):
        p = payload((1,), (2,), (0,), ((1, 1, 1),))
        model = mb._build_model(validate_instance(p), 1)
        vector = np.zeros(len(model['lower']))
        vector[model['y'][0]] = vector[model['x'][0, 0]] = 1
        vector[model['p'][0]] = vector[model['payment'][0, 0]] = 5
        fake = SimpleNamespace(x=vector, fun=-3, status=1, message='mock invalid weak assignment',
                               mip_dual_bound=-7, mip_gap=1, mip_node_count=2)
        with patch.object(mb, 'milp', return_value=fake):
            result = mb.solve_milp_instance(p)
        self.assertTrue(result['numeric_recovery_issue'])
        self.assertTrue(result['has_valid_menu'])
        self.assertTrue(result['quality_comparison_eligible'])
        self.assertEqual(F(result['net_profit']), -2)
        self.assertEqual(result['solver_status'], 'limit')
        self.assertEqual(result['profit_upper_bound'], 7)
        self.assertFalse(result['original_problem_certificate']['certified_optimal'])

    def test_phase_partition_and_no_incumbent_accounting(self):
        fake = SimpleNamespace(x=None, fun=None, status=1, message='limit',
                               mip_dual_bound=-7, mip_gap=None, mip_node_count=0)
        with patch.object(mb, 'milp', return_value=fake):
            result = mb.solve_milp_instance(payload())
        self.assertFalse(result['has_valid_menu'])
        self.assertFalse(result['quality_comparison_eligible'])
        self.assertIsNone(result['net_profit'])
        self.assertIsNone(result['menu'])
        self.assertIsNone(result['accounting'])
        for result in [result, benchmark(payload(), budget_seconds=None)]:
            phases = result['phase_seconds']
            self.assertAlmostEqual(phases['total'], sum(phases[k] for k in
                ['parse_validation', 'preprocess_optimization', 'recovery', 'replay']), places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
