"""Small deterministic checks of the exact experiment adapter, not a benchmark."""
import copy
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pht_experiment import load_instance, replay_menu, solve_instance, validate_instance


def instance(alpha=(1, 2, 4), costs=(1, 2, 4), services=('1/5', '2/5', '4/5'),
             buyers=((1, 3, 2), (2, 1, 3), (3, 2, 4), (3, 4, 1))):
    return {'schema_version': 'pht-experiment-v1', 'instance_id': 'interface-check',
            'products': [{'q': i + 1, 'alpha': str(a), 'activation_cost': str(c),
                          'unit_service_cost': str(s)}
                         for i, (a, c, s) in enumerate(zip(alpha, costs, services))],
            'buyers': [{'type_id': f'k{i + 1}', 'access_cap': cap, 'theta': str(theta),
                        'count': count} for i, (cap, theta, count) in enumerate(buyers)],
            'metadata': {'purpose': 'interface correctness check'}}


def value(result):
    return Fraction(result['accounting']['net_profit'])


class InterfaceTests(unittest.TestCase):
    def test_exact_validation(self):
        base = instance()
        self.assertEqual(validate_instance(base)['parameters'], {'Q': 3, 'K': 4, 'B': 3, 'T': 5})
        for category, index, field, bad in [
            ('products', 0, 'alpha', 1.0), ('products', 0, 'alpha', True),
            ('products', 0, 'activation_cost', '-1'), ('products', 0, 'alpha', '1/0'),
            ('products', 0, 'q', True), ('buyers', 0, 'count', 1.0),
            ('buyers', 0, 'count', True), ('buyers', 0, 'count', 0),
            ('buyers', 0, 'theta', '0'), ('buyers', 0, 'access_cap', 4),
            ('buyers', 0, 'access_cap', False)]:
            payload = copy.deepcopy(base)
            payload[category][index][field] = bad
            with self.subTest(category=category, field=field, bad=bad), self.assertRaises(ValueError):
                validate_instance(payload)
        duplicate = copy.deepcopy(base)
        duplicate['buyers'].append({**duplicate['buyers'][0], 'type_id': 'another-id'})
        with self.assertRaises(ValueError):
            validate_instance(duplicate)
        equal_alpha = copy.deepcopy(base)
        equal_alpha['products'][1]['alpha'] = '1'
        with self.assertRaises(ValueError):
            validate_instance(equal_alpha)

    def test_choice_ties_and_exit(self):
        payload = instance(alpha=(1, 2), costs=(0, 0), services=(0, 0),
                           buyers=((2, 1, 2), (1, 1, 3), (0, 5, 1)))
        replay = replay_menu(payload, [1, 2], {'1': '1', '2': '2'})
        self.assertEqual([x['product'] for x in replay['buyer_choices']], [2, 1, 0])
        self.assertEqual(replay['accounting']['revenue'], '7')
        self.assertEqual([x['utility'] for x in replay['buyer_choices']], ['0', '0', '0'])
        replay = replay_menu(payload, [1, 2], {'1': '2', '2': '3'})
        self.assertEqual([x['product'] for x in replay['buyer_choices']], [0, 0, 0])

    def test_zero_license_empty_and_fixed_negative(self):
        payload = instance(alpha=(1, 2), costs=(3, 7), services=(0, 0), buyers=((0, 2, 4),))
        joint = solve_instance(payload)
        self.assertEqual(joint['menu']['products'], [])
        self.assertEqual(value(joint), 0)
        fixed = solve_instance(payload, 'fixed', fixed_products=[1, 2])
        self.assertEqual(fixed['menu']['products'], [1, 2])
        self.assertEqual(value(fixed), -10)
        self.assertEqual(fixed['accounting']['activation_cost'], '10')
        self.assertEqual(fixed['subscription_counts'], {'1': 0, '2': 0})
        self.assertEqual(value(solve_instance(payload, 'fixed', fixed_products=[])), 0)
        for method in ('capacity', 'G0', 'G1', 'G2', 'contiguous'):
            result = solve_instance(payload, method, capacity=0)
            self.assertEqual(value(result), 0)
            self.assertEqual(result['menu']['products'], [])

    def test_fixed_zero_demand_cost_and_completion(self):
        payload = instance(alpha=(1, 2), costs=(3, 7), services=(0, 10), buyers=((2, 2, 4),))
        result = solve_instance(payload, 'fixed', fixed_products=[1, 2])
        self.assertEqual(value(result), -2)
        self.assertEqual(result['subscription_counts'], {'1': 4, '2': 0})
        self.assertEqual(result['accounting']['activation_cost'], '10')
        self.assertGreater(Fraction(result['menu']['prices']['2']), 4)

    def test_subset_enumeration_and_capacity_frontier(self):
        rng = random.Random(20261018)
        tested_catalogs = 0
        for case in range(18):
            q = 2 + case % 3
            alpha = sorted(rng.sample(range(1, 15), q))
            costs = [rng.randint(0, 8) for _ in range(q)]
            services = [str(Fraction(rng.randint(0, 7), 3)) for _ in range(q)]
            pairs = rng.sample([(cap, theta) for cap in range(q + 1) for theta in range(1, 5)], 6)
            buyers = [(cap, theta, rng.randint(1, 5)) for cap, theta in pairs]
            payload = instance(alpha, costs, services, buyers)
            payload['instance_id'] = f'subset-oracle-{case}'
            fixed_values = []
            for size in range(q + 1):
                for catalog in combinations(range(1, q + 1), size):
                    result = solve_instance(payload, 'fixed', fixed_products=list(catalog))
                    fixed_values.append((size, value(result)))
                    tested_catalogs += 1
            exact = solve_instance(payload, 'joint')
            self.assertEqual(value(exact), max(v for _, v in fixed_values))
            capacity = solve_instance(payload, 'capacity', capacity=q)
            self.assertEqual(value(exact), value(capacity))
            for item in capacity['frontier']:
                self.assertEqual(Fraction(item['net_profit']),
                                 max(v for size, v in fixed_values if size <= item['capacity']))
        self.assertEqual(tested_catalogs, 168)

    def test_heuristic_order_and_cache_isolation(self):
        payload = instance()
        for cap in (1, 2, 3):
            results = [solve_instance(payload, method, capacity=cap) for method in ('G0', 'G1', 'G2')]
            optimum = solve_instance(payload, 'capacity', capacity=cap)
            self.assertLessEqual(value(results[0]), value(results[1]))
            self.assertLessEqual(value(results[1]), value(results[2]))
            self.assertLessEqual(value(results[2]), value(optimum))
            for result in results:
                self.assertEqual(result['proof_status'], 'heuristic')
                self.assertTrue(result['replay_verified'])
                self.assertEqual(result['cache_stats']['requests'],
                                 result['cache_stats']['solves'] + result['cache_stats']['hits'])
            repeated = solve_instance(payload, 'G2', capacity=cap)
            self.assertEqual(repeated, results[2])
            restricted = solve_instance(payload, 'contiguous', capacity=cap)
            self.assertLessEqual(value(restricted), value(optimum))

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / 'input.json'
            target = path / 'result.json'
            source.write_text(json.dumps(instance()), encoding='utf-8')
            self.assertEqual(load_instance(source)['instance_id'], 'interface-check')
            command = [sys.executable, str(ROOT / 'src' / 'pht_experiment.py'),
                       '--input', str(source), '--method', 'G2', '--capacity', '2',
                       '--output', str(target)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = target.read_bytes()
            self.assertEqual(json.loads(first)['proof_status'], 'heuristic')
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(target.read_bytes(), first)


if __name__ == '__main__':
    unittest.main(verbosity=2)
