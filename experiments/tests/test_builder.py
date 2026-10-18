"""Meaningful invariants for data construction and pre-solver selection."""
import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_instances', ROOT / 'src/build_instances.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_hash = builder.load_protocol()
        cls.ledger = builder.audit_windows(cls.protocol,
            ROOT / cls.protocol['source']['audit_csv_path'],
            ROOT / cls.protocol['source']['finiteness_audit_path'])
        cls.selected, cls.monthly = builder.select_windows(cls.ledger, cls.protocol)
        cls.instances = [json.loads(line) for line in (ROOT / 'data/pilot/instances.jsonl').read_text().splitlines()]

    def test_full_audit_coverage(self):
        self.assertEqual(len(self.ledger), 2960)
        self.assertEqual(len({(r['client'], r['evaluation_month']) for r in self.ledger}), 2960)
        self.assertEqual(sum(r['model_eligible'] for r in self.ledger), 2248)
        self.assertEqual(sum(not r['model_eligible'] for r in self.ledger), 712)
        self.assertTrue(all(bool(r['exclusion_reasons']) == (not r['model_eligible']) for r in self.ledger))
        self.assertEqual([m['model_eligible_windows'] for m in self.monthly],
                         [157, 153, 305, 311, 324, 322, 329, 347])

    def test_late_activation_and_quality_failures_retained(self):
        strict = [r for r in self.ledger if r['computable_before_quality_order'] and
                  'quantized_quality_not_strictly_increasing' not in r['exclusion_reasons']]
        self.assertEqual(len(strict), 2266)
        self.assertEqual(sum('late_activation_in_reference' in r['exclusion_reasons'] for r in strict), 18)
        self.assertEqual(sum('quantized_quality_not_strictly_increasing' in r['exclusion_reasons']
                             for r in self.ledger), 105)

    def test_pilot_selection_and_all_mode(self):
        self.assertEqual(len(self.selected), 24)
        self.assertEqual(len({r['client'] for r in self.selected}), 24)
        self.assertTrue(all(v == 3 for v in Counter(r['evaluation_month'] for r in self.selected).values()))
        all_rows, _ = builder.select_windows(self.ledger, self.protocol, use_all=True)
        self.assertEqual(len(all_rows), 2248)
        self.assertEqual(len(self.instances), 216)
        self.assertEqual(len({i['instance_id'] for i in self.instances}), 216)

    def test_selection_independent_of_solver_results(self):
        modified = copy.deepcopy(self.ledger)
        for i, row in enumerate(modified):
            row['pretend_solver_profit'] = 10**9 - i
            row['pretend_baseline_gap'] = i / 10
        selected, _ = builder.select_windows(list(reversed(modified)), self.protocol)
        self.assertEqual([(r['client'], r['evaluation_month']) for r in selected],
                         [(r['client'], r['evaluation_month']) for r in self.selected])

    def test_exact_schema_and_dimension(self):
        for ins in self.instances:
            self.assertEqual(ins['schema_version'], 'pht-experiment-v1')
            self.assertEqual([p['q'] for p in ins['products']], list(range(1, 7)))
            alphas = [Fraction(p['alpha']) for p in ins['products']]
            self.assertTrue(all(a > 0 for a in alphas))
            self.assertTrue(all(a < b for a, b in zip(alphas, alphas[1:])))
            self.assertEqual(len(ins['buyers']), 12)
            self.assertEqual({b['access_cap'] for b in ins['buyers']}, {2, 4, 6})
            self.assertEqual({Fraction(b['theta']) for b in ins['buyers']}, {1, 2, 3, 4})
            self.assertTrue(all(type(b['count']) is int and b['count'] > 0 for b in ins['buyers']))
            self.assertEqual(sum(b['count'] for b in ins['buyers']), 120)
            self.assertEqual(ins['metadata']['dimensions'], {'Q': 6, 'B': 3, 'K': 12, 'T': 5})
            for p in ins['products']:
                self.assertGreaterEqual(Fraction(p['activation_cost']), 0)
                self.assertGreaterEqual(Fraction(p['unit_service_cost']), 0)

    def test_association_changes_joint_counts_only(self):
        groups = defaultdict(list)
        for ins in self.instances:
            meta = ins['metadata']
            groups[(meta['source_client'], meta['evaluation_month'], meta['cost_profile'])].append(ins)
        for triplet in groups.values():
            self.assertEqual(len(triplet), 3)
            products = triplet[0]['products']
            for ins in triplet:
                self.assertEqual(ins['products'], products)
                cap_marginal, theta_marginal = Counter(), Counter()
                for b in ins['buyers']:
                    cap_marginal[b['access_cap']] += b['count']
                    theta_marginal[Fraction(b['theta'])] += b['count']
                self.assertEqual(cap_marginal, {2: 40, 4: 40, 6: 40})
                self.assertEqual(theta_marginal, {1: 30, 2: 30, 3: 30, 4: 30})

    def test_same_source_products_and_parameter_provenance(self):
        row_index = {(r['client'], r['evaluation_month']): r for r in self.selected}
        for ins in self.instances:
            meta = ins['metadata']
            row = row_index[(meta['source_client'], meta['evaluation_month'])]
            self.assertEqual(meta['quality_origin'], 'same_meter_same_month_delayed_versions')
            self.assertEqual(meta['protocol_sha256'], self.protocol_hash)
            self.assertEqual(meta['source_sha256'], self.protocol['source']['archive_sha256'])
            self.assertEqual(set(meta['parameter_sources']), {'alpha_q', 'c_q', 's_q', 'bar_q_k', 'theta_k', 'n_k'})
            self.assertEqual([p['alpha'] for p in ins['products']],
                             [row[f'alpha_quantized_delay_{d}'] for d in [240, 120, 60, 30, 15, 0]])

    def test_cost_vectors_match_original_and_repeat(self):
        source_reference = json.loads((ROOT / 'data/source/legacy_cost_reference.json').read_text())
        for profile, expected in self.protocol['cost_scenarios'].items():
            actual = builder.legacy_costs(profile)
            self.assertEqual(actual, builder.legacy_costs(profile))
            self.assertEqual(actual[0], expected['activation_cost_exact'])
            self.assertEqual(actual[1], expected['unit_service_cost_exact'])
            self.assertEqual(actual[0], source_reference['profiles'][profile]['activation_cost_exact'])
            self.assertEqual(actual[1], source_reference['profiles'][profile]['unit_service_cost_exact'])

    def test_rebuild_identical_inputs_and_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = builder.build_trace(Path(temporary))
            recorded = json.loads((ROOT / 'data/pilot/build_summary.json').read_text())
            self.assertEqual(summary, recorded)
            for filename in ['instances.jsonl', 'window_eligibility.csv', 'selection_manifest.json']:
                self.assertEqual((Path(temporary) / filename).read_bytes(),
                                 (ROOT / 'data/pilot' / filename).read_bytes())

    def test_synthetic_dimensions_counts_and_determinism(self):
        for q, b, levels in [(1, 1, 1), (12, 3, 4), (20, 20, 8)]:
            ins = builder.make_synthetic_instance(q, b, levels, 42)
            self.assertEqual(ins, builder.make_synthetic_instance(q, b, levels, 42))
            self.assertEqual(len(ins['products']), q)
            self.assertEqual(len({r['access_cap'] for r in ins['buyers']}), b)
            self.assertEqual(len(ins['buyers']), b * levels)
            self.assertTrue(all(type(r['count']) is int and r['count'] > 0 for r in ins['buyers']))
            self.assertEqual(ins['metadata']['dimensions']['T'], levels + 1)
        with self.assertRaises(ValueError):
            builder.make_synthetic_instance(3, 4, 2)


if __name__ == '__main__':
    unittest.main()
