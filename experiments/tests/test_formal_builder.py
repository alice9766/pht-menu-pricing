"""Validate the predeclared formal grid without invoking any solver."""
import copy
import csv
import importlib.util
import json
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_formal', ROOT / 'src/build_formal.py')
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


class FormalBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.protocol_hash = formal.load_protocol()
        cls.all_real = formal.read_jsonl(ROOT / 'data/all/instances.jsonl')
        cls.pilot = formal.read_jsonl(ROOT / 'data/pilot/instances.jsonl')
        cls.mapping = formal.read_jsonl(ROOT / 'data/formal/mapping_instances.jsonl')
        cls.scaling = formal.read_jsonl(ROOT / 'data/formal/scaling_instances.jsonl')
        cls.jobs = formal.read_jsonl(ROOT / 'data/formal/jobs.jsonl')
        with (ROOT / 'data/source/window_quality_audit.csv').open() as f:
            cls.audit = list(csv.DictReader(f))

    def test_source_hashes_and_data_protocol_unchanged(self):
        formal.verify_sources(self.protocol)
        snapshot = next(x for x in self.protocol['source_snapshots']
                        if x['path'] == 'protocol/data_protocol.json')
        self.assertEqual(snapshot['sha256'], '2fae45103ebcb1a7912257139c67b6283792e2b7777f18fdfa8d96bfb7d1bbb9')

    def test_job_grid_and_unique_ids(self):
        self.assertEqual(len(self.jobs), 23499)
        self.assertEqual(len({j['job_id'] for j in self.jobs}), 23499)
        self.assertEqual(Counter(j['suite'] for j in self.jobs),
                         {'real_full': 20232, 'real_comparison': 1080, 'real_capacity': 216,
                          'mapping_sensitivity': 432, 'synthetic_scaling': 1539})
        self.assertEqual(Counter(j['timing_mode'] for j in self.jobs),
                         {'bulk_validation': 20664, 'isolated_benchmark': 2835})

    def test_job_input_binding_and_budgets(self):
        inputs = {'data/all/instances.jsonl': self.all_real, 'data/pilot/instances.jsonl': self.pilot,
                  'data/formal/mapping_instances.jsonl': self.mapping,
                  'data/formal/scaling_instances.jsonl': self.scaling}
        for job in self.jobs:
            self.assertEqual(inputs[job['input_path']][job['input_line_index']]['instance_id'], job['instance_id'])
            self.assertEqual(job['formal_protocol_sha256'], self.protocol_hash)
            self.assertEqual(job['optimization_budget_seconds'], 5)
            self.assertEqual(job['wall_budget_seconds'], 10)
            self.assertEqual(job['memory_limit_mib'], 2048)
            self.assertEqual(job['options'], {'capacity': 6} if job['method'] == 'capacity' else {})

    def test_input_registry_hashes(self):
        registry = json.loads((ROOT / 'data/formal/input_registry.json').read_text())
        for item in registry:
            self.assertEqual(formal.digest_file(ROOT / item['input_path']), item['sha256'])

    def test_pilot_is_original_fixed_subset(self):
        all_index = {i['instance_id']: i for i in self.all_real}
        self.assertEqual(len(self.pilot), 216)
        self.assertEqual(len({i['metadata']['source_client'] for i in self.pilot}), 24)
        for ins in self.pilot:
            self.assertEqual(ins, all_index[ins['instance_id']])

    def test_gamma_pairs_change_only_quality_and_provenance(self):
        by_base = {i['instance_id']: i for i in self.pilot}
        self.assertEqual(len(self.mapping), 432)
        seen = Counter()
        for ins in self.mapping:
            base = by_base[ins['metadata']['base_instance_id']]
            gamma = ins['metadata']['quality_mapping_gamma']
            seen[(base['instance_id'], gamma)] += 1
            self.assertEqual(ins['buyers'], base['buyers'])
            for p, original in zip(ins['products'], base['products']):
                self.assertEqual(p['q'], original['q'])
                self.assertEqual(p['activation_cost'], original['activation_cost'])
                self.assertEqual(p['unit_service_cost'], original['unit_service_cost'])
            alpha = list(map(lambda p: Fraction(p['alpha']), ins['products']))
            self.assertTrue(all(x < y for x, y in zip(alpha, alpha[1:])))
        self.assertEqual(set(seen.values()), {1})
        self.assertEqual(set(gamma for _, gamma in seen), {'0.5', '2'})
        regenerated, audits = formal.build_mapping(self.pilot, self.audit, self.protocol, self.protocol_hash)
        self.assertEqual(regenerated, self.mapping)
        self.assertEqual(len(audits), 72)
        self.assertTrue(all(r['strictly_increasing'] for r in audits))

    def test_synthetic_grid_has_exact_dimensions_and_margins(self):
        self.assertEqual(len(self.scaling), 117)
        coords = Counter()
        for ins in self.scaling:
            meta = ins['metadata']; q, b, t = [meta['dimensions'][x] for x in ['Q','B','T']]
            coords[(q,b,t)] += 1
            self.assertEqual(len(ins['products']), q)
            self.assertEqual(len(ins['buyers']), b*(t-1))
            self.assertEqual(len({x['access_cap'] for x in ins['buyers']}), b)
            self.assertEqual(len({x['theta'] for x in ins['buyers']}), t-1)
            self.assertEqual(sum(x['count'] for x in ins['buyers']), 512)
            self.assertTrue(all(type(x['count']) is int and x['count'] > 0 for x in ins['buyers']))
            caps, theta = Counter(), Counter()
            for buyer in ins['buyers']:
                caps[buyer['access_cap']] += buyer['count']; theta[buyer['theta']] += buyer['count']
            self.assertEqual(set(caps.values()), {512//b})
            self.assertEqual(set(theta.values()), {512//(t-1)})
            alpha = [Fraction(p['alpha']) for p in ins['products']]
            self.assertEqual(alpha[-1], 1)
            self.assertTrue(alpha[0] > 0 and all(x < y for x,y in zip(alpha,alpha[1:])))
        self.assertEqual(len(coords), 13)
        self.assertEqual(set(coords.values()), {9})

    def test_synthetic_exact_reproduction_and_paired_quality(self):
        quality = {}
        for ins in self.scaling:
            meta = ins['metadata']; dim = meta['dimensions']
            regenerated = formal.make_scaling_instance(dim['Q'],dim['B'],dim['T'],meta['seed'],
                    meta['cost_profile'],meta['scaling_axes'],self.protocol,self.protocol_hash)
            self.assertEqual(ins, regenerated)
            key = (dim['Q'],meta['seed'])
            alphas = [p['alpha'] for p in ins['products']]
            if key in quality:
                self.assertEqual(alphas,quality[key])
            quality[key] = alphas

    def test_method_and_repeat_assignment_prespecified(self):
        scaling_index = {i['instance_id']:i for i in self.scaling}
        assignment = defaultdict(list)
        for job in self.jobs:
            if job['suite'] == 'synthetic_scaling':
                assignment[job['instance_id']].append((job['method'],job['repeat']))
            else:
                self.assertEqual(job['repeat'],1)
        for instance_id, assigned in assignment.items():
            q = scaling_index[instance_id]['metadata']['dimensions']['Q']
            methods = self.protocol['comparison']['methods'] if q <= 32 else ['joint']
            self.assertEqual(set(assigned), {(method,rep) for method in methods for rep in [1,2,3]})
        regenerated = formal.jobs_for(self.protocol,self.protocol_hash,self.all_real,self.pilot,self.mapping,self.scaling)
        self.assertEqual(regenerated,self.jobs)


if __name__ == '__main__':
    unittest.main()
