"""Deterministic runner checks using a tiny stand-in API, never a benchmark."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_formal import InputStore, canonical_digest, cpu_for_instance, existing_records, completed_resource_status

STUB = '''import time, json, hashlib

def benchmark(payload, method='joint', budget_seconds=5, **options):
    if method == 'G0':
        raise RuntimeError('intentional test error')
    if method == 'G1':
        time.sleep(.3)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return {'status':'exact','input_sha256':digest,'net_profit':'0',
            'parameters':{'Q':1,'K':1,'B':1,'T':2},
            'phase_seconds':{'parse_validation':0,'preprocess_optimization':0,
                             'recovery':0,'replay':0,'total':0}}
'''


class RunnerTests(unittest.TestCase):
    def test_returned_result_resource_boundaries(self):
        job={'wall_budget_seconds':10.,'memory_limit_mib':2048}
        self.assertEqual(completed_resource_status(10,2048,job),'completed')
        self.assertEqual(completed_resource_status(10.0001,2048,job),'over_budget_completed')
        self.assertEqual(completed_resource_status(1,2048.0001,job),'memory_limit_completed')
        self.assertEqual(completed_resource_status(12,2100,job),'memory_limit_completed')

    def test_indexed_inputs_and_cpu_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'instance_id': f'x{i}', 'value': i} for i in range(4)]
            (root / 'inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            store = InputStore(root, cache_size=2)
            self.assertEqual(store.load('inputs.jsonl', 3), rows[3])
            self.assertEqual(store.load('inputs.jsonl', 0), rows[0])
            self.assertEqual(store.load('inputs.jsonl', 3), rows[3])
            with self.assertRaises(ValueError):
                store.load('inputs.jsonl', 4)
            self.assertEqual(cpu_for_instance('shared-case', [2, 5]), cpu_for_instance('shared-case', [2, 5]))
            self.assertIn(cpu_for_instance('shared-case', [2, 5]), [2, 5])

    def test_partial_tail_and_complete_final_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.jsonl'
            item = {'job_id': 'ok', 'runner_status': 'completed'}
            prefix = json.dumps(item).encode() + b'\n'
            path.write_bytes(prefix + b'{"job_id":"incom')
            latest, count = existing_records(path)
            self.assertEqual(latest['ok'], item)
            self.assertEqual(count, Counter({'ok': 1}))
            self.assertEqual(path.read_bytes(), prefix)
            self.assertEqual(len(list(path.parent.glob('*.partial_tail.*'))), 1)
            path.write_text(json.dumps(item))
            existing_records(path)
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            path.write_bytes(b'bad\n'+prefix)
            with self.assertRaises(ValueError):
                existing_records(path)
            self.assertEqual(path.read_bytes(), b'bad\n'+prefix)

    def test_isolation_statuses_bulk_reuse_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'src').mkdir(); (root/'protocol').mkdir(); (root/'data/formal').mkdir(parents=True)
            (root/'src/benchmark_api.py').write_text(STUB)
            protocol = root/'protocol/formal_protocol.json'
            protocol.write_text('{"source_snapshots":[]}\n')
            protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
            payload={'instance_id':'same-instance','metadata':{'source_sha256':'source-test'}}
            (root/'data/inputs.jsonl').write_text(json.dumps(payload)+'\n')
            jobs=[]
            for mode, methods in [('bulk_validation',['joint','joint']),
                                  ('isolated_benchmark',['joint','G0','G1'])]:
                for i,method in enumerate(methods):
                    jobs.append({'job_id':f'{mode}-{i}','suite':'test','instance_id':'same-instance',
                        'input_path':'data/inputs.jsonl','input_line_index':0,'method':method,'repeat':i+1,
                        'timing_mode':mode,'optimization_budget_seconds':.02,'wall_budget_seconds':.06,
                        'memory_limit_mib':2048,'options':{},'formal_protocol_sha256':protocol_hash})
            (root/'data/formal/jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
            command=[sys.executable,str(ROOT/'src/run_formal.py'),'--package-root',str(root),'--workers','1']
            first=subprocess.run(command+['--mode','isolated','--limit','1'],capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            second=subprocess.run(command+['--mode','isolated'],capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            records=[json.loads(x) for x in (root/'results/formal_isolated/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),3)
            self.assertEqual({x['runner_status'] for x in records},{'completed','error','hard_timeout'})
            self.assertEqual(len({x['cpu_id'] for x in records}),1)
            self.assertEqual(len({x['worker_pid'] for x in records}),3)
            self.assertTrue(all(x['rss_scope']=='fresh_process' for x in records))
            hard=next(x for x in records if x['runner_status']=='hard_timeout')
            self.assertIsNone(hard['result'])
            third=subprocess.run(command+['--mode','bulk'],capture_output=True,text=True)
            self.assertEqual(third.returncode,0,third.stderr)
            bulk=[json.loads(x) for x in (root/'results/formal_bulk/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(bulk),2)
            self.assertEqual(len({x['worker_pid'] for x in bulk}),1)
            self.assertTrue(all(x['rss_scope']=='persistent_worker_cumulative' for x in bulk))
            self.assertTrue(all(x['input_sha256']==canonical_digest(payload) for x in bulk))
            (root/'src/benchmark_api.py').write_text(STUB+'\n# changed implementation\n')
            changed=subprocess.run(command+['--mode','bulk'],capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('Resume identity differs',changed.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
