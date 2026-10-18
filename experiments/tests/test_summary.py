import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[1]/'src'/'summarize_formal.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def record(iid, method, profit='10', suite='comparison', repeat=1, runner='completed', worker='ok', seconds=1, attempt=1):
    api=None if profit is None else {'proof_status':'exact' if method in ('joint','capacity') else 'heuristic',
          'accounting':{'net_profit':profit},'menu':{'products':[1], 'prices':{'1':'1'}},
          'parameters':{'Q':1,'B':1,'K':1,'T':2},'phase_seconds':{'preprocess_optimization':seconds,'total':seconds+1},'status':worker}
    return {'job_id':f'{suite}:{iid}:{method}:rep{repeat}','suite':suite,'instance_id':iid,'method':method,'repeat':repeat,
            'attempt':attempt,'timing_mode':'isolated_benchmark','cpu_id':2,'runner_status':runner,'worker_status':worker,
            'formal_protocol_sha256':'protocol','input_sha256':iid,'result':api}


class SummaryTests(unittest.TestCase):
    def test_cluster_means_windows_before_meters(self):
        rows=[{'source_client':'a','evaluation_month':'1','v':0},
              {'source_client':'a','evaluation_month':'1','v':0},
              {'source_client':'a','evaluation_month':'2','v':10},
              {'source_client':'b','evaluation_month':'1','v':20}]
        out=s.clustered_summary(rows,'v')
        self.assertEqual(out['meter_equal']['mean'],12.5)
        self.assertEqual(out['instance_weighted']['mean'],7.5)

    def test_exact_and_zero_reference(self):
        r=[record('zero','joint','0'),record('zero','G0','0'),record('small','joint','1/10'),record('small','G0','1/20')]
        reps,_,_=s.representatives(r);rows,summary,_=s.quality_tables(reps,{},1e-7,1e-8)
        zero=next(x for x in rows if x['instance_id']=='zero');small=next(x for x in rows if x['instance_id']=='small')
        self.assertIsNone(zero['relative_gap']);self.assertTrue(zero['exact_profit_match'])
        self.assertEqual(small['relative_gap'],.5)

    def test_timeout_kept_and_excluded_from_paired_time(self):
        r=[record('a','joint'),record('a','G2'),record('b','joint'),record('b','G2',None,runner='hard_timeout',worker=None)]
        _,groups,_=s.representatives(r)
        cases,summary,pairs,ps,_=s.timing_tables(r,groups,{})
        self.assertEqual(ps[0]['instances_with_both_methods_attempted'],2)
        self.assertEqual(ps[0]['matched_completed_instances'],1)
        status=s.status_tables(r,r)
        self.assertTrue(any(x['method']=='G2' and x['timeout_jobs']==1 for x in status))

    def test_partial_repeat_not_used_for_speed_ratio(self):
        r=[record('a','joint',repeat=1),record('a','joint',repeat=2),record('a','G2',repeat=1),record('a','G2',None,repeat=2,runner='hard_timeout')]
        _,groups,_=s.representatives(r);_,_,_,pairs,_=s.timing_tables(r,groups,{})
        self.assertEqual(pairs[0]['matched_completed_instances'],0)

    def test_first_replay_is_not_best_of_repeat(self):
        r=[record('a','MILP','9',repeat=1),record('a','MILP','10',repeat=2)]
        reps,_,_=s.representatives(r)
        self.assertEqual(s.profit(reps['comparison','a','MILP']),9)

    def test_cross_suite_capacity_reference(self):
        j=record('a','joint',suite='real_full');c=record('a','capacity',suite='real_capacity')
        c['result']['frontier']=[{'capacity':0,'net_profit':'0'},{'capacity':1,'net_profit':'10'}]
        reps,_,_=s.representatives([j,c]);rows,_=s.capacity_tables(reps,{})
        self.assertEqual(rows[0]['minimal_capacity_attaining_joint_optimum'],1)

    def test_gamma_pairs_and_duplicate_base(self):
        r=[record('base','joint',suite='real_full'),record('base','joint'),record('sqrt','joint','11',suite='mapping_sensitivity')]
        ins={'base':{'metadata':{'source_client':'m','evaluation_month':'2020-01'}},
             'sqrt':{'metadata':{'source_client':'m','evaluation_month':'2020-01','base_instance_id':'base','quality_mapping_gamma':'0.5'}}}
        reps,_,_=s.representatives(r);joints,issues=s.global_joint(reps,ins);rows,_=s.gamma_tables(joints)
        self.assertEqual(len(joints),2);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['profit_difference'],1)

    def test_incomplete_joint_not_used_as_exact_reference(self):
        incomplete=record('a','joint','0',repeat=1)
        incomplete['result']['proof_status']='incomplete'
        complete=record('a','joint','10',repeat=2)
        r=[incomplete,complete,record('a','G0','9')]
        reps,_,_=s.representatives(r)
        rows,_,_=s.quality_tables(reps,{},1e-7,1e-8)
        self.assertEqual(rows[0]['reference_profit_exact'],'10')
        self.assertEqual(rows[0]['relative_gap'],.1)

    def test_milp_numeric_issue_is_reported_separately(self):
        milp=record('a','MILP','5')
        milp['result']['quality_comparison_eligible']=False
        milp['result']['numeric_recovery_issue']=True
        reps,_,_=s.representatives([record('a','joint'),milp])
        rows,summary,_=s.quality_tables(reps,{},1e-7,1e-8)
        group=next(x for x in summary if x['grouping']=='all')
        self.assertEqual(group['numeric_or_assignment_exclusions'],1)
        self.assertEqual(group['relative_gap']['instance_weighted']['n'],0)
        self.assertEqual(group['all_returned_menu_relative_gap']['instance_weighted']['mean'],.5)
        self.assertFalse(s.is_successful_timing(milp))

    def test_milp_limit_incumbent_is_not_completed_timing(self):
        limited=record('a','MILP','9',worker='limit')
        limited['result'].update(raw_status=1,quality_comparison_eligible=True)
        self.assertTrue(s.timeout(limited))
        self.assertFalse(s.is_successful_timing(limited))
        limited['worker_status']='limit_replay_mismatch'
        self.assertTrue(s.timeout(limited))
        optimal=record('a','MILP','10',worker='optimal')
        optimal['result'].update(raw_status=0,optimization_budget_overrun_seconds=.0001,quality_comparison_eligible=True)
        self.assertFalse(s.timeout(optimal))
        self.assertFalse(s.is_successful_timing(optimal))

    def test_optimal_overrun_keeps_profit_but_excludes_budgeted_timing(self):
        optimal=record('a','MILP','10',worker='optimal',seconds=5.0001)
        optimal['result'].update(raw_status=0,optimization_budget_overrun_seconds=.0001,quality_comparison_eligible=True)
        rows=[record('a','joint'),optimal]
        self.assertEqual(s.profit(optimal),10)
        self.assertTrue(s.is_completed(optimal))
        self.assertFalse(s.timeout(optimal))
        self.assertTrue(s.optimization_overrun(optimal))
        status=s.status_tables(rows,rows)
        milp=next(x for x in status if x['population']=='final_jobs' and x['method']=='MILP')
        self.assertEqual(milp['optimization_overrun_jobs'],1)
        self.assertEqual(milp['timeout_jobs'],0)
        _,groups,_=s.representatives(rows)
        _,_,_,pairs,_=s.timing_tables(rows,groups,{})
        self.assertEqual(pairs[0]['matched_completed_instances'],0)
        optimal['result'].pop('optimization_budget_overrun_seconds')
        optimal['result']['budget_seconds']=5
        self.assertTrue(s.optimization_overrun(optimal))

    def test_missing_planned_and_attempt_retention(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);rows=[record('a','joint',None,runner='interrupted',attempt=1),record('a','joint',attempt=2)]
            (td/'r.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
            inputs=[{'instance_id':'a','products':[{}],'metadata':{'source_client':'m','evaluation_month':'2020-01'}}]
            (td/'i.jsonl').write_text('\n'.join(json.dumps(x) for x in inputs))
            jobs=[record('a','joint'),record('b','joint')]
            (td/'j.jsonl').write_text('\n'.join(json.dumps(x) for x in jobs))
            out=s.summarize([td/'r.jsonl'],[td/'i.jsonl'],td/'out',jobs_path=td/'j.jsonl')
            self.assertEqual(out['raw_attempts'],2);self.assertEqual(out['final_jobs'],1)
            self.assertEqual(out['audit']['missing_planned_jobs'],1)

if __name__=='__main__':unittest.main()
