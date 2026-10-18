import copy
import json
import sys
from fractions import Fraction
from pathlib import Path

import unittest
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import summarize_final as s


def market(alpha='1'):
    return {'schema_version':'pht-experiment-v1','instance_id':'i','products':[
        {'q':1,'alpha':alpha,'activation_cost':'2','unit_service_cost':'0'},
        {'q':2,'alpha':'3','activation_cost':'1','unit_service_cost':'0'}],
        'buyers':[{'type_id':'k','access_cap':2,'theta':'1','count':1}],
        'metadata':{'source_client':'meter','evaluation_month':'2014-02','cost_profile':'balanced','association':'positive'}}


def record(method='joint',repeat=1,price='2',suite='real_comparison'):
    payload=market();menu={'products':[1],'prices':{'1':price}}
    r={'instance_id':'i','method':method,'repeat':repeat,'suite':suite,'job_id':f'{suite}:{method}:{repeat}',
       'runner_status':'completed','worker_status':'exact' if method=='joint' else 'optimal_replay_mismatch',
       'timing_mode':'isolated','cpu_id':0,'result':{'menu':menu,'accounting':s.independent_replay(payload,menu)['accounting'],
       'proof_status':'exact' if method=='joint' else 'numeric_weak_ic','completed_optimization':True,'raw_status':0,'choices_match':False,
       'quality_comparison_eligible':False,'numeric_recovery_issue':True,'has_valid_menu':True,
       'phase_seconds':{'preprocess_optimization':1.,'recovery':.1,'replay':.1,'total':1.2},'budget_seconds':5.}}
    return r


def test_independent_tie_zero_utility_chooses_highest_quality():
    p=market();m={'products':[1,2],'prices':{'1':'1','2':'3'}}
    out=s.independent_replay(p,m)
    assert out['buyer_choices'][0]['product']==2
    assert Fraction(out['net_profit'])==0


def test_independent_zero_demand_products_still_pay_fixed_cost():
    out=s.independent_replay(market(),{'products':[1,2],'prices':{'1':'10','2':'10'}})
    assert out['net_profit']=='-3'
    assert out['buyer_choices'][0]['product']==0


def test_empty_menu_valid_and_repeated_prices_invalid():
    assert s.independent_replay(market(),{'products':[],'prices':{}})['net_profit']=='0'
    with unittest.TestCase().assertRaises(ValueError):s.independent_replay(market(),{'products':[1],'prices':{'1':'1','01':'1'}})


def test_milp_mismatch_and_negative_profit_included():
    r=record('MILP');inputs={'i':{'payload':market(),'metadata':{}}}
    audit=s.independent_audit([r],inputs)
    assert not audit['issues']
    assert s.profit(r)==-2
    assert s.is_successful_timing(r)
    reps={('real_comparison','i','MILP'):r,('real_comparison','i','joint'):record(price='1')}
    rows,summary,unavailable=s.quality_tables(reps,inputs,1e-7,1e-8)
    assert len(rows)==1 and rows[0]['eligible_for_algorithm_quality']
    assert summary[0]['algorithm_quality_eligible_instances']==1
    assert summary[0]['numeric_or_assignment_exclusions']==0


def test_representative_is_first_valid_never_best():
    a,b=record('G0',1,price='0'),record('G0',2,price='1')
    reps,_,_=s.representatives([b,a])
    assert reps[('real_comparison','i','G0')]['repeat']==1


def test_timeout_incumbent_counts_for_profit_not_completed_time():
    r=record('MILP');r['result']['raw_status']=1;r['worker_status']='limit'
    assert s.profit(r) is not None
    assert s.timeout(r) and not s.is_successful_timing(r)
    assert s.quality_coverage([r],[r])[0]['valid_timeout_menu_runs']==1


def test_incomplete_repeats_not_in_primary_timing():
    rows=[record('G0',i,price='1') for i in (1,2,3)]
    rows[-1]['result']['budget_exhausted']=True
    _,groups,_=s.representatives(rows)
    cases,summaries,_,_,_=s.timing_tables(rows,groups,{'i':{'metadata':{}}},rows)
    assert cases[0]['completed_repeats']==2
    assert summaries[0]['optimization_seconds']['n']==0


def test_independent_audit_catches_stated_profit_error():
    r=record();r['result']['accounting']['net_profit']='1000'
    audit=s.independent_audit([r],{'i':{'payload':market()}})
    assert audit['counts']['reported_profit_mismatch']==1
    assert s.profit(r)==-2


def test_cross_mapping_fixed_prices_all_nine_pairs_and_regret_above_one():
    joints=[];inputs={}
    for gamma,alpha,price in [('0.5','1/2','1/2'),('1','1','1'),('2','2','2')]:
        iid='base'+gamma;p=market(alpha);p['products'][0]['activation_cost']='0';p['products']=p['products'][:1]
        profit=Fraction(price)
        # gamma .5 target optimum .5, source2 asks2 so demand becomeszero.
        inputs[iid]={'payload':p}
        joints.append({'instance_id':iid,'base_instance_id':'base','source_client':'meter','evaluation_month':'2014-02',
                       'cost_profile':'balanced','association':'positive','gamma':gamma,'products':[1],'prices':{'1':price},'profit_exact':str(profit)})
    rows,summary,issues=s.cross_mapping(joints,inputs)
    assert len(rows)==9 and len(summary)==9 and not issues
    assert all(r['absolute_regret']==0 for r in rows if r['source_gamma']==r['target_gamma'])
    r=next(r for r in rows if r['source_gamma']=='2' and r['target_gamma']=='0.5')
    assert r['replay_profit']==0 and r['relative_regret']==1
    # Cross-replay may legitimately be negative and regret > 100% with fixedcosts.
    p=market();assert Fraction(s.independent_replay(p,{'products':[1],'prices':{'1':'10'}})['net_profit'])<0


def test_meter_equal_clustering_not_scenario_equal():
    rows=[{'source_client':'a','evaluation_month':'m','value':0}]*9+[{'source_client':'b','evaluation_month':'m','value':10}]
    out=s.clustered_summary(rows,'value')
    assert out['instance_weighted']['mean']==1
    assert out['meter_equal']['mean']==5
    assert out['meter_equal_mean_ci95']['clusters']==2


def test_cli_summary_writes_compact_tables_and_coverage(tmp_path):
    p=market();r=record('MILP',price='1')
    inputpath=tmp_path/'inputs.jsonl';resultpath=tmp_path/'results.jsonl';jobpath=tmp_path/'jobs.jsonl'
    inputpath.write_text(json.dumps(p)+'\n');resultpath.write_text(json.dumps(r)+'\n');jobpath.write_text(json.dumps({k:v for k,v in r.items() if k!='result'})+'\n')
    summary=s.summarize([resultpath],[inputpath],tmp_path/'output',jobs_path=jobpath)
    assert summary['execution_complete']
    assert summary['quality_coverage'][0]['valid_menu_runs']==1
    assert (tmp_path/'output'/'tables'/'all_jobs.csv').exists()
    assert summary['table_files']['timing_by_instance']['rows']==1


def capacity_fixture():
    p=market();p['products']=p['products'][:1];p['buyers'][0]['access_cap']=1
    empty={'products':[],'prices':{}};check=s.independent_replay(p,empty)
    ref=record();ref['result'].update(menu=empty,accounting=check['accounting'],buyer_choices=check['buyer_choices'])
    cap=copy.deepcopy(ref);cap.update(method='capacity',suite='real_capacity',job_id='capacity')
    cap['result'].update(capacity_limit=1,frontier=[{'capacity':h,'menu':empty,'net_profit':'0'} for h in (0,1)])
    inputs={'i':{'payload':p,'metadata':{}}}
    s.independent_audit([ref,cap],inputs)
    reps={('real_comparison','i','joint'):ref,('real_capacity','i','capacity'):cap}
    return cap,reps,inputs


def test_capacity_exact_zero_optimum_is_not_divided_by_zero():
    cap,reps,inputs=capacity_fixture()
    rows,summary=s.capacity_tables(reps,inputs)
    assert rows[0]['certified_optimal_frontier']
    assert rows[0]['reference_zero']
    assert rows[0]['minimal_capacity_attaining_joint_optimum']==0
    assert all(p['relative_to_joint_optimum'] is None for p in rows[0]['frontier'])
    assert summary[0]['zero_joint_optimum_instances']==1
    audit=s.capacity_certification_audit(rows,[cap])
    assert audit['all_planned_frontiers_certified']


def test_capacity_incomplete_exact_label_is_not_certification():
    cap,reps,inputs=capacity_fixture();cap['result']['completed_optimization']=False
    rows,summary=s.capacity_tables(reps,inputs)
    assert not rows[0]['certified_optimal_frontier']
    assert rows[0]['frontier_kind']=='feasible_lower_bounds'
    assert rows[0]['minimal_capacity_attaining_joint_optimum'] is None
    assert not summary[0]['frontier']
    assert not s.capacity_certification_audit(rows,[cap])['all_planned_frontiers_certified']


def test_capacity_unproven_values_and_missing_jobs_retained():
    cap,reps,inputs=capacity_fixture();cap['result']['proof_status']='feasible_incumbent'
    rows,summary=s.capacity_tables(reps,inputs)
    assert len(rows[0]['frontier'])==2
    assert not rows[0]['certified_optimal_frontier']
    planned=[cap,{**cap,'instance_id':'missing'}]
    audit=s.capacity_certification_audit(rows,planned)
    assert len(audit['unproven_frontiers'])==1
    assert len(audit['missing_frontiers'])==1


class FinalSummaryTests(unittest.TestCase):
    pass


def _wrap(function):
    def method(self):
        if function.__name__ == 'test_cli_summary_writes_compact_tables_and_coverage':
            with tempfile.TemporaryDirectory() as folder:function(Path(folder))
        else:function()
    return method


for _name,_function in list(globals().items()):
    if _name.startswith('test_') and callable(_function):
        setattr(FinalSummaryTests,_name,_wrap(_function))

if __name__=='__main__':unittest.main()
