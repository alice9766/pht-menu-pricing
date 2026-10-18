"""MILP formulation and its separate canonical replay checks."""
from fractions import Fraction
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import milp_baseline as mb
from pht_experiment import solve_instance, validate_instance


def payload(alpha=(1,2), c=(0,0), s=(0,0), buyers=((2,1,1),)):
    return {'schema_version':'pht-experiment-v1', 'instance_id':'milp-check',
            'products':[{'q':i+1,'alpha':str(a),'activation_cost':str(cc),'unit_service_cost':str(ss)} for i,(a,cc,ss) in enumerate(zip(alpha,c,s))],
            'buyers':[{'type_id':str(i),'access_cap':cap,'theta':str(t),'count':n} for i,(cap,t,n) in enumerate(buyers)], 'metadata':{}}


class MilpBaselineTests(unittest.TestCase):
    def test_single_product_exact_replay(self):
        p=payload((1,),(1,),('1/2',),((1,2,3),))
        r=mb.solve_milp_instance(p,time_limit=2)
        self.assertEqual(r['raw_status'],0)
        self.assertTrue(r['has_incumbent'])
        self.assertAlmostEqual(r['weak_ic_objective'],3.5)
        self.assertEqual(Fraction(r['accounting']['net_profit']),Fraction(7,2))
        self.assertTrue(r['choices_match'])
        self.assertEqual(set(r['phase_seconds']), {'parse_validation','preprocess_optimization','recovery','replay','total'})
        self.assertLessEqual(r['solver']['solver_time_limit_seconds'],2)
        json.dumps(r,allow_nan=False)

    def test_no_license_empty_and_capacity_zero(self):
        for p,L in [(payload(buyers=((0,2,3),)),None),(payload(),0),(payload(buyers=()),None)]:
            r=mb.solve_milp_instance(p,time_limit=2,capacity=L)
            self.assertEqual(r['raw_status'],0)
            self.assertEqual(Fraction(r['accounting']['net_profit']),0)

    def test_small_random_numeric_objective_bounds(self):
        rng=random.Random(20261018)
        for case in range(12):
            q=1+case%3
            a=sorted(rng.sample(range(1,9),q))
            c=[rng.randrange(5) for _ in a];s=[rng.randrange(4) for _ in a]
            pairs=rng.sample([(cap,t) for cap in range(q+1) for t in range(1,4)],4)
            buyers=[(cap,t,rng.randrange(1,4)) for cap,t in pairs]
            p=payload(a,c,s,buyers)
            opt=float(Fraction(solve_instance(p)['accounting']['net_profit']))
            r=mb.solve_milp_instance(p,time_limit=2)
            self.assertTrue(r['has_incumbent'])
            if r['raw_status']==0:
                self.assertAlmostEqual(r['weak_ic_objective'],opt,places=5)
            if r['profit_upper_bound'] is not None:
                self.assertGreaterEqual(r['profit_upper_bound']+1e-5,opt)
            if r['accounting']:
                self.assertLessEqual(float(Fraction(r['accounting']['net_profit'])),opt+1e-7)

    def test_weak_tie_incumbent_is_not_claimed_canonical(self):
        p=payload((1,2),(0,0),(0,100),((2,1,1),))
        model=mb._build_model(validate_instance(p),2)
        vector=np.zeros(len(model['lower']))
        for j in model['y']:vector[j]=1
        vector[model['p'][0]]=1;vector[model['p'][1]]=2
        vector[model['x'][0,0]]=1;vector[model['payment'][0,0]]=1
        fake=SimpleNamespace(x=vector,status=0,message='mock weak tie',mip_dual_bound=-1,mip_gap=0,mip_node_count=0)
        with patch.object(mb,'milp',return_value=fake):r=mb.solve_milp_instance(p)
        self.assertEqual(r['weak_ic_choices'],[1])
        self.assertEqual(r['buyer_choices'][0]['product'],2)
        self.assertFalse(r['choices_match'])
        self.assertIn('replay_mismatch',r['status'])
        self.assertEqual(Fraction(r['accounting']['net_profit']),-98)
        self.assertNotEqual(r['proof_status'],'exact')
        self.assertEqual(r['profit_upper_bound'],1)

    def test_rational_recovery_of_boundary_roundoff(self):
        p=payload((1,2),(0,0),(0,0),((1,1,2),(2,1,3)))
        data=validate_instance(p)
        restored,info,raw=mb._restore_rational_prices(data,[1,2],
            {1:Fraction('1.0000000000000002'),2:Fraction('2.000000000000001')},[1,2],3)
        self.assertEqual(restored,{1:Fraction(1),2:Fraction(2)})
        self.assertTrue(info['candidate_exact_weak_ic_valid'])
        self.assertFalse(info['raw_exact_weak_ic_valid'])
        self.assertFalse(info['numeric_recovery_issue'])

    def test_recovery_unanchored_component_and_genuine_failure(self):
        data=validate_instance(payload((1,2),(0,0),(0,0),((2,2,1),)))
        restored,info,_=mb._restore_rational_prices(data,[1,2],
            {1:Fraction('0.5000000000000001'),2:Fraction('2.5000000000000004')},[2],5)
        self.assertEqual(restored[2]-restored[1],2)
        self.assertFalse(info['numeric_recovery_issue'])
        _,bad,_=mb._restore_rational_prices(data,[1,2],
            {1:Fraction(5),2:Fraction(6)},[2],7)
        self.assertTrue(bad['numeric_recovery_issue'])

    def test_no_incumbent_not_replaced_by_empty_menu(self):
        fake=SimpleNamespace(x=None,status=1,message='limit',mip_dual_bound=-7,mip_gap=None,mip_node_count=0)
        with patch.object(mb,'milp',return_value=fake):r=mb.solve_milp_instance(payload())
        self.assertFalse(r['has_incumbent']);self.assertIsNone(r['accounting']);self.assertIsNone(r['menu'])
        self.assertEqual(r['profit_upper_bound'],7)
        self.assertIsNone(r['weak_ic_objective'])
        json.dumps(r,allow_nan=False)


if __name__=='__main__':unittest.main(verbosity=2)
