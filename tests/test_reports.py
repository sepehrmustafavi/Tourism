import unittest
from pathlib import Path
from dataclasses import replace
import tempfile
import numpy as np
from tourism.core import Observation,Posterior
from tourism.data import Catalog,make_synthetic_catalog,write_json,read_json
from tourism.experiment import Config,run_episode,simulated_answer,truth
from tourism.report import paired_test,holm,summarize
from tourism.human import make_survey

class ReportTests(unittest.TestCase):
    def test_holm(self):
        ts=holm([{'p':.01},{'p':.04},{'p':.03}])
        np.testing.assert_allclose([x['p_holm'] for x in ts],[.03,.06,.06])
    def test_identical_paired(self):
        r=paired_test([0,0,0],repeats=100)
        self.assertEqual(r['p'],1);self.assertEqual(r['mean_difference'],0)
    def test_simulator_independent_of_fit_tau(self):
        cat=Catalog(make_synthetic_catalog());cfg=Config();v,t=truth(cat,0,11,'mixed',cfg)
        alternative=replace(cfg,tau=.7,particles=256,cost=.04)
        for pair in [(0,1),(2,3),(4,5)]:
            for target in ['V','T']:
                self.assertEqual(simulated_answer(v,t,pair,target,0,11,'mixed',cfg),
                                 simulated_answer(v,t,pair,target,0,11,'mixed',alternative))
    def test_conflicting_duplicate_event(self):
        cat=Catalog(make_synthetic_catalog());p=Posterior(6,7,4,32)
        p.update(Observation('id',0,1,'V','A',(3.,0.)),cat.phi,cat.psi,cat.context)
        with self.assertRaises(ValueError):p.update(Observation('id',0,1,'V','B',(3.,0.)),cat.phi,cat.psi,cat.context)
    def test_judgment_cap(self):
        cat=Catalog(make_synthetic_catalog());cfg=Config(particles=32,actions=8,budget=10,judgment_budget=3,k=3)
        r=run_episode(cat,cfg,'random_matched')
        self.assertEqual(r['rows'][-1]['judgments_requested'],2)
        t=run_episode(cat,cfg,'travel_only')
        self.assertLessEqual(t['rows'][-1]['judgments_requested'],3)
    def test_all_methods(self):
        from tourism.experiment import METHODS
        cat=Catalog(make_synthetic_catalog());cfg=Config(particles=16,actions=4,budget=1,k=3,shortlist=2)
        for method in METHODS:
            r=run_episode(cat,cfg,method)
            self.assertTrue(np.isfinite(r['rows'][-1]['utility']),method)
    def test_survey_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cat=Catalog(make_synthetic_catalog(100));write_json(root/'cat.json',cat.payload)
            made=make_survey(root/'cat.json',root/'surveys','P001')
            protocol=read_json(made['protocol'])
            self.assertEqual(len(protocol['query_pairs']),24)
            self.assertFalse(set(map(tuple,protocol['test_pairs']))&set(map(tuple,protocol['query_pairs'])))
            for method in ['full','random_matched']:
                r=run_episode(cat,Config(particles=16,actions=4,budget=1,k=3,shortlist=2),method)
                write_json(root/'runs'/f'{method}.json',r)
            result=summarize(root/'runs',root/'report')
            self.assertEqual(result['groups'],2);self.assertTrue((root/'report'/'users.csv').exists())

if __name__=='__main__':unittest.main()
