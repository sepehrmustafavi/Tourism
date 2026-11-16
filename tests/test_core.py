import unittest
from itertools import combinations
import tempfile
from pathlib import Path
import numpy as np
from tourism.core import Posterior, Observation, query_value, regret, build_actions, list_scores
from tourism.data import Catalog, make_synthetic_catalog, validate_features, write_json
from tourism.experiment import Config, run_episode, run_suite, split_catalog
from tourism.human import validate_human

class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.cat=Catalog(make_synthetic_catalog(80));self.phi=self.cat.phi;self.psi=self.cat.psi
        self.ctx=(3.,0.);self.chi=self.cat.context(self.ctx);self.p=Posterior(6,7,4,particles=128,seed=3)
    def test_ordered_logistic(self):
        p=self.p.comparison(np.array([-1000.,-1.,0.,1.,1000.]))
        np.testing.assert_allclose(p.sum(1),1);self.assertTrue(np.all(p>=0));self.assertAlmostEqual(p[2,0],p[2,1])
    def test_matched_outcomes(self):
        q=self.p.outcome_probs(self.phi,self.psi,self.chi,0,1)
        self.assertEqual(q.shape,(128,16));np.testing.assert_allclose(q.sum(1),1)
    def test_skip(self):
        s=self.p.state.copy();w=self.p.weights.copy()
        self.p.update(Observation('skip',0,1,'T','skip',self.ctx),self.phi,self.psi,self.cat.context)
        np.testing.assert_array_equal(s,self.p.state);np.testing.assert_allclose(w,self.p.weights)
    def test_deduplication(self):
        o=Observation('unique',0,1,'V','A',self.ctx)
        self.assertTrue(self.p.update(o,self.phi,self.psi,self.cat.context));w=self.p.weights.copy()
        self.assertFalse(self.p.update(o,self.phi,self.psi,self.cat.context));np.testing.assert_array_equal(w,self.p.weights)
    def test_historical_context(self):
        seen=[]
        def fn(c):seen.append(c);return self.cat.context(c)
        self.p.update(Observation('old',0,1,'T','A',(2.,1.)),self.phi,self.psi,fn)
        self.assertTrue(all(x==(2.,1.) for x in seen))
    def test_logit_jacobian(self):
        s=np.zeros_like(self.p.state)
        self.assertAlmostEqual(self.p.log_prior(s)[0],-6*(self.p.alpha+self.p.beta)*np.log(2))
    def test_evsi_explicit_regret(self):
        rng=np.random.default_rng(5);j=rng.random((128,15));w=rng.dirichlet(np.ones(128));l=rng.dirichlet(np.ones(16),size=128)
        expected=regret(j,w)
        for col in l.T:
            prob=w@col;expected-=prob*regret(j,w*col/prob)
        self.assertAlmostEqual(query_value(j,w,l),expected,places=12)
    def test_evsi_known_answer(self):
        j=np.array([[1.,0.],[0.,1.]]);w=np.array([.5,.5])
        self.assertAlmostEqual(query_value(j,w,np.eye(2)),.5)
        self.assertAlmostEqual(query_value(j,w,np.ones((2,1))),0)
        self.assertAlmostEqual(query_value(j,w,np.eye(2),.2),.3)
    def test_hypothetical_no_mutation(self):
        s=self.p.state.copy();w=self.p.weights.copy();q=self.p.outcome_probs(self.phi,self.psi,self.chi,0,1)
        query_value(np.zeros((128,2)),w,q)
        np.testing.assert_array_equal(s,self.p.state);np.testing.assert_array_equal(w,self.p.weights)
    def test_rejuvenation(self):
        s=self.p.state.copy();self.p.resample_move(self.phi,self.psi,self.cat.context)
        self.assertGreater(self.p.mh_acceptances,0);self.assertFalse(np.array_equal(s,self.p.state))
        np.testing.assert_allclose(self.p.weights,np.full(128,1/128))
    def test_action_pool(self):
        args=(self.phi,self.psi,self.chi,self.cat.destinations,self.cat.distance,range(8),3,16)
        a=build_actions(*args);self.assertTrue(a);self.assertEqual(a,build_actions(*args))
        for row in a:self.assertEqual(len({self.cat.destinations[i] for i in row}),3)
    def test_singleton(self):
        j,_,_,d=list_scores(np.zeros((2,2)),np.zeros((2,2)),[(0,)],np.zeros((2,2)),(.2,.6,.2))
        self.assertEqual(d[0],0);np.testing.assert_allclose(j,.4)
    def test_critical_missing(self):
        self.cat.items[0]['attributes']['budget']=None
        self.assertNotIn(0,self.cat.feasible(self.ctx));self.assertEqual(self.cat.feasible(self.ctx,require_verified=True),[])
    def test_evidence(self):
        item=self.cat.items[0]
        self.assertEqual(validate_features(item,{'nature':{'value':1,'span':'invented quotation'}}),{})
        self.assertEqual(validate_features(item,{'nature':{'value':float('nan'),'span':item['text']}}),{})
    def test_disjoint_split(self):
        a,b,c=[set(split_catalog(self.cat,s).destinations) for s in ('train','validation','test')]
        self.assertFalse(a&b or b&c or a&c)
    def test_episode(self):
        cfg=Config(particles=32,actions=8,candidates=10,shortlist=3,budget=2,k=3)
        a=run_episode(self.cat,cfg,'full');b=run_episode(self.cat,cfg,'random_matched')
        self.assertEqual(a['rows'][0]['action_hash'],b['rows'][0]['action_hash'])
        self.assertTrue(all(r['regret']>=-1e-12 and r['constraint_violations']==0 for r in a['rows']))
        self.assertLessEqual(len(a['queries']),2)
    def test_resume(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cat.json';write_json(path,self.cat.payload)
            kwargs=dict(catalog_path=path,out=Path(td)/'out',config=Config(particles=16,actions=4,candidates=8,budget=0,k=2),methods=['full'],seeds=[11],users=1,regimes=['mixed'])
            self.assertEqual(run_suite(**kwargs)['completed_now'],1);self.assertEqual(run_suite(**kwargs)['completed_now'],0)
    def test_human_all_skip(self):
        cat=self.cat.subset(range(12));pairs=list(combinations(range(12),2))
        h={'item_ids':[i['id'] for i in cat.items],'context':[3.,0.],'weights':[.25,.6,.15],
           'warmup_pairs':pairs[:6],'query_pairs':pairs[6:20],'test_pairs':pairs[20:32],
           'ratings':{'V':[2]*12,'T':[2]*12},'answers':{},'catalog_hash':cat.hash,'consent':True,'completed':True}
        for i,j in pairs:
            for t in ('V','T'):h['answers'][f'{i}:{j}:{t}']='skip'
        validate_human(h,cat)
        r=run_episode(cat,Config(particles=32,actions=8,budget=2),'random_matched',human=h)
        self.assertEqual(r['rows'][-1]['judgments_answered'],0);self.assertIsNone(r['rows'][-1]['nll_T'])
    def test_human_leak(self):
        h={'consent':True,'completed':True,'catalog_hash':self.cat.hash,'item_ids':['1','2'],
           'warmup_pairs':[[0,1]],'test_pairs':[[0,1]],'query_pairs':[]}
        with self.assertRaises(ValueError):validate_human(h,self.cat)

if __name__=='__main__':unittest.main()
