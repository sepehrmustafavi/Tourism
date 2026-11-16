import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from tourism.data import Catalog, make_synthetic_catalog, write_json, read_json
from tourism.experiment import Config, run_episode
from tourism.report import summarize

class ProtocolTests(unittest.TestCase):
    def test_invalid_config(self):
        for kwargs in ({'tau':0},{'response_skip':1.1},{'weights':(1,1,1)},
                       {'judgment_budget':-1},{'particles':1},{'context':(float('nan'),0)}):
            with self.assertRaises(ValueError):Config(**kwargs)

    def test_human_primary_and_no_answer_denominators(self):
        # These are test fixtures relabeled solely to exercise report branching;
        # they never enter research output directories.
        cat=Catalog(make_synthetic_catalog(80))
        cfg=Config(particles=16,actions=4,budget=1,k=3,shortlist=2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for method in ('full','random_matched'):
                result=run_episode(cat,cfg,method)
                for row in result['rows']:
                    row.update(evidence='human_stated_preference',regime='human',user='fixture')
                    row['nll_T']=1. if method=='full' else 2.
                write_json(root/'runs'/f'{method}.json',result)
            summarize(root/'runs',root/'report')
            test=read_json(root/'report'/'summary.json')['paired_tests'][0]
            self.assertEqual(test['metric'],'nll_T')
            self.assertEqual(test['mean_difference'],1.)
            # All-skipped held-out labels retain zero answered and full skip counts.
            skipped=run_episode(cat,replace(cfg,response_skip=1),'full')['rows'][-1]
            self.assertEqual(skipped['heldout_answered_T'],0)
            self.assertIsNone(skipped['nll_T'])
            self.assertGreater(skipped['heldout_skipped_T'],0)

    def test_report_rejects_mixed_actions(self):
        cat=Catalog(make_synthetic_catalog(80));cfg=Config(particles=16,actions=4,budget=0,k=3)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for method in ('full','random_matched'):
                result=run_episode(cat,cfg,method)
                if method=='random_matched':result['rows'][-1]['action_hash']='different'
                write_json(root/'runs'/f'{method}.json',result)
            with self.assertRaisesRegex(ValueError,'Incomparable'):
                summarize(root/'runs',root/'report')

if __name__=='__main__':unittest.main()
