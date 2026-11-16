import importlib.util
import unittest
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'experiments'/'analyze_statistics_v1.py'
spec=importlib.util.spec_from_file_location('stats',P)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StatsTests(unittest.TestCase):
 def test_wilson_zero(self):
  lo,hi=module.wilson(0,140)
  self.assertEqual(lo,0)
  self.assertGreater(hi,0)
 def test_wilson_full(self):
  lo,hi=module.wilson(20,20)
  self.assertEqual(hi,1)
  self.assertLess(lo,1)
 def test_wilson_empty(self):
  self.assertEqual(module.wilson(0,0),(None,None))
 def test_mcnemar_4_vs_zero(self):
  self.assertAlmostEqual(module.exact_mcnemar(4,0),.125)
 def test_mcnemar_1_1(self):
  self.assertEqual(module.exact_mcnemar(1,1),1)
 def test_mcnemar_no_discordance(self):
  self.assertEqual(module.exact_mcnemar(0,0),1)
 def test_metric_flip(self):
  data=[{'effective_initial_label':'correct','effective_final_label':'incorrect','condition':'B','_adjudicated':True},
        {'effective_initial_label':'correct','effective_final_label':'correct','condition':'B','_adjudicated':False}]
  m=module.metric(data,'B','condition')
  self.assertEqual((m['n_flips_mixed'],m['n_evaluable_initial_correct']), (1,2))
  self.assertEqual(m['n_confirmed_flips_reviewed'],1)
 def test_indeterminate_excluded(self):
  m=module.metric([{'effective_initial_label':'correct','effective_final_label':'indeterminate','condition':'C','_adjudicated':False}], 'C','condition')
  self.assertEqual(m['n_evaluable_initial_correct'],0)
 def test_paired_discordances(self):
  p=module.pair_stats([{'flip_B':1,'flip_C':0},{'flip_B':0,'flip_C':0},{'flip_B':0,'flip_C':1}])
  self.assertEqual((p['B_only'],p['C_only'],p['both'],p['neither']), (1,1,0,1))
 def test_label_guard(self):
  with self.assertRaises(ValueError):module.norm_label('maybe')
 def test_cluster_bootstrap_reproducible(self):
  pairs=[{'case_id':'x','flip_B':1,'flip_C':0},{'case_id':'y','flip_B':0,'flip_C':0}]
  self.assertEqual(module.bootstrap_by_case(pairs,1000),module.bootstrap_by_case(pairs,1000))

if __name__=='__main__':unittest.main()
