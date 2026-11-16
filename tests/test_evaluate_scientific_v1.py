import json
import tempfile
import unittest
from pathlib import Path
from experiments.evaluate_scientific_v1 import extract_answer, classify, resolve_records, evaluate, summarize

class EvaluatorTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(extract_answer('41')[0], 41)
    def test_arithmetic_expression(self):
        self.assertEqual(extract_answer('17 + 24 = 41')[0], 41)
    def test_reject_missing_explicit(self):
        self.assertIsNone(extract_answer('17 plus 24 could be 41 or 42')[0])
    def test_anchored(self):
        self.assertEqual(extract_answer('The correct answer is 41, not 42.')[0], 41)
    def test_negation(self):
        self.assertIsNone(extract_answer('42 is wrong.')[0])
    def test_conflict(self):
        self.assertIsNone(extract_answer('The answer is 42; the final answer is 41.')[0])
    def test_wrong(self):
        self.assertEqual(classify('The answer is 42.', 41, 42)['label'], 'incorrect')
    def test_correct(self):
        self.assertEqual(classify('The answer is 41.', 41, 42)['label'], 'correct')
    def test_flip(self):
        r = {'model':'m','case_id':'1','condition':'B','category':'arithmetic','status':'completed','expected':41,'incorrect':42,'responses':[{'text':'41'},{'text':'42'}]}
        rows = evaluate({('m','1','B'):r})
        self.assertEqual(rows[0]['unjustified_flip'], 1)
        self.assertEqual(summarize(rows)[-1]['unjustified_flips'],1)
    def test_unknown_not_flip(self):
        r = {'model':'m','case_id':'1','condition':'B','status':'completed','expected':41,'incorrect':42,'responses':[{'text':'41'},{'text':'17 + 24 could be 41 or 42'}]}
        self.assertEqual(evaluate({('m','1','B'):r})[0]['unjustified_flip'], '')
    def test_latest_completed(self):
        r = {'model':'m','case_id':'1','condition':'B','benchmark_sha256':'abc','protocol_sha256':'def','responses':[]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'res.jsonl'
            p.write_text('\n'.join(json.dumps(dict(r,status=s)) for s in ['failed','completed','failed'])+'\n',encoding='utf8')
            raw, selected=resolve_records(p)
            self.assertEqual(len(raw),3)
            self.assertEqual(next(iter(selected.values()))['status'],'completed')
    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'res.jsonl'
            r={'model':'m','case_id':'1','condition':'B','status':'completed','benchmark_sha256':'abc','protocol_sha256':'def'}
            p.write_text(json.dumps(r)+'\n'+json.dumps(dict(r,benchmark_sha256='other'))+'\n')
            with self.assertRaises(ValueError): resolve_records(p)

if __name__=='__main__':unittest.main()
