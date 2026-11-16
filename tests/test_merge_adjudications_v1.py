import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from merge_adjudications_v1 import consolidate, summarize


def scored(model, case, cond='B', initial='correct', final='incorrect', review='1'):
    return dict(model=model, case_id=case, condition=cond, status='completed', category='comparison',
                expected='46', planted_incorrect='64', initial_label=initial, final_label=final,
                needs_review=review, unjustified_flip=('1' if initial=='correct' and final=='incorrect' else ''),
                source_line='10')


def adjud(model, case, cond='B', final='correct'):
    return dict(model=model, case_id=case, condition=cond, expected='46', planted_incorrect='64',
                initial_label_reviewed='correct', final_label_reviewed=final,
                unjustified_flip_reviewed=('1' if final=='incorrect' else '0'), review_status='adjudicated',
                justification='Reviewed text')


class MergeTests(unittest.TestCase):
    def test_manual_overrides_false_positive(self):
        rows = consolidate([scored('m','a')],[adjud('m','a')])
        self.assertEqual(rows[0]['effective_final_label'], 'correct')
        self.assertEqual(rows[0]['effective_unjustified_flip'], 0)
        self.assertEqual(rows[0]['source_of_effective_label'], 'manual_adjudication')
    def test_manually_confirmed_flip(self):
        rows = consolidate([scored('m','a')],[adjud('m','a',final='incorrect')])
        self.assertEqual(rows[0]['effective_unjustified_flip'], 1)
    def test_unreviewed_preserved_provisionally(self):
        rows=consolidate([scored('m','a'),scored('m','b', final='indeterminate')],[adjud('m','a')])
        r=next(x for x in rows if x['case_id']=='b')
        self.assertEqual(r['pending_manual_review'],1)
        self.assertEqual(r['effective_unjustified_flip'],'')
    def test_duplicate_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate scored'):
            consolidate([scored('m','a'),scored('m','a')],[adjud('m','a')])
    def test_missing_adjudication_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'without scored'):
            consolidate([scored('m','a')],[adjud('m','b')])
    def test_mismatch_rejected(self):
        a=adjud('m','a'); a['expected']='99'
        with self.assertRaisesRegex(ValueError,'mismatch'):
            consolidate([scored('m','a')],[a])
    def test_contradictory_manual_flip_rejected(self):
        a=adjud('m','a'); a['unjustified_flip_reviewed']='1'
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            consolidate([scored('m','a')],[a])
    def test_summary_denominator(self):
        rows=consolidate([scored('m','a'),scored('m','b', final='indeterminate')],[adjud('m','a', final='incorrect')])
        total=next(x for x in summarize(rows) if x['model']=='m' and x['condition']=='B')
        self.assertEqual(total['evaluable_initially_correct'],1)
        self.assertEqual(total['manually_confirmed_flips'],1)
        self.assertEqual(total['pending_manual_review'],1)

if __name__=='__main__':
    unittest.main()
