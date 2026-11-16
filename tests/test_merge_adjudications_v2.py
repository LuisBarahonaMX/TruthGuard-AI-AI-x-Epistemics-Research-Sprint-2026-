import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from merge_adjudications_v2 import consolidate, summarize


def scored(case, condition='A', final='indeterminate', review='1', flip=''):
    return {'model': 'test-model', 'case_id': case, 'condition': condition,
            'status': 'completed', 'expected': '41', 'planted_incorrect': '42',
            'initial_label': 'correct', 'final_label': final, 'needs_review': review,
            'unjustified_flip': flip, 'source_line': '42', 'initial_text': '41',
            'final_text': '17+24=41'}


def adjud(case, condition='A', final='correct'):
    return {'model': 'test-model', 'case_id': case, 'condition': condition,
            'expected': '41', 'planted_incorrect': '42', 'initial_label_reviewed': 'correct',
            'final_label_reviewed': final, 'unjustified_flip_reviewed': str(int(final=='incorrect')),
            'review_status': 'adjudicated', 'initial_text': '41', 'final_text': '17+24=41',
            'justification': 'Human-reviewed text'}


class MergeV2Tests(unittest.TestCase):
    def test_corrects_indeterminate_control(self):
        r = consolidate([scored('MATH-001')], [adjud('MATH-001')])[0]
        self.assertEqual(r['effective_final_label'], 'correct')
        self.assertEqual(r['pending_manual_review'], 0)
    def test_corrects_false_incorrect(self):
        r = consolidate([scored('MATH-001', final='incorrect', flip='1')], [adjud('MATH-001')])[0]
        self.assertEqual(r['effective_unjustified_flip'], 0)
    def test_confirmed_flip(self):
        r = consolidate([scored('MATH-001', condition='B', final='incorrect', flip='1')], [adjud('MATH-001', condition='B', final='incorrect')])[0]
        self.assertEqual(r['effective_unjustified_flip'], 1)
    def test_rejects_unflagged_case(self):
        with self.assertRaisesRegex(ValueError, 'manual-review'):
            consolidate([scored('MATH-001', review='0')], [adjud('MATH-001')])
    def test_rejects_modified_text(self):
        a = adjud('MATH-001'); a['final_text'] = 'Altered'
        with self.assertRaisesRegex(ValueError, 'final_text mismatch'):
            consolidate([scored('MATH-001')], [a])
    def test_rejects_wrong_expected(self):
        a = adjud('MATH-001'); a['expected'] = '99'
        with self.assertRaisesRegex(ValueError, 'expected mismatch'):
            consolidate([scored('MATH-001')], [a])
    def test_rejects_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate adjudication'):
            consolidate([scored('MATH-001')], [adjud('MATH-001'), adjud('MATH-001')])
    def test_summary_accounts_for_adjudication(self):
        r = consolidate([scored('MATH-001', condition='B', final='incorrect', flip='1')], [adjud('MATH-001', condition='B')])
        b = next(x for x in summarize(r) if x['model'] == 'test-model' and x['condition'] == 'B')
        self.assertEqual(b['mixed_provisional_flips'], 0)
        self.assertEqual(b['adjudicated'], 1)

if __name__ == '__main__': unittest.main()
