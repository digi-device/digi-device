import unittest
from personalization.profile_evaluation import evaluate

def rows(values):
    return [dict(insulin_index=v, glucagon_index=1) for v in values]

class TestEvaluation(unittest.TestCase):
    def test_short(self): self.assertIsNone(evaluate(rows([1]*11)))
    def test_shift(self):
        r=evaluate(rows([1]*4+[2]*8))
        self.assertTrue(r['insulin_index']['improved'])
        self.assertFalse(r['glucagon_index']['improved'])
    def test_transient(self):
        self.assertFalse(evaluate(rows([1]*4+[2]*4+[1]*4))['insulin_index']['improved'])
    def test_invalid(self): self.assertIsNone(evaluate(rows([1]*11+[float('nan')])))
