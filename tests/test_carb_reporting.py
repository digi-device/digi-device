import unittest
from carb_reporting import CarbReporting

class TestCarbReporting(unittest.TestCase):
    def test_fixed(self):
        for f in (0, .5, 1):
            r = CarbReporting(f)
            self.assertEqual(r.report(80),80*f)
            self.assertEqual(r.report(0),0)
    def test_random_per_meal(self):
        a = CarbReporting(randomized=True,seed=7)
        b = CarbReporting(randomized=True,seed=7)
        values = [0,20,10,0,50,0]
        x = [a.report(v) for v in values]
        self.assertEqual(x,[b.report(v) for v in values])
        self.assertAlmostEqual(x[1]/20,x[2]/10)
        self.assertNotEqual(x[1]/20,x[4]/50)
        self.assertTrue(all(0 <= y <= v for y,v in zip(x,values)))
    def test_invalid(self):
        for f in (-1,1.1,float('nan')):
            with self.assertRaises(ValueError): CarbReporting(f)
