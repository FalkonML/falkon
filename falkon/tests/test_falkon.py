import unittest

import numpy as np

from falkon.tests.helpers import gen_random

#TODO: FINISH THIS!
class TestFalkon(unittest.TestCase):
    def test_regression(self):
        X = gen_random(1000, 10, np.float64, F=False, seed=21)
        Y = gen_random(1000, 1, np.float64, F=False, seed=21)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
