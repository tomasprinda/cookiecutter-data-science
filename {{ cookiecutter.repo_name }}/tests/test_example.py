import io
import unittest
import numpy as np


class TestExample(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_example(self):
        self.assert_true(True)




if __name__ == '__main__':
    unittest.main()
