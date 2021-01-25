# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
#%%
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from analyze.reasoning_patterns_UNTESTED import parents_of_Y_not_descended_from_X


class TestReasoning(unittest.TestCase):

    # @unittest.skip("")
    # TODO probably delete this (see TODO in function decleration)
    # def test_parents_of_Y_not_descended_from_X(self):
    #     example = get_path_example()
    #     self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X1'), ['X2'])
    #     self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X2'), ['X2'])
    #     self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X3'), ['D', 'X2'])
    #     with self.assertRaises(Exception):      
    #         parents_of_Y_not_descended_from_X(example, 'A', 'X1')
    #     with self.assertRaises(Exception):      
    #         parents_of_Y_not_descended_from_X(example, 'U', 'A')

    




if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestReasoning)
    unittest.TextTestRunner().run(suite)



