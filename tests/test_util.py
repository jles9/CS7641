import unittest
from util import get_data
import pdb

class TestLoading(unittest.TestCase):
    """
    Class containing test cases for data loading and processing utilities
    """

    def setUp(self) -> None:
        pass


    def test_get_data(self) -> None:
        '''
        Test loading sample Binary Classification Data by checking for correct data dimensions in the dataframe
        '''
        fname = "BinClassTest/train.csv"
        data0 = get_data(fname)
        self.assertEqual(data0.shape[0], 12870)
        self.assertEqual(data0.shape[1], 17)