import unittest
from data import CSVCOVDataset 
# from . import data


class CSVCOVDatasetTest(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_post_init(self):
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        dataset = CSVCOVDataset("./examples/THEO.csv", "./examples/THEO_COV.csv", column_names)

        for data in dataset :
            print(data)

    
    

if __name__ == '__main__' :
    unittest.main()