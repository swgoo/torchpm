import unittest
from torchpm.para import *
from torchpm.misc import *

class ParaTest(unittest.TestCase) :
    def setUp(self) -> None:
        return super().setUp()
    
    def test_theta(self) :
        ThetaInit(0.1, 1, 2)

    def test_covariance(self) :
        cov = CovarianceVectorInitList(
            [CovarianceVectorInit((1.,),('eta1',)),CovarianceVectorInit((1.,0.2,1.1),('eta2','eta3'),is_diagonal=False)])
        vector_list = cov.covariance_list
        scaler_list = cov.scalers
        covariance = get_covariance(vector_list, scaler_list)
        pass