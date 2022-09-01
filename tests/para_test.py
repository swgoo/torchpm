import unittest
from torchpm.para import *
from torchpm.misc import *

class ParaTest(unittest.TestCase) :
    def setUp(self) -> None:
        return super().setUp()
    
    def test_theta(self) :
        ThetaInit(0.1, 1, 2)

    @classmethod
    def get_omega(cls):
        return CovarianceVectorInitList(
            [CovarianceVectorInit((0.1,0.1,0.1),('k_a_eta','v_eta','k_e_eta'))])

    @classmethod
    def get_sigma(cls):
        return CovarianceVectorInitList(
            [CovarianceVectorInit((0.1,0.1),('prop_err','add_err'))])


    def test_covariance_vector_init_list(self) :
        cov = self.get_omega()
        vector_list = cov.covariance_list()
        scaler_list = cov.scaler_list()
        covariance = get_covariance(vector_list, scaler_list, True)
    
    def test_covariance_vector_list_and_scaler_list(self):
        cov = CovarianceVectorInitList(
            [CovarianceVectorInit((1.,),('eta1',)),CovarianceVectorInit((1.,0.2,1.1),('eta2','eta3'),is_diagonal=False)])
        vector_list = cov.covariance_list()
        scaler_list = cov.scaler_list()
        vector_list[0] = CovarianceVector(tensor([3.]), ('eta0',))
        scaler_list[0] = None # type: ignore
        covariance = get_covariance(vector_list, scaler_list, True)
        pass