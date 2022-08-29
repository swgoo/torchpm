import unittest
from torchpm.para import *

class ParaTest(unittest.TestCase) :
    def setUp(self) -> None:
        return super().setUp()
    
    def test_theta(self) :
        ThetaInit(0.1, 1, 2)