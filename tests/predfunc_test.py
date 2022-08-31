import unittest

from torchpm.para import *
from torchpm.data import *
from torchpm import predfunc

from torch.utils.data import DataLoader

if __name__ == '__main__' :
    unittest.main()

class NumericFunction(predfunc.NumericPredictionFunction) :
    def __init__(self, dataset):
        super().__init__(dataset)
        self.theta_0 = ThetaInit(0., 1.5, 10)
        self.theta_1 = ThetaInit(0, 30, 100)
        self.theta_2 = ThetaInit(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()
    
    def parameter_fuction(self, p):
        p['k_a'] = self.theta_0*tc.exp(self.eta_0[get_id(p)])
        p['v'] = self.theta_1*tc.exp(self.eta_1[get_id(p)])
        p['k_e'] = self.theta_2*tc.exp(self.eta_2[get_id(p)])
        return p
    
    def pred_function(self, t, y, p) -> tc.Tensor :
        mat = tc.zeros(2,2, device=y.device)
        mat[0,0] = -p['k_a']
        mat[1,0] = p['k_a']
        mat[1,1] = -p['k_e']
        return mat @ y

    def error_function(self, y_pred: tc.Tensor, parameters: Dict[str, tc.Tensor]) -> tc.Tensor:
        y = y_pred/parameters['v']
        return y +  y * self.eps_0[get_id(parameters)] + self.eps_1[get_id(parameters)]

class BasementFunction(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset):
        super().__init__(dataset)
        self.theta_0 = ThetaInit(0., 1.5, 10.)
        self.theta_1 = ThetaInit(0., 30., 100.)
        self.theta_2 = ThetaInit(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()
    
    def parameter_fuction(self, para):
        para['k_a'] = self.theta_0*tc.exp(self.eta_0[get_id(para)])
        para['v'] = self.theta_1*tc.exp(self.eta_1[get_id(para)])
        para['k_e'] = self.theta_2*tc.exp(self.eta_2[get_id(para)])
        para['AMT'] = tc.tensor(320., device=para['ID'].device)
        return para

    def pred_function(self, t, p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        return  (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        
    def error_function(self, y_pred, p):
        p['v_v'] = p['v'] 
        return y_pred +  y_pred * self.eps_0[get_id(p)] + self.eps_1[get_id(p)]

class PredFuncTest(unittest.TestCase) :
    def setUp(self) :
        dataframe = pd.read_csv('examples/THEO.csv')
        self.symbolic_dataset = PMDataset(dataframe)
        dataframe = pd.read_csv('examples/THEO_ODE.csv')
        self.numeric_dataset = PMDataset(dataframe)

    def test_simbolic_predfunc(self) :
        function = BasementFunction(self.symbolic_dataset)
        for batch in DataLoader(dataset=self.symbolic_dataset, batch_size=5) :
            result = function(batch)
            print(result)
        pass

    def test_numeric_predfunc(self) :
        function = NumericFunction(self.numeric_dataset)
        for batch in DataLoader(dataset=self.numeric_dataset, batch_size = 5) :
            result = function(batch)
            print(result)
        pass