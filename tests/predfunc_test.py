import unittest

from torchpm.para import *
from torchpm.data import *
from torchpm import predfunc

from torch import nn

from torch.utils.data import DataLoader

if __name__ == '__main__' :
    unittest.main()

class NumericFunction(predfunc.NumericPredictionFunction) :
    def __init__(self, dataset):
        super().__init__(dataset)
        self.theta_0 = ThetaInit(0.1, 1.5, 10)
        self.theta_1 = ThetaInit(0.1, 30, 100)
        self.theta_2 = ThetaInit(0.1, 0.08, 1)

        self.eta_0 = EtaDict()
        self.eta_1 = EtaDict()
        self.eta_2 = EtaDict()

        self.eps_0 = EpsDict()
        self.eps_1 = EpsDict()
    
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

class SymbolicFunction(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset):
        super().__init__(dataset)
        self.k_a = ThetaInit(0.1, 1.5, 10.)
        self.v = ThetaInit(0.1, 30., 100.)
        self.k_e = ThetaInit(0.01, 0.08, 1)

        self.k_a_eta = EtaDict()
        self.v_eta = EtaDict()
        self.k_e_eta = EtaDict()

        self.prop_err = EpsDict()
        self.add_err = EpsDict()
    
    def parameter_fuction(self, para):
        para['k_a'] = self.k_a*tc.exp(self.k_a_eta[get_id(para)])
        para['v'] = self.v*tc.exp(self.v_eta[get_id(para)])
        para['k_e'] = self.k_e*tc.exp(self.k_e_eta[get_id(para)])
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
        return y_pred +  y_pred * self.prop_err[get_id(p)] + self.add_err[get_id(p)]

class TransformerSymbolicFunction(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset):
        super().__init__(dataset)
        self.k_a = ThetaInit(0.1, 1.5, 10.)
        self.v = ThetaInit(0.1, 30., 100.)
        self.k_e = ThetaInit(0.01, 0.08, 1)

        self.k_a_eta = EtaDict()
        self.v_eta = EtaDict()
        self.k_e_eta = EtaDict()

        self.prop_err = EpsDict()
        self.add_err = EpsDict()

        self.layer_normal = nn.LayerNorm([3,1])
        
        self.tdl1 = nn.TransformerDecoderLayer(
                d_model=1,
                nhead=1,
                dim_feedforward=1,
                dropout=0,
                batch_first=True,
                norm_first=True)
        
        self.tdl2 = nn.TransformerDecoderLayer(
                d_model=1,
                nhead=1,
                dim_feedforward=1,
                dropout=0,
                batch_first=True,
                norm_first=True)

        self.attention = nn.MultiheadAttention(
                embed_dim=1,
                num_heads=1, batch_first=True)


        self.linear = nn.Sequential(
                nn.Linear(2,2), 
                nn.GELU(),
                nn.Linear(2,3))

    def parameter_fuction(self, para):

        covariates = torch.stack([para['BWT'], para['zero']]).t().unsqueeze(-1)
        # pk_parameters = torch.stack([self.k_a, self.v, self.k_e]).repeat([para['ID'].size()[0],1]).unsqueeze(-1)
        # pk_parameters_norm = self.layer_normal(pk_parameters)

        # output = self.tdl1(tgt = pk_parameters, memory=covariates)
        # output = self.tdl2(tgt=output, memory=covariates)

        output, attention = self.attention(covariates, covariates, covariates)
        # output = self.linear(output.squeeze(-1))

        # output = pk_parameters + output.unsqueeze(-1)

        # output = output.squeeze(-1).t()

        # para['k_a'] = output[0]*tc.exp(self.k_a_eta[get_id(para)])
        # para['v'] = output[1]*tc.exp(self.v_eta[get_id(para)])
        # para['k_e'] = output[2]*tc.exp(self.k_e_eta[get_id(para)])
        
        
        output = self.linear(output.squeeze(-1)).t()

        para['k_a'] = self.k_a*torch.exp(output[0])*tc.exp(self.k_a_eta[get_id(para)])
        para['v'] = self.v*torch.exp(output[1])*tc.exp(self.v_eta[get_id(para)])
        para['k_e'] = self.k_e*torch.exp(output[2])*tc.exp(self.k_e_eta[get_id(para)])

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
        return y_pred +  y_pred * self.prop_err[get_id(p)] + self.add_err[get_id(p)]

class PredFuncTest(unittest.TestCase) :
    def setUp(self) :
        dataframe = pd.read_csv('examples/THEO.csv')
        self.symbolic_dataset = PMDataset(dataframe)
        dataframe = pd.read_csv('examples/THEO_ODE.csv')
        self.numeric_dataset = PMDataset(dataframe)

    def test_simbolic_predfunc(self) :
        function = SymbolicFunction(self.symbolic_dataset)
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