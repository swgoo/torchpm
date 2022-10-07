from turtle import forward
import unittest

from torchpm.para import *
from torchpm.data import *
from torchpm import predfunc

from torch import nn

from torch.utils.data import DataLoader

if __name__ == '__main__' :
    unittest.main()

class NumericPredFormula(predfunc.PredFormula):
    def forward(self, y: Tensor, t: Tensor, p: Dict[str, Tensor], theta: Dict[str, Tensor], eta: Mapping[str, Tensor]) -> Tensor:
        mat = tc.zeros(2,2, device=y.device)
        mat[0,0] = -p['k_a']
        mat[1,0] = p['k_a']
        mat[1,1] = -p['k_e']
        return mat @ y

class PropAddErrorFunction(predfunc.ErrorFunction):
    def forward(self, y_pred: Tensor, parameters: Dict[str, Tensor], theta: Dict[str, Tensor], eta: Dict[str, Tensor], eps: Dict[str, Tensor]) -> Tensor:
        y = y_pred/parameters['v']
        return y +  y * eps['prop'] + eps['add']

class Comp1GutParameterFunction(predfunc.ParameterFunction):
    def forward(self, para: Dict[str, Tensor], theta: Dict[str, Tensor], eta: Dict[str, Tensor]) -> Dict[str, Tensor]:
        para['k_a'] = theta['k_a']*tc.exp(eta['k_a'])
        para['v'] = theta['v']*tc.exp(eta['v'])
        para['k_e'] = theta['k_e']*tc.exp(eta['k_e'])
        para['AMT'] = para['AMT']*para['BWT']
        return para

class NumericComp1GutParameterFunction(predfunc.ParameterFunction):
    def forward(self, para: Dict[str, Tensor], theta: Dict[str, Tensor], eta: Dict[str, Tensor]) -> Dict[str, Tensor]:
        para['k_a'] = theta['k_a']*tc.exp(eta['k_a'])
        para['v'] = theta['v']*tc.exp(eta['v'])
        para['k_e'] = theta['k_e']*tc.exp(eta['k_e'])
        return para
    
class SymbolicPredFomula(predfunc.PredFormula):
    def forward(self, y: Tensor, t: Tensor, p: Dict[str, Tensor], theta: Dict[str, Tensor], eta: Mapping[str, Tensor]) -> Tensor:
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        # TODO y로 누적을 할것인가 아닌가 결정해야함.
        return  (dose * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))

# class TransformerSymbolicFunction(predfunc.SymbolicPredictionFunction) :

#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.k_a = ThetaInit(0.1, 1.5, 10.)
#         self.v = ThetaInit(0.1, 30., 100.)
#         self.k_e = ThetaInit(0.01, 0.08, 1)

#         self.k_a_eta = EtaDict()
#         self.v_eta = EtaDict()
#         self.k_e_eta = EtaDict()

#         self.prop_err = EpsDict()
#         self.add_err = EpsDict()

#         self.layer_normal = nn.LayerNorm([3,1])
        
#         self.tdl1 = nn.TransformerDecoderLayer(
#                 d_model=1,
#                 nhead=1,
#                 dim_feedforward=1,
#                 dropout=0,
#                 batch_first=True,
#                 norm_first=True)
        
#         self.tdl2 = nn.TransformerDecoderLayer(
#                 d_model=1,
#                 nhead=1,
#                 dim_feedforward=1,
#                 dropout=0,
#                 batch_first=True,
#                 norm_first=True)

#         self.attention = nn.MultiheadAttention(
#                 embed_dim=1,
#                 num_heads=1, batch_first=True)


#         self.linear = nn.Sequential(
#                 nn.Linear(2,2), 
#                 nn.GELU(),
#                 nn.Linear(2,3))

#     def parameter_fuction(self, para):

#         covariates = torch.stack([para['BWT'], para['zero']]).t().unsqueeze(-1)
#         # pk_parameters = torch.stack([self.k_a, self.v, self.k_e]).repeat([para['ID'].size()[0],1]).unsqueeze(-1)
#         # pk_parameters_norm = self.layer_normal(pk_parameters)

#         # output = self.tdl1(tgt = pk_parameters, memory=covariates)
#         # output = self.tdl2(tgt=output, memory=covariates)

#         output, attention = self.attention(covariates, covariates, covariates)
#         # output = self.linear(output.squeeze(-1))

#         # output = pk_parameters + output.unsqueeze(-1)

#         # output = output.squeeze(-1).t()

#         # para['k_a'] = output[0]*tc.exp(self.k_a_eta[get_id(para)])
#         # para['v'] = output[1]*tc.exp(self.v_eta[get_id(para)])
#         # para['k_e'] = output[2]*tc.exp(self.k_e_eta[get_id(para)])
        
        
#         output = self.linear(output.squeeze(-1)).t()

#         para['k_a'] = self.k_a*torch.exp(output[0])*tc.exp(self.k_a_eta[get_id(para)])
#         para['v'] = self.v*torch.exp(output[1])*tc.exp(self.v_eta[get_id(para)])
#         para['k_e'] = self.k_e*torch.exp(output[2])*tc.exp(self.k_e_eta[get_id(para)])

#         para['AMT'] = tc.tensor(320., device=para['ID'].device)
#         return para

#     def pred_function(self, t, p):
#         dose = p['AMT'][0]
#         k_a = p['k_a']
#         v = p['v']
#         k_e = p['k_e']
#         return  (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        
#     def error_function(self, y_pred, p):
#         p['v_v'] = p['v'] 
#         return y_pred +  y_pred * self.prop_err[get_id(p)] + self.add_err[get_id(p)]

class PredFuncTest(unittest.TestCase) :
    def setUp(self) :
        dataframe = pd.read_csv('examples/THEO.csv')
        self.symbolic_dataset = PMDataset(dataframe)
        dataframe = pd.read_csv('examples/THEO_ODE.csv')
        self.numeric_dataset = PMDataset(dataframe)

    def test_simbolic_predfunc(self) :
        k_a = ThetaInit(0.1, 1.5, 10.)
        v = ThetaInit(0.1, 30., 100.)
        k_e = ThetaInit(0.01, 0.08, 1)
        function = predfunc.SymbolicPredictionFunction()
        function.dataset = self.symbolic_dataset
        function.init_theta({'k_a': k_a, 'v': v, 'k_e': k_e})
        function.init_eps_by_names(['prop', 'add'])
        function.init_eta_by_names(['k_a', 'v', 'k_e'])
        function.parameter_functions = [Comp1GutParameterFunction()]
        function.pred_formulae = [SymbolicPredFomula()]
        function.error_functions = [PropAddErrorFunction()]
        
        for batch in DataLoader(dataset=self.symbolic_dataset, batch_size=5) :
            result = function(batch)
            print(result)
        pass
        

    def test_numeric_predfunc(self) :
        k_a = ThetaInit(0.1, 1.5, 10.)
        v = ThetaInit(0.1, 30., 100.)
        k_e = ThetaInit(0.01, 0.08, 1)
        function = predfunc.NumericPredictionFunction()
        function.dataset = self.symbolic_dataset
        function.init_theta({'k_a': k_a, 'v': v, 'k_e': k_e})
        function.init_eps_by_names(['prop', 'add'])
        function.init_eta_by_names(['k_a', 'v', 'k_e'])
        function.parameter_functions = [NumericComp1GutParameterFunction()]
        function.pred_formulae = [NumericPredFormula()]
        function.error_functions = [PropAddErrorFunction()]
        
        for batch in DataLoader(dataset=self.numeric_dataset, batch_size = 5) :
            result = function(batch)
            print(result)
        pass