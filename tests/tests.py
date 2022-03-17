
import logging
from re import X
import unittest
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torchpm import funcgen, scale, predfunction, diff, models
from torchpm.data import CSVDataset

from transformers import DistilBertModel, DistilBertConfig


class TestTemplate(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

class CustomTransformerDecoderLayer(nn.Module) :
            def __init__(self, d_model, nhead) -> None:
                super().__init__()
                self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=0, batch_first=True)
                self.multi_head_attention = nn.MultiheadAttention(d_model, nhead, dropout=0, batch_first=True)
                self.lin1 = nn.Sequential(nn.Linear(d_model,d_model),
                                        nn.GELU(),
                                        nn.Linear(d_model,d_model))
                self.lin2 = nn.Sequential(nn.Linear(d_model,d_model),
                                        nn.GELU(),
                                        nn.Linear(d_model,d_model),)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.norm3 = nn.LayerNorm(d_model)
                self.norm4 = nn.LayerNorm(d_model)
            
            def forward(self, tgt, memory, memory_value, attn_mask=None, key_padding_mask=None):    
                # sa, _ = self.self_attention(memory, memory, memory_value, key_padding_mask = key_padding_mask)
                # x1 = self.norm1(memory_value + sa)
                # x1 = self.norm2(self.lin1(x1) + x1)
                # x1 = self.lin1(memory_value + sa)

                attn, attention_score = self.multi_head_attention(tgt, memory, memory_value, attn_mask=attn_mask, key_padding_mask = key_padding_mask)
                # attn, attention_score = self.multi_head_attention(tgt, x1, x1, attn_mask=attn_mask, key_padding_mask = key_padding_mask)
                # x2 = self.norm3(tgt + attn)
                # x2 = self.norm4(self.lin2(x2) + x2)
                x2 = self.norm4(self.lin2(tgt + attn))
                return x2, attention_score

class TotalTest(unittest.TestCase) :

    def setUp(self):
        dataset_file_path = './examples/THEO.csv'
        self.column_names =  ['ID', 'AMT', 'TIME', 'DV', 'BWT', 'BWTZ', 'CMT', "MDV", "RATE", "tmpcov", "tmpcov2", "tmpcov3", "tmpcov4", "tmpcov5"]

        self.device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        self.dataset = CSVDataset(dataset_file_path, self.column_names, self.device)

    def tearDown(self):
        pass

    def test_transformers(self) :
        config = DistilBertConfig(attention_dropout = 0, dim=1, dropout=0.0, hidden_dim=2, n_heads=1, n_layers=3, seq_classif_dropout=0.0, vocab_size=5)
        model = DistilBertModel(config)
        r = model(input_ids=tc.tensor([[1,2,3]]), output_attentions=True, output_hidden_states = True)
        for k, v in r.items() :
            print(k)
            print(v)
    
    def test_basement_model(self):
        device = self.device
        dataset = self.dataset
        column_names = self.column_names

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, theta, eta, cmt, amt, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :

                k_a = theta[0]*tc.exp(eta[0])
                v = theta[1]*tc.exp(eta[1])
                k_e = theta[2]*tc.exp(eta[2])

                # k_a = theta[0]*tc.exp(eta[0])
                # v = theta[1]*tc.exp(eta[1])
                # k_e = theta[2]*tc.exp(eta[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e, 'eta': eta, 'bwt': bwt}

        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                dose = 320
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                
                return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, cmt, pk, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 32,  0.08], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 3
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 2
        sigma_inits = [tc.tensor([0.0177, 0.0762], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for k, v in eval_values["parameters"].items() :
            print(k)
            print(v)

    def test_pred_time(self):
        device = self.device
        dataset = self.dataset
        column_names = self.column_names

        '''
        TODO
        
        attention_score 잘 안됨. 그냥 일단 R^2로 cov 한 번 걸러준 후에 
        non-numeric categorical covariate는 따로 one hot vector화 해서 처리하는 게 낫다.
        category가 10개가 넘어가는 non-numeric categorical value는 learnable embbeding을 고려해야 할수도 있다. 단 PK는 데이터가 작으므로 저차원으로 해야한다.
        ordered categorical covariate는 scale을 정확하게 모르는 경우이니 아예 scale을 추정하도록 하는 것이 맞는데 어떻게 하나?
           예를 들면 알러지 단계, 1,2,3 단계가 있다고 하자 사실은 수치가 10 20 50일수도 있다.
        
        cov의 갯수가 많으면 attention_score를 활용하는 것이 도움이 될수도 있다. 하지만 cov의 갯수가 적은 경우에는 별로 효과가 없는듯 하고 심지어 유효한 cov의 attention score가 낮게 나오는 현상도 나타났다.

        활용법
        *SCM대비 장점? 수식을 넣어줄 필요가 없어서 조합수가 줄어든다.
            1. basement 모델을 최적화를 한다.
            2. eta와 쌍을 이루어서 상관계수 값으로 -0.1 ~ 0.1 사이인 cov들은 제거한다.
            3. 2에서 통과한 cov들로 최대로 최적화 되는 ofv 를 구한다.
            4. cov를 하나씩 빼면서 3.84이상 ofv가 증가한 경우 해당 cov는 제거한다.
            5. 4까지 통과한 cov들로 SCM을 한다.

        '''

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()
                #TODO:cov의 평균값을 받아서 scale문제를 해결
                #TODO: Normalization Layer 사용시 주의. 
                #TODO: 모델 개선 scale문제로 작동을 안하는 경우임.
                self.lin = nn.Sequential(nn.Linear(1,3),
                                        nn.LeakyReLU(),
                                        nn.Linear(3,3))

            def forward(self, theta, eta, cmt, amt, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :

                cov = tc.stack([bwtz]) # OFV 80
                # cov = tc.stack([bwt/70]) # OFV 87.43
                cov = self.lin(cov.t()).t()



                k_a = theta[0]*tc.exp(eta[0]*cov[0])
                v = theta[1]*tc.exp(eta[1]*cov[1])
                k_e = theta[2]*tc.exp(eta[2]*cov[2])

                # k_a = theta[0]*tc.exp(eta[0])
                # v = theta[1]*tc.exp(eta[1])
                # k_e = theta[2]*tc.exp(eta[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e}

        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                dose = 320
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                
                return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, cmt, pk, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 32,  0.08], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 3
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 2
        sigma_inits = [tc.tensor([0.0177, 0.0762], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())
        assert(0, 0)
    
    def test_pred_time_attention(self):

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = self.dataset

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()
                #TODO:cov의 평균값을 받아서 scale문제를 해결
                #TODO: Normalization Layer 사용시 주의. 
                #TODO: 모델 개선 scale문제로 작동을 안하는 경우임.
                d_model = 1
                nhead = 1
                # self.norm1 = nn.LayerNorm(2)
                self.lin = nn.Sequential(nn.Linear(5,3))

                self.eta_embedding = nn.Embedding(4,1, padding_idx=0) 
                self.cov_embedding = nn.Embedding(6,1, padding_idx=0)

                # self.transformer_encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=0)
                # self.transformer_encoder_layers = nn.TransformerEncoder()

                self.transformer_decoder1 = CustomTransformerDecoderLayer(d_model, nhead)
                self.transformer_decoder2 = CustomTransformerDecoderLayer(d_model, nhead)
                self.transformer_decoder3 = CustomTransformerDecoderLayer(d_model, nhead)
                self.transformer_decoder4 = CustomTransformerDecoderLayer(d_model, nhead)
                self.transformer_decoder5 = CustomTransformerDecoderLayer(d_model, nhead)
                self.transformer_decoder6 = CustomTransformerDecoderLayer(d_model, nhead)

                # self.lin = nn.Sequential(nn.Linear(3,3))

            def forward(self, theta, eta, cmt, amt, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :


                #TODO cov랑 eta중 긴쪽으로 길이 맞춰주고 mask 처리해서 처리
                
                cov = tc.stack([bwtz, tmpcov2, tmpcov3, tmpcov4, tmpcov5]).t()

                embedded_eta = self.eta_embedding(tc.tensor([1,2,3], device = device)).unsqueeze(0).repeat([eta.size()[-1],1,1])

                embedded_cov = self.cov_embedding(tc.tensor([1,2,3], device = device)).unsqueeze(0).repeat([eta.size()[-1],1,1])

                cov_for_attn = tc.stack([bwtz, tmpcov2, tmpcov3]).t().unsqueeze(-1)

                cov_mask = tc.tensor([0,0,0], device=device).repeat([eta.size()[-1],1])

                attn_mask = tc.tensor([[0,0,0], [0,0,0], [0,0,0]], device=device, dtype=tc.bool)

                x, attention_score = self.transformer_decoder1(embedded_eta, embedded_cov, embedded_cov, attn_mask = attn_mask, key_padding_mask = cov_mask)
                x = x.nan_to_num()
                x, attention_score = self.transformer_decoder2(embedded_eta, x,            x, attn_mask = attn_mask, key_padding_mask = cov_mask)
                x = x.nan_to_num()
                x, attention_score = self.transformer_decoder3(embedded_eta, x,            x, attn_mask = attn_mask, key_padding_mask = cov_mask)
                x = x.nan_to_num()
                # x, attention_score = self.transformer_decoder4(embedded_eta, x, cov_for_attn, attn_mask = attn_mask, key_padding_mask = cov_mask)
                # x, attention_score = self.transformer_decoder5(embedded_eta, x, cov_for_attn, attn_mask = attn_mask, key_padding_mask = cov_mask)
                # x, attention_score = self.transformer_decoder6(embedded_eta, x, cov_for_attn, attn_mask = attn_mask, key_padding_mask = cov_mask)
                x = x.squeeze(-1)
                cov = self.lin(cov)
                x_eff = x
                x = cov * x
                x = x.t()

                # k_a = theta[0] * tc.exp(x[0])
                # v = theta[1] * (bwt/70)**0.25 * tc.exp(x[1])
                # k_e = theta[2] * tc.exp(x[2])

                k_a = theta[0] * tc.exp(eta[0] + x[0]) 
                v = theta[1] * tc.exp(eta[1] + x[1]) 
                k_e = theta[2] * tc.exp(eta[2] + x[2])

                # k_a = theta[0] * tc.exp(eta[0]) * x[0] 
                # v = theta[1] * tc.exp(eta[1]) * x[1]
                # k_e = theta[2] * tc.exp(eta[2]) * x[2]

                # k_a = theta[0] * tc.exp(eta[0]*x[0])
                # v = theta[1] * tc.exp(eta[1]*x[1])
                # k_e = theta[2] * tc.exp(eta[2]*x[2])

               # k_a = theta[0]*tc.exp(eta[0])
                # v = theta[1]*tc.exp(eta[1])* (bwt/70)**0.25
                # k_e = theta[2]*tc.exp(eta[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e, "attention_score": attention_score, 'eta': eta, 'x_eff': x_eff}
                # return {'k_a': k_a, 'v' : v, 'k_e': k_e, 'eta': eta, 'x': x}

        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                dose = 320
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                
                return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, cmt, pk, bwt, bwtz, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 32,  0.08], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 3
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 2
        sigma_inits = [tc.tensor([0.0177, 0.0762], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = self.column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        model.fit_population(learning_rate =1, tolerance_grad = 1e-5, tolerance_change= 1e-9)

        for p in model.descale().named_parameters():
            print(p)

        # print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for k, v in eval_values["parameters"].items() :
            print(k)
            print(v)
        
        print('total_loss', eval_values['total_loss'])
        assert(eval_values['total_loss'] < 90)

    def test_pred_amt(self):
        dataset_file_path = './examples/THEO_AMT.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE", "tmpcov2", "tmpcov3", "tmpcov4", "tmpcov5"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, theta, eta, cmt, amt, bwt, tmpcov, tmpcov2, tmpcov3, tmpcov4, tmpcov5) :

                k_a = 1.4901 
                v = 32.4667
                k_e = 0.0873

                amt = amt * theta[0] #* bwt
                return {'k_a': k_a, 'v' : v, 'k_e': k_e, 'AMT': amt, 'bwt': bwt}
        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                # dose = pk['AMT']
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                # print(dose)
                return (amt / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, cmt, pk, bwt, tmpcov) :
                return y_pred + eps[0] #+  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 1
        theta_init = tc.tensor([200.], device=device)
        theta_lower_boundary  = tc.tensor([0.], device = device)
        theta_upper_boundary  = tc.tensor([500.], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 0
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = None # []
        omega_scales = None#[scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 1
        sigma_inits = [tc.tensor([0.1], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        # model.differential_module.sigma[0].requires_grad = False

        model.fit_population(learning_rate = 1.3, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)
        
        eval_values = model.descale().evaluate()
        for k, v in eval_values.items() :
            if k == "parameters" :
                for kk, vv in v :
                    print(kk)
                    print(vv)

        #TODO
        # print(model.descale().covariance_step())
        assert(0, 0)
    
    #TODO Test
    def test_ODE(self):

        dataset_file_path = './examples/THEO_ODE.csv'

        column_names = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT', 'COV']

        device = tc.device("cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(nn.Module) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, theta, eta, cmt, amt, cov) :
                k_a = theta[0]*tc.exp(eta[0])
                v = theta[1]*tc.exp(eta[1])
                k_e = theta[2]*tc.exp(eta[2])
                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(nn.Module) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                mat = tc.zeros(2,2, device=y.device)
                mat[0,0] = -pk['k_a']
                mat[1,0] = pk['k_a']
                mat[1,1] = -pk['k_e']
                return mat @ y
        pred_fn = PredFunction()

        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, cmt, pk, COV) :
                v = pk['v']
                return y_pred/v + y_pred/v * eps[0] + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 30,  0.1], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)

        eta_size = 3
        omega_init = [tc.tensor([0.2,
                                0.1, 0.2,
                                0.1, 0.1, 0.2], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_init[0], omega_diagonals[0])]

        eps_size = 2
        sigma_init = [tc.tensor([0.2, 0.1], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_init[0], sigma_diagonals[0])]

        pred_function_module = predfunction.PredictionFunctionByODE(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)

        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)
        
        print(model.descale().covariance_step())
        assert(0, 0)


if __name__ == '__main__' :
    unittest.main()
