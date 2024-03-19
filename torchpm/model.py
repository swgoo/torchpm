import abc
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import torchode
from lightning import LightningModule
from torch import nn, Tensor, tensor
import torch

from .parameter import RandomEffect, RandomEffectConfig

from .data import Dict, Tensor
from .data import *

class ODESolver(LightningModule):
    # Ordinary Differential Equation Solver
    @torch.no_grad()
    def __init__(
        self, 
        atol=1e-5, 
        rtol=1e-3,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #ODE solver
        term = torchode.ODETerm(self.ode, with_args=True)
        step_method = torchode.Tsit5(term=term)
        step_size_controller = torchode.IntegralController(atol=atol, rtol=rtol, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
        self.ode_solver = torch.compile(solver)

    @abc.abstractmethod
    def ode(self, t:Tensor, y:Tensor, kwargs:Dict[str,Tensor]):...

    def forward(self, 
            time : Tensor,
            init : Tensor,
            kwargs : Dict[str,Tensor]):
        for time_d, init_d in zip(time.size(), init.size()[:2]):
            assert time_d == init_d

        ys = []
        pre_times = time[:,0]
        pre_ys = init[:,0,:]
        ys.append(pre_ys)
        for i, (cur_times, cur_init_values) in enumerate(zip(time.T[1:], init.permute(1,0,2)[1:])) :
            init_values = pre_ys + cur_init_values
            init_state = torchode.InitialValueProblem(
                y0=init_values,
                t_start=pre_times,
                t_eval= cur_times.unsqueeze(-1))
            cur_kwargs = {}
            for k,v in kwargs.items():
                cur_kwargs[k] = v[:,i]

            sol = self.ode_solver.solve(
                init_state,
                args=cur_kwargs)
            cur_zs = sol.ys[:,0]
            ys.append(cur_zs) 
            pre_ys = cur_zs
            pre_times = cur_times
        return torch.stack(ys).permute(1,0,2) # [Time, batch, feat.] -> [batch,time,feat.]

@dataclass
class FFNConfig:
    # Feed Forward Net Configuration
    dims : Tuple[int,...] = (1, 1)
    hidden_norm_layer : bool = True
    hidden_act_fn : str | None = 'SiLU'
    output_act_fn : str | None = None
    bias : bool = True
    dropout : float = 0.2

class FFN(nn.Module):
    def __init__(
            self,
            config : FFNConfig,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        
        net = []
        pre_dim = config.dims[0]
        for next_dim in config.dims[1:-1]:
            lin = nn.Linear(pre_dim, next_dim, bias=config.bias)
            net.append(lin)
            if config.hidden_norm_layer :
                norm = nn.LayerNorm(next_dim)
                net.append(norm)
            if config.hidden_act_fn is not None:
                act_f = getattr(nn, config.hidden_act_fn, nn.SiLU)()
                net.append(act_f)
            net.append(nn.Dropout(config.dropout))
            pre_dim = next_dim
        
        lin = nn.Linear(pre_dim, config.dims[-1], bias=config.bias)
        net.append(lin)

        if config.output_act_fn is not None :
            output_act_f = getattr(nn, config.output_act_fn, nn.SiLU)()
            net.append(output_act_f)
        self.net = nn.Sequential(*net)

    def forward(self, input : Tensor):
        return self.net(input)

class MixedEffectsModel(LightningModule):
    def __init__(self,
            random_effect_configs : List[RandomEffectConfig],
            num_id : int = 1,
            lr: float = 1e-2,
            eps=1e-3,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_id = num_id
        random_effects = []
        for conf in random_effect_configs :
            random_effects.append(RandomEffect(config=conf, num_id=num_id, eps=eps))
        self.random_effects = nn.ModuleList(random_effects)
        self.register_buffer("error_std_train", tensor(0.5, dtype=float))
        self.register_buffer("error_std_val", tensor(0.5, dtype=float))
        self._y_preds_train = tensor([], dtype=float, device=self.device)
        self._y_preds_val = tensor([], dtype=float, device=self.device)
        self._y_trues_train = tensor([], dtype=float, device=self.device)
        self._y_trues_val = tensor([], dtype=float, device=self.device)
        self.eps = eps

    @abc.abstractmethod
    def forward(self, init:Tensor, time:Tensor, iv:Tensor, id : Tensor) -> Tensor: ...

    def error_func(self, y_pred : Tensor, y_true : Tensor) -> tensor:
        return y_pred - y_true

    @torch.no_grad()
    def on_train_epoch_start(self) -> None: 
        self._y_trues_train : Tensor = tensor([], dtype=torch.float, device=self.device)
        self._y_preds_train : Tensor = tensor([], dtype=torch.float, device=self.device)
    
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        self.error_std_train = (self._y_trues_train - self._y_preds_train).std(correction=1) + self.eps
        dist = torch.distributions.Normal(tensor([0.], device=self.device), self.error_std_train)
        error = self.error_func(self._y_preds_train, self._y_trues_train)
        loss = (-2*dist.log_prob(error)).sum()
        for random_effect in self.random_effects :
            loss += random_effect.two_nll(torch.arange(1,self.num_id+1, dtype=torch.int, device=self.device))
        self.log('train_loss', loss, prog_bar=True) 

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None: 
        self._y_trues_val : Tensor = tensor([], dtype=torch.float, device=self.device)
        self._y_preds_val : Tensor = tensor([], dtype=torch.float, device=self.device)

    def on_validation_epoch_end(self) -> None:
        self.error_std_val = (self._y_trues_val - self._y_preds_val).std(correction=1) + self.eps
        dist = torch.distributions.Normal(tensor([0.], device=self.device), self.error_std_val)
        error = self.error_func(self._y_preds_val, self._y_trues_val)
        loss = (-2*dist.log_prob(error)).sum()
        for random_effect in self.random_effects :
            loss += random_effect.two_nll(torch.arange(1,self.num_id+1, dtype=torch.int, device=self.device))
        self.log('val_loss', loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        input = MixedEffectsTimeData(**batch)
        y_pred = self(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id = input.id)
        y_true = input.dv

        batch_size = batch['id'].size(0)
        mask = y_true.isnan().logical_not()
        y_true = torch.masked_select(y_true, mask)
        y_pred = torch.masked_select(y_pred, mask)
        error = self.error_func(y_pred, y_true)
        self._y_trues_train = torch.cat([self._y_trues_train, y_true])
        self._y_preds_train = torch.cat([self._y_preds_train, y_pred])

        dist = torch.distributions.Normal(tensor([0.], device=self.device), self.error_std_train)
        loss = (-2*dist.log_prob(error)).sum()*batch_size/self.num_id

        for random_effect in self.random_effects :
            loss += random_effect.two_nll(input.id)*batch_size/self.num_id
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input = MixedEffectsTimeData(**batch)
        y_pred = self(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id = input.id)
        y_true = input.dv

        batch_size = batch['id'].size(0)
        mask = y_true.isnan().logical_not()
        y_true = torch.masked_select(y_true, mask)
        y_pred = torch.masked_select(y_pred, mask)
        error = self.error_func(y_pred, y_true)
        self._y_trues_val = torch.cat([self._y_trues_val, y_true])
        self._y_preds_val = torch.cat([self._y_preds_val, y_pred])

        dist = torch.distributions.Normal(tensor([0.], device=self.device), self.error_std_val)
        loss = (-2*dist.log_prob(error)).sum()*batch_size/self.num_id 

        for random_effect in self.random_effects :
            loss += random_effect.two_nll(input.id)*batch_size/self.num_id
        
        return loss
    
    #ipred는 id를 0이상으로 해서 입력, pred는 id를 0으로 입력해서 알아서 처리
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tensor:
        input = MixedEffectsTimeData(**batch)
        y_pred = self.forward(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id = input.id)
        y_true = input.dv
        mask = y_true.isnan()
        return y_pred.masked_fill(mask, float('nan'))

    
    def on_test_start(self) -> None:
        for random_effect in self.random_effects:
            random_effect.simulation()

    def on_test_end(self) -> None:
        for random_effect in self.random_effects:
            random_effect.simulation(False)

    # simulation
    def test_step(self, batch, batch_idx):
        input = MixedEffectsTimeData(**batch)
        
        return self.forward(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id=input.id,
            simulation=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer