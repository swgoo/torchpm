import abc
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import torchode
from lightning import LightningModule
from torch import nn, Tensor, tensor
import torch
from math import prod

from .data import Dict, Tensor
from .data import *

class ODESolver(LightningModule):
    # Ordinary Differential Equation Solver
    @torch.no_grad()
    def __init__(
        self, 
        atol=1e-6, 
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
        
        for td, id in zip(time.size(), init.size()[:2]):
            assert td == id
        for v in kwargs.values():
            for td, vd in zip(time.size(), v.size()[:2]):
                assert td == vd

        ys = []
        time = time.T #[batch, time] -> [time, batch]
        init = init.permute([1,0,2]) # [batch, time, dim] -> [time, batch, dim]

        pre_times = time[0]
        pre_ys = init[0]
        ys.append(pre_ys)
        for i, (cur_times, cur_init_values) in enumerate(zip(time[1:], init[1:])) :
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
        return torch.stack(ys).permute([1,0,2]) # [Time, batch, dim] -> [batch,Time,dim]

class BoundaryFixedEffect(LightningModule) :
    def __init__(
            self,
            init_values: Tuple[float, ...] | float, 
            lower_boundary : Tuple[float, ...] | float | None,
            upper_boundary : Tuple[float, ...] | float | None):
        super().__init__()

        self.register_buffer('iv', tensor(init_values, dtype=float))
        assert self.iv.dim() == 0 | self.iv.dim() == 1

        if lower_boundary is None :    
            self.register_buffer('lb', 1.e-6 * torch.ones_like(self.iv))
        else :
            self.register_buffer('lb', tensor(lower_boundary, dtype=float))
            assert self.iv.size() == self.lb.size()
        
        if upper_boundary is None :
            self.register_buffer('ub', 1.e6 * torch.ones_like(self.iv))
        else :
            self.register_buffer('ub', tensor(upper_boundary, dtype=float))
            assert self.iv.size() == self.ub.size()
        
        self.register_buffer('alpha', 0.1 - torch.log((self.iv - self.lb)/(self.ub - self.lb)/(1 - (self.iv - self.lb)/(self.ub - self.lb))))
        self.register_parameter("var", nn.Parameter(0.1 * torch.ones_like(self.iv)))

    def forward(self) :
        return torch.exp(self.var - self.alpha)/(torch.exp(self.var - self.alpha) + 1)*(self.ub - self.lb) + self.lb


class RandomEffect(LightningModule):
    def __init__(
            self,
            num_id : int,
            dim: int, 
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_id = num_id
        self.dim = dim
        self.random_effects_emb = nn.Embedding(num_id + 1, dim, padding_idx = 0)
        nn.init.zeros_(self.random_effects_emb.weight)
    
    def forward(self, id: Tensor, simulation: bool):
        if simulation :
            loc = torch.zeros(self.dim, dtype=float, device=id.device)
            random_effects = torch.distributions.Normal(loc, self.standard_deviation()).sample(id.size()[:-1])
        else :
            mask = (id != 0).unsqueeze(-1)
            random_effects = self.random_effects_emb(id) * mask
        return random_effects
    
    def regularize(self, id : Tensor) -> Tensor:
        rv : Tensor = self.random_effects_emb(id)
        rv = rv.unsqueeze(-2)
        loss : Tensor =  rv @ (self.standard_deviation().diag().inverse()) @ rv.permute(0,2,1)
        return loss.squeeze().sum()
    
    def standard_deviation(self, eps : float = 1e-6) -> Tensor :
        return self.random_effects_emb.weight[1:].t().std(-1) + eps * torch.ones(self.dim, dtype=float, device=self.device)
    
@dataclass
class MaskedFFNConfig:
    # Feed Forward Net Configuration
    input_dim : int
    output_mask : Tuple[float,...] | Tensor
    hidden_state_dims : Tuple[int,...] = (1, 1)
    hidden_norm_layer : bool = True
    hidden_act_fn : str | None = 'SiLU'
    output_act_fn : str | None = None
    dropout : float = 0.2

    @torch.no_grad()
    def __post_init__(self):
        if not isinstance(self.output_mask, Tensor) :
            self.output_mask = tensor(self.output_mask, dtype=float)

class MaskedFFN(LightningModule):   
    # Feed Forward Net
    def __init__(
            self,
            config : MaskedFFNConfig,
            *args, 
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.register_buffer('output_mask', config.output_mask)
        
        net = []
        pre_dim = config.input_dim
        for next_dim in config.hidden_state_dims:
            lin = nn.Linear(pre_dim, next_dim)
            net.append(lin)
            if config.hidden_norm_layer :
                norm = nn.LayerNorm(next_dim)
                net.append(norm)
            if config.hidden_act_fn is not None:
                act_f = getattr(nn, config.hidden_act_fn, nn.SiLU)()
                net.append(act_f)
            net.append(nn.Dropout(config.dropout))
            pre_dim = next_dim
        
        output_layer = nn.Linear(pre_dim, prod(config.output_mask.size()))
        net.append(output_layer)
        if config.output_act_fn is not None :
            output_act_f = getattr(nn, config.output_act_fn, nn.SiLU)()
            net.append(output_act_f)
        self.net = nn.Sequential(*net)

    def forward(self, input : Tensor) -> Tensor :
        output : Tensor = self.net(input)
        output = output.reshape(*output.size()[:-1], *self.output_mask.size())
        return output * self.output_mask

class MixedEffectsModel(LightningModule):
    def __init__(self,
            num_id : int = 1,
            random_effects_dim : int = (1),
            loss_fn : str = "MSELoss",
            lr: float = 3e-4,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.loss_fn = getattr(nn, loss_fn)()
        self.random_effect = RandomEffect(num_id=num_id, dim= random_effects_dim)
        self.register_buffer("_loss_regulization", tensor(1, dtype=float))
        self.register_buffer("_total_residuals_train", tensor([], dtype=torch.float))

    @abc.abstractmethod
    def forward(self, init:Tensor, time:Tensor, iv:Tensor, id : Tensor, simulation: bool = False) -> Tensor: ...
    
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        residual = outputs['residual']
        self._total_residuals_train = torch.cat([self._total_residuals_train, residual.unsqueeze(0)], dim = 0)

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        self._total_residuals_train : Tensor = tensor([], dtype=torch.float, device=self.device)
        
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        self._loss_regulization = self._total_residuals_train.mean(0) + 1e-5

    def training_step(self, batch, batch_idx):
        input = MixedEffectsTimeData(**batch)
        y_pred = self(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id = input.id)
        y_true = input.dv

        mask = y_true.isnan().logical_not()
        y_true = torch.masked_select(y_true, mask)
        y_pred = torch.masked_select(y_pred, mask)
        residual = self.loss_fn(y_pred, y_true)

        loss = residual/self._loss_regulization
        
        loss += self.random_effect.regularize(input.id)
        self.log('train_loss', loss, prog_bar=True)
        
        return {"loss": loss, "residual": residual}
    
    # random_effect로 인한 과적합 확인 목적
    def validation_step(self, batch, batch_idx):
        input = MixedEffectsTimeData(**batch)
        id = torch.zeros_like(input.id, device=self.device)
        y_pred = self.forward(
            init = input.init,
            time = input.time,
            iv = input.iv,
            id= id)
        
        y_true = input.dv
        mask = y_true.isnan().logical_not()
        y_true = torch.masked_select(y_true, mask)
        y_pred = torch.masked_select(y_pred, mask)

        loss = self.loss_fn(y_pred, y_true)
        self.log('valid_loss', loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tensor:
        batch : MixedEffectsTimeData = MixedEffectsTimeData(*batch)
        id = torch.zeros_like(batch.id, device=self.device)
        return self.forward(
            init = batch.init,
            time = batch.time,
            iv = batch.iv,
            id= id)
    
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
        return [optimizer], []