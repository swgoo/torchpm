from dataclasses import dataclass
from lightning import LightningModule
import torch
from torch import tensor, Tensor, nn


@dataclass
class RandomEffectConfig:
    dim: int
    init_value: tuple[tuple[float, ...], ...] | None = None,
    covariance: bool = False

class RandomEffect(LightningModule):
    @torch.no_grad()
    def __init__(
            self,
            config: RandomEffectConfig,
            num_id: int,
            eps : float = 1e-3,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.random_variables = nn.Embedding(num_id + 1, self.config.dim, padding_idx = 0)
        self.register_buffer('_loc', torch.zeros(self.config.dim, dtype=torch.float32))
        self.eps = eps
        self._simulation = False

        if config.init_value is None :
            nn.init.normal_(self.random_variables.weight, std=0.01)
        else :
            init_covariance = tensor(config.init_value, dtype=torch.float, device=self.device)
            assert init_covariance.dim() == 2
            weight = torch.distributions.MultivariateNormal(self._loc, init_covariance).sample(torch.Size([num_id+1]))
            self.random_variables.weight *= 0
            self.random_variables.weight += weight
        
    def forward(self, id: Tensor):
        mask = (id != 0).unsqueeze(-1) # if id == 0 -> random_variable = 0
        if self._simulation :
            random_effects = torch.distributions.MultivariateNormal(self._loc, self.covariance_matrix()).sample(id.size()[:-1])
            random_effects = random_effects * mask
        else :
            random_effects = self.random_variables(id) * mask

            
        return random_effects # batch
    
    def two_nll(self, id : Tensor) -> Tensor:
        # negative log likelihood
        mask = (id != 0).unsqueeze(-1)
        random_variables : Tensor = self.random_variables(id) * mask
        if self.config.covariance :
            dist = torch.distributions.MultivariateNormal(self._loc, self.covariance_matrix())
        else :
            dist = torch.distributions.MultivariateNormal(self._loc, precision_matrix=(1/self.covariance_matrix().diag()).diag())
        return (-2*dist.log_prob(random_variables)).sum()
    
    def covariance_matrix(self) -> Tensor :
        if self.config.covariance :
            return (self.random_variables.weight[1:]).t().cov(correction=1).nan_to_num(0.) + self.eps*torch.ones(self.config.dim).diag()
        else :
            return (self.random_variables.weight[1:].t().var(dim=-1, correction=1).nan_to_num(0.) + self.eps).diag()
    
    def simulation(self, simulation = True):
        self._simulation =  simulation

class BoundaryFixedEffect(LightningModule) :
    def __init__(
            self,
            init_value: tuple[float, ...] | float, 
            lower_boundary: tuple[float, ...] | float | None = None,
            upper_boundary: tuple[float, ...] | float | None = None):
        super().__init__()

        self.register_buffer('iv', tensor(init_value, dtype=torch.float32))
        assert self.iv.dim() == 0 or self.iv.dim() == 1

        if lower_boundary is None :    
            self.register_buffer('lb', 1.e-6 * torch.ones_like(self.iv))
        else :
            self.register_buffer('lb', tensor(lower_boundary, dtype=torch.float32))
            assert self.iv.size() == self.lb.size()
        
        if upper_boundary is None :
            self.register_buffer('ub', 1.e6 * torch.ones_like(self.iv))
        else :
            self.register_buffer('ub', tensor(upper_boundary, dtype=torch.float32))
            assert self.iv.size() == self.ub.size()
        
        self.register_buffer('alpha', 0.1 - torch.log((self.iv - self.lb)/(self.ub - self.lb)/(1 - (self.iv - self.lb)/(self.ub - self.lb))))
        self.register_parameter("var", nn.Parameter(0.1 * torch.ones_like(self.iv)))

    def forward(self) :
        return torch.exp(self.var - self.alpha)/(torch.exp(self.var - self.alpha) + 1)*(self.ub - self.lb) + self.lb