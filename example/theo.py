from typing import Dict
from torch import Tensor, tensor
from torchpm.data import Dict, List, Tensor, tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers

from torchpm.module import List, RandomEffectConfig

from lightning.pytorch import seed_everything
import lightning.pytorch as pl


class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig], 
            num_id: int = 1, 
            lr: float = 0.001, 
            eps=1e-5, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, eps, *args, **kwargs)
        self.v = BoundaryFixedEffect([30.], [10.], [50.])
        self.k_a = BoundaryFixedEffect([1.5], [1.], [10])
        self.k_e = BoundaryFixedEffect([0.1], [0.05], [5.])
    
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("v", self.v().squeeze())
        self.log("k_a", self.k_a().squeeze())
        self.log("k_e", self.k_e().squeeze())
        self.log("error", self.error_std_train)
        omega = self.random_effects[0].covariance_matrix()
        self.log("omega_v", omega[0,0])
        self.log("omega_k_a", omega[1,1])
        self.log("omega_k_e", omega[2,2])
        return 


    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        
        random_effects = self.random_effects[0](id) # batch, feat.

        v = self.v()*random_effects[:,0].exp().unsqueeze(-1) # batch, 1

        k_a = self.k_a()*random_effects[:,1].exp().unsqueeze(-1) # batch, 1
        
        k_e = self.k_e()*random_effects[:,2].exp().unsqueeze(-1) # batch, 1

        return ((320 / v * k_a) / (k_a - k_e) * ((-k_e*time).exp() - (-k_a*time).exp())).unsqueeze(-1)

class TheoODESolver(ODESolver):
    def ode(self, t: Tensor, y: Tensor, kwargs: Dict[str, Tensor]):
        k_a = kwargs['k_a']
        k_e = kwargs['k_e']
        return torch.stack([-y[:,0]*k_a, y[:,0]*k_a - y[:,1]*k_e]).T
    
class TheoODEModel(TheoModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig], 
            num_id: int = 1, 
            lr: float = 0.001, 
            eps=1e-3, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, eps, *args, **kwargs)
        self.ode_solver = TheoODESolver(atol=1e-3, rtol=1e-2)

    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        
        random_effects = self.random_effects[0](id) # batch, feat.

        v = self.v()*random_effects[:,0].exp().unsqueeze(-1).repeat(1,time.size(1)) # batch, time, 1

        k_a = self.k_a()*random_effects[:,1].exp().unsqueeze(-1).repeat(1,time.size(1)) # batch, time, 1
        
        k_e = self.k_e()*random_effects[:,2].exp().unsqueeze(-1).repeat(1,time.size(1)) # batch, time, 1

        ys = self.ode_solver.forward(time=time, init=init, kwargs={'v':v, "k_a":k_a, 'k_e':k_e})

        return ys / v.unsqueeze(-1)
    
    
def main(
        dir:str,
        model_name:str,
        seed : int,
        max_epochs = 3_000,
        batch_size = 12,
        ode_mode = False,
        lr = 5e-3,):
    random_effect_config = RandomEffectConfig(3, init_value=[[0.1,0.,0.],[0.,0.1,0.],[0.,0.,0.1]])

    df = pd.read_csv('example/THEO.csv', na_values='.')
    
    id_len = len(df['ID'].unique())
    
    seed_everything(seed, workers=True)
    if ode_mode :
        df['central_init'] = len(df) * [0.]
        df['gut_dv'] = len(df) * [float('nan')]
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['gut_dv','CONC'],
            init_column_names=['AMT', "central_init"],
            iv_column_names=[],
        )
        
        model = TheoODEModel(
            random_effect_configs=[random_effect_config],
            lr=lr,
            num_id=id_len
        )
    else:
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['CONC'],
            init_column_names=['AMT'],
            iv_column_names=[],
        )
        model = TheoModel(
            random_effect_configs=[random_effect_config],
            lr=lr, 
            num_id=id_len)
    datamodule = MixedEffectsTimeDataModule(
        dataset_config=dataset_config, 
        batch_size=batch_size,
        train_data=df
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{dir}", name=f"{model_name}", version=f"seed_{seed}")
    trainer = Trainer(
        max_epochs=max_epochs, 
        check_val_every_n_epoch=100,
        logger=[tb_logger],
        deterministic=True,
        accumulate_grad_batches=int(id_len/batch_size))
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    batch_size = 12
    dir = 'theo_dataset'
    main(
        dir=dir, 
        model_name='ode_base_model', 
        seed=42, 
        max_epochs=3_000, 
        lr=1e-3, 
        batch_size=batch_size,
        ode_mode=True)
    main(
        dir=dir, 
        model_name='base_model', 
        seed=42, 
        max_epochs=3_000, 
        lr=1e-3, 
        batch_size=batch_size,
        ode_mode=False)
    