import math
from torch import Tensor
from torchpm.data import List, Tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import Trainer

from torchpm.module import RandomEffectConfig

correlations = [i*0.2 for i in range(1,5)]
iv_column_names = ['BWT']
iv_column_names.append("BWT0")
iv_column_names.extend([f'BWT0_{cor}' for cor in correlations])

def make_dataset(rng : np.random.Generator):
    theo_df = pd.read_csv('example/THEO.csv', na_values='.')
    theo_df['BWT'] = (theo_df['BWT'] - theo_df['BWT'].mean()) / theo_df['BWT'].std()

    df_list = []
    df_by_id = theo_df.groupby("ID", sort=True)
    random_variables = rng.standard_normal(size=(12,5))

    for (_, d), ran_var in zip(df_by_id, random_variables):
        d['BWT0'] = ran_var[0].repeat(len(d))
        for cor, r_v in zip(correlations, ran_var) :
            iv_name = f"BWT0_{int(cor*10)}"
            d[iv_name] = d['BWT'] * cor + r_v * math.sqrt(1-cor**2)
        df_list.append(d)
    df = pd.concat(df_list)
    df['zero'] = [0.]*len(df)

    return df

class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig] = [RandomEffectConfig(3)], 
            num_id: int = 12, 
            lr: float = 0.0003, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, *args, **kwargs)
        self.v = BoundaryFixedEffect([10], [0.1], [100])
        self.k_a = BoundaryFixedEffect([1], [0.01], [5])
        self.k_e = BoundaryFixedEffect([1], [0.01], [10])

        #bwt, fixed, random, 0.2,0.4,0.6,0.8,
        iv_dim = 7
        self.v_cov_regulation = BoundaryFixedEffect((0.5)*iv_dim, (0.)*iv_dim, (1.)*iv_dim)
        self.k_a_cov_regulation = BoundaryFixedEffect((0.5)*iv_dim, (0.)*iv_dim, (1.)*iv_dim)
        self.k_e_cov_regulation = BoundaryFixedEffect((0.5)*iv_dim, (0.)*iv_dim, (1.)*iv_dim)

        ffn_dims = (iv_dim, iv_dim, 1)
        self.v_ffn = FFN(FFNConfig(ffn_dims, output_act_fn="Sigmoid"))
        self.k_a_ffn = FFN(FFNConfig(ffn_dims, output_act_fn="Sigmoid"))
        self.k_e_ffn = FFN(FFNConfig(ffn_dims, output_act_fn="Sigmoid"))
    
    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor, simulation: bool = False) -> Tensor:
        iv = iv.permute(1,2,0)
        v_cov = self.v_ffn(iv * self.v_cov_regulation())
        k_a_cov = self.k_a_ffn(iv * self.k_a_cov_regulation())
        k_e_cov = self.k_e_ffn(iv * self.k_e_cov_regulation())
        
        v = self.v*self.random_effects[0](id, simulation = simulation)[0].exp() * v_cov.exp()
        k_a = self.k_a*self.random_effects[0](id, simulation = simulation)[1].exp() * k_a_cov.exp()
        k_e = self.k_e*self.random_effects[0](id, simulation = simulation)[2].exp() * k_e_cov.exp()

        return (320 / v * k_a) / (k_a - k_e) * ((-k_e*time).exp() - (-k_a*time).exp())
    
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        loss = output['loss']
        num_batch = batch['id'].size()
        loss += 3.84 * num_batch/self.num_id * self.v_cov_regulation().sum()
        loss += 3.84 * num_batch/self.num_id * self.k_a_cov_regulation().sum()
        loss += 3.84 * num_batch/self.num_id * self.k_e_cov_regulation().sum()
        output['loss'] = loss
        return loss

if __name__ == "__main__": 
    rng = np.random.default_rng(42)

    for i in range(100):
        df = make_dataset(rng)
        
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['CONC'],
            init_column_names=['AMT'],
            iv_column_names=iv_column_names,
        )

        datamodule = MixedEffectsTimeDataModule(
            dataset_config=dataset_config, 
            batch_size=12,
            train_data=df
        )


        trainer = Trainer()
        trainer.fit(TheoModel(), datamodule=datamodule)

    