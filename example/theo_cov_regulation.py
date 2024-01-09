import math
from torch import Tensor
from torchpm.data import List, Tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from torchpm.module import RandomEffectConfig

correlations = [0.2, 0.4, 0.6, 0.8]
iv_column_names = ['BWT']
iv_column_names.append("BWT0")
iv_column_names.extend([f'BWT0_{int(cor*10)}' for cor in correlations])
iv_column_names.append('zero')

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
    # return theo_df

class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig] = [RandomEffectConfig(3)], 
            num_id: int = 12, 
            lr: float = 0.0003, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, *args, **kwargs)
        # self.v = BoundaryFixedEffect([32.8], [0.1], [100])
        self.v = lambda : 30.
        # self.k_a = BoundaryFixedEffect([1.22], [0.01], [5])
        self.k_a = lambda : 1.2
        # self.k_e = BoundaryFixedEffect([0.192], [0.01], [10])
        self.k_e = lambda : 0.2

        #bwt, fixed, random, 0.2,0.4,0.6,0.8,
        iv_dim = 7
        self.v_cov_regulation_para = nn.Parameter(tensor((0.5,)*iv_dim))
        self.k_a_cov_regulation_para = nn.Parameter(tensor((0.5,)*iv_dim))
        self.k_e_cov_regulation_para = nn.Parameter(tensor((0.5,)*iv_dim))

        ffn_dims = (iv_dim, iv_dim, 1)
        self.v_ffn = FFN(FFNConfig(ffn_dims,dropout=0.0, hidden_norm_layer=False))
        self.k_a_ffn = FFN(FFNConfig(ffn_dims))
        self.k_e_ffn = FFN(FFNConfig(ffn_dims))
    
    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        iv = iv.permute(1,2,0)
        v_cov = self.v_ffn(iv * self.v_cov_regulation_para.softmax(0)).squeeze(-1)
        # v_cov = self.v_ffn(iv).squeeze(-1)
        # k_a_cov = self.k_a_ffn(iv * self.k_a_cov_regulation_para.sigmoid()).squeeze(-1)
        # k_e_cov = self.k_e_ffn(iv * self.k_e_cov_regulation_para.sigmoid()).squeeze(-1)
        
        v = self.v()* v_cov.t().exp()*self.random_effects[0](id)[0].exp()
        k_a = self.k_a()  *self.random_effects[0](id)[1].exp()
        k_e = self.k_e()  *self.random_effects[0](id)[2].exp()

        # v = self.v()*self.random_effects[0](id)[0].exp()
        # k_a = self.k_a()  *self.random_effects[0](id)[1].exp()
        # k_e = self.k_e() *self.random_effects[0](id)[2].exp()

        return ((320 / v * k_a) / (k_a - k_e + 1e-6) * ((-k_e*time).exp() - (-k_a*time).exp())).t()
    
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        loss = output['loss']
        # num_batch = batch['id'].size(0)
        # loss += num_batch/self.num_id * self.v_cov_regulation_para.sigmoid().sum()
        # loss += num_batch/self.num_id * self.k_a_cov_regulation_para.sigmoid().sum()
        # loss += num_batch/self.num_id * self.k_e_cov_regulation_para.sigmoid().sum()
        output['loss'] = loss
        return output

if __name__ == "__main__": 
    f = open('example/regulation.txt', 'w')
    f.write('BWT,BWT0,BWT0_2,BWT0_4,BWT0_6,BWT0_8,zero,\n')
    for i in range(100):
        rng = np.random.default_rng(i)
        df = make_dataset(rng)
        
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['CONC'],
            init_column_names=['AMT'],
            iv_column_names=iv_column_names,
        )
        model = TheoModel(lr=1e-4)
        datamodule = MixedEffectsTimeDataModule(
            dataset_config=dataset_config, 
            batch_size=12,
            train_data=df
        )
        # early_stop = EarlyStopping(monitor='train_loss', patience=10, min_delta=1)
        trainer = Trainer(max_epochs=30000, check_val_every_n_epoch=100)
        trainer.fit(model, datamodule=datamodule)
        for v in model.v_cov_regulation_para.detach().numpy().tolist():
            f.write(f'{v},')
        # f.write(str(model.k_a_cov_regulation_para.detach().sigmoid().numpy()))
        # f.write(str(model.k_e_cov_regulation_para.detach().sigmoid().numpy()))
        f.write('\n')
    f.close()


    