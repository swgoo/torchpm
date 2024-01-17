import math
from torch import Tensor
from torchpm.data import List, Tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np
import pathlib

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import loggers as pl_loggers

from torchpm.module import RandomEffectConfig

from lightning.pytorch import seed_everything
import lightning.pytorch as pl

def make_dataset_and_iv_col_names(correlation: float, random_seed : int, dataset_repeat=10, num_rand_column: int = 0, add_cor_column = True):
    df = pd.read_csv('example/THEO.csv', na_values='.')
    ids = df['ID'].unique()
    rng = np.random.default_rng(random_seed)
    iv_column_names = []
    for i in range(num_rand_column):
        iv_column_names.append(f"rand{i}")
    if add_cor_column :
        iv_column_names.append(f"BWT{correlation:.2f}")
    else :
        iv_column_names.append('BWT')
    
    bwts = []
    df_by_id = df.groupby("ID", sort=True)
    for _, d in df_by_id:
        bwts.append(d['BWT'].iloc[0])
    ln_bwt = np.log(df['BWT'])
    ln_bwt_mean = np.log(bwts).mean()
    ln_bwt_std = np.log(bwts).std(ddof=1)
    df['BWT'] = (ln_bwt - ln_bwt_mean) / ln_bwt_std

    df_list = []
    for i in range(dataset_repeat) :
        repeated_df = df.copy(deep=True)
        repeated_df['ID'] = repeated_df['ID'] + i*len(ids)
        df_list.append(repeated_df)
    df = pd.concat(df_list)
            
    df_list = []
    df_by_id = df.groupby("ID", sort=True)
    for _, d in df_by_id:
        for i in range(num_rand_column) :
            d[f'rand{i}'] = rng.standard_normal(1).repeat(len(d))
        if add_cor_column :
            iv_name = f"BWT{correlation:.2f}"
            d[iv_name] = d['BWT'] * correlation + rng.standard_normal(1) * math.sqrt(1-correlation**2)
        df_list.append(d)
    df = pd.concat(df_list)
    return df , iv_column_names

class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig], 
            iv_dim,
            num_id: int = 1, lr: float = 0.01, 
            weight_decay: float = 0., 
            eps=1e-6, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, weight_decay, eps, *args, **kwargs)
        self.v = BoundaryFixedEffect([30.], [20], [100])
        # self.v = lambda : 30.7611
        self.k_a = BoundaryFixedEffect([1.5], [0.01], [5])
        # self.k_a = lambda : 1.4786
        self.k_e = BoundaryFixedEffect([0.1], [0.01], [2])
        # self.k_e = lambda : 0.0991

        ffn_dims = (iv_dim, iv_dim, 1)
        ffn_config = FFNConfig(ffn_dims, dropout=0.0, hidden_norm_layer=False, bias=False, output_norm=False, hidden_act_fn="SiLU")
        self.v_ffn = FFN(config=ffn_config)

    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        v_cov = self.v_ffn(iv).squeeze(-1) # batch, time, 1 -> batch, time
        random_effects = self.random_effects[0](id) # batch, feat.
        
        v = self.v()*v_cov.exp()*random_effects[:,0].exp().unsqueeze(-1) # batch, time 
        # v = self.v()*random_effects[:,0].exp().unsqueeze(-1) # batch, time 
        k_a = self.k_a()*random_effects[:,1].exp().unsqueeze(-1) # batch
        k_e = self.k_e()*random_effects[:,2].exp().unsqueeze(-1) # batch

        return ((320 / v * k_a) / (k_a - k_e) * ((-k_e*time).exp() - (-k_a*time).exp())).unsqueeze(-1)
    
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        # batch_size = batch['id'].size(0)
        # iv = batch['iv'].permute(1,2,0)
        # loss += 0.1*(self.v_cov_regulation_para.sigmoid()*(iv**2)/self.v_cov_regulation_para.nelement()).sum()*batch_size/self.num_id
        # loss += (self.v_cov_regulation_para.sigmoid()).sum()*batch_size/self.num_id
        # for para in self.v_ffn.parameters() :
            # loss += 1/2*(para**2).sum()*batch_size/self.num_id
        # output['loss'] = loss
        # self.log('train_ofv', loss, prog_bar=True)
        return output

def main(
        correlation, 
        prefix:str, 
        repeat : int = 10, 
        dataset_repeat = 10, 
        num_rand_column = 1, 
        add_cor_column = True,
        max_epochs = 3_000):
    file_name = f'{prefix}_{correlation}'
    random_effect_config = RandomEffectConfig(3)

    for i in range(repeat):
        df, iv_column_names = make_dataset_and_iv_col_names(correlation=correlation, random_seed=i, dataset_repeat=dataset_repeat, num_rand_column=num_rand_column, add_cor_column=add_cor_column)
        
        if not pathlib.Path(f'example/{file_name}.txt').exists() :
            f = open(f'example/{file_name}.txt', 'w')
            f.write(f"{','.join(iv_column_names)}\n")
            f.close()
        
        id_len = len(df['ID'].unique())
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['CONC'],
            init_column_names=['AMT'],
            iv_column_names=iv_column_names,
        )
        datamodule = MixedEffectsTimeDataModule(
            dataset_config=dataset_config, 
            batch_size=id_len,
            train_data=df
        )
        seed_everything(i, workers=True)
        model = TheoModel(
            random_effect_configs=[random_effect_config],
            iv_dim=len(iv_column_names), 
            lr=5e-3, 
            num_id=id_len)

        # early_stop = EarlyStopping(monitor='train_loss', patience=10, min_delta=1)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{file_name}/{i}")
        trainer = Trainer(
            max_epochs=max_epochs, 
            check_val_every_n_epoch=100, 
            logger=[tb_logger], 
            deterministic=True, 
            accumulate_grad_batches=1)
        trainer.fit(model, datamodule=datamodule)
        f = open(f'example/{file_name}.txt', 'a')
        for batch in datamodule.train_dataloader() : 
            v_ffn = model.v_ffn
            iv = batch['iv']
            v_ffn.zero_grad()
            iv.requires_grad_(True)
            v_cov = v_ffn(iv, explain= True, rule="alpha1beta0")
            v_cov_sum = v_cov.sum()
            v_cov_sum.backward()
            explain = iv.grad
            explain = explain.sum([1,0]).numpy()
            explain = [str(v) for v in explain]
            f.write(f"{','.join(explain)}\n")
            f.close()

        # for v in model.v_cov_regulation_para.detach().numpy().tolist():
        # f.write(str(model.k_a_cov_regulation_para.detach().sigmoid().numpy()))
        # f.write(str(model.k_e_cov_regulation_para.detach().sigmoid().numpy()))

if __name__ == "__main__":
    # for i in [0.0, 0.5, 1.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
    # for i in [1.0, 0.0, 0.5]:
    # for i in [1.0, 0.5, 0.0]:
    for i in [1.0,0.0,0.5,0.7,0.3]:
        main(
            i,
            prefix='s', 
            repeat=100, 
            dataset_repeat=1, 
            num_rand_column=1, 
            add_cor_column=True,
            max_epochs=5_000)