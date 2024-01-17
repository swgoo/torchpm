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

def make_dataset(random_seed : int, correlations, add_random_column = True):
    theo_df = pd.read_csv('example/THEO.csv', na_values='.')

    df_by_id = theo_df.groupby("ID", sort=True)
    bwts = []
    for _, d in df_by_id:
        bwts.append(d['BWT'].iloc[0])

    ln_bwt = np.log(theo_df['BWT'])
    ln_bwt_mean = np.log(bwts).mean()
    ln_bwt_std = np.log(bwts).std(ddof=1)
    theo_df['BWT'] = (ln_bwt - ln_bwt_mean) / ln_bwt_std

    if add_random_column :
        rng = np.random.default_rng(random_seed)
        df_list = []
        df_by_id = theo_df.groupby("ID", sort=True)
        for _, d in df_by_id : 
            for i in range(0,360,12):
                d_copy = d.copy()
                d_copy['ID'] = d['ID']+i
                df_list.append(d_copy)
        df = pd.concat(df_list)
                
        df_list = []
        random_variables = rng.standard_normal(size=(360,len(correlations)+1))
        df_by_id = df.groupby("ID", sort=True)

        for (_, d), ran_var in zip(df_by_id, random_variables):
            d['rand0'] = ran_var[0].repeat(len(d))
            # d['rand1'] = ran_var[1].repeat(len(d))
            for cor, r_v in zip(correlations, ran_var[1:]) :
                iv_name = f"BWT{cor:.2f}"
                d[iv_name] = d['BWT'] * cor + r_v * math.sqrt(1-cor**2)
            df_list.append(d)
        df = pd.concat(df_list)
        # df['one'] = [1.]*len(df)

        return df
    else :
        return theo_df

class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            iv_dim,
            random_effect_configs: List[RandomEffectConfig] = [RandomEffectConfig(3)], 
            num_id: int = 360, 
            lr: float = 1e-4,
            weight_decay: float = 1e-2,
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, weight_decay, *args, **kwargs)
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
        iv = iv.permute(1,2,0)
        v_cov = self.v_ffn(iv).squeeze(-1)
        
        v = self.v()*v_cov.t().exp()*self.random_effects[0](id)[0].exp()
        k_a = self.k_a()*self.random_effects[0](id)[1].exp()
        k_e = self.k_e()*self.random_effects[0](id)[2].exp()

        return ((320 / v * k_a) / (k_a - k_e) * ((-k_e*time).exp() - (-k_a*time).exp())).t()
    
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        loss = output['loss']
        # batch_size = batch['id'].size(0)
        # iv = batch['iv'].permute(1,2,0)
        # loss += 0.1*(self.v_cov_regulation_para.sigmoid()*(iv**2)/self.v_cov_regulation_para.nelement()).sum()*batch_size/self.num_id
        # loss += (self.v_cov_regulation_para.sigmoid()).sum()*batch_size/self.num_id
        # for para in self.v_ffn.parameters() :
            # loss += 1/2*(para**2).sum()*batch_size/self.num_id
        # output['loss'] = loss
        # self.log('train_ofv', loss, prog_bar=True)
        return output
    
    def configure_optimizers(self):
        optimizer =  torch.optim.Adamax(self.parameters(),self.hparams.lr)
        lr_schduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
        return [optimizer], [lr_schduler]

def run(correlation):
    file_name = f'nonfix_regulation_{correlation}'
    correlations = [correlation]
    # iv_column_names = ['BWT']
    iv_column_names = []
    iv_column_names.append("rand0")
    # iv_column_names.append("rand1")
    iv_column_names.extend([f'BWT{cor:.2f}' for cor in correlations])
    # iv_column_names.append('one')
    if not pathlib.Path(f'example/{file_name}.txt').exists() :
        f = open(f'example/{file_name}.txt', 'w')
        f.write(f"{','.join(iv_column_names)},\n")
        f.close()

    for i in range(30):
        df = make_dataset(i, correlations, add_random_column=True)
        
        dataset_config = MixedEffectsTimeDatasetConfig(
            dv_column_names=['CONC'],
            init_column_names=['AMT'],
            iv_column_names=iv_column_names,
        )
        datamodule = MixedEffectsTimeDataModule(
            dataset_config=dataset_config, 
            batch_size=360,
            train_data=df
        )
        seed_everything(i, workers=True)
        model = TheoModel(iv_dim=len(iv_column_names), lr=1e-2, num_id=360)

        # early_stop = EarlyStopping(monitor='train_loss', patience=10, min_delta=1)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{file_name}/{i}")
        trainer = Trainer(max_epochs=8_000, check_val_every_n_epoch=500, logger=[tb_logger], deterministic=True)
        trainer.fit(model, datamodule=datamodule)
        f = open(f'example/{file_name}.txt', 'a')
        for batch in datamodule.train_dataloader() : 
            v_ffn = model.v_ffn
            iv = batch['iv'].permute(1,2,0)
            v_ffn.zero_grad()
            iv.requires_grad_(True)
            v_cov = v_ffn(iv, explain= True, rule="alpha1beta0")
            v_cov_sum = v_cov.sum()
            v_cov_sum.backward()
            explain = iv.grad
            for v in explain.detach().sum(1).sum(0).numpy().tolist():
                f.write(f'{v},')

        # for v in model.v_cov_regulation_para.detach().numpy().tolist():
        # f.write(str(model.k_a_cov_regulation_para.detach().sigmoid().numpy()))
        # f.write(str(model.k_e_cov_regulation_para.detach().sigmoid().numpy()))
        f.write('\n')
        f.close()

if __name__ == "__main__":
    # for i in [0.0, 0.5, 1.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
    # for i in [1.0, 0.0, 0.5]:
    # for i in [1.0, 0.5, 0.0]:
    for i in [0.0,0.5]:
        run(i)