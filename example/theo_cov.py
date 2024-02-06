
import datetime
from typing import Any, Literal, Optional, Sequence
from torch import Tensor, tensor
from torchpm.data import List, Tensor, tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import LightningModule, Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import BasePredictionWriter
from torchpm.module import RandomEffectConfig

from lightning.pytorch import seed_everything

def cov_fn(typical_value, cov1, cov2, cov3):
    return typical_value*np.exp(cov1)+typical_value*cov2+cov3
    
def pred_gut_one_compartment_model(amt: np.array, v: np.array, k_a: np.array, k_e: np.array, time: np.array) -> np.array:
    return (amt / v * k_a) / (k_a - k_e) * (np.exp(-k_e*time) - np.exp(-k_a*time))

def make_dataset(
        v_cov_fn: callable = cov_fn,
        v_cov1_std: float = 0.3,
        v_cov2_std: float = 0.3,
        v_cov3_std: float = 0.3,
        v_tv : float = 30.,
        # v_std: float = 1.,
        k_a_tv: float = 1.5,
        k_a_std: float = 1.,
        k_e_tv: float = 0.1,
        k_e_std: float = 1.,
        eps : float = 0.05,
        num_id: int = 12,
        time: List[float] = [0.1,0.25,0.75,1.,2.,4.,8.,12.,24.],
        amt: float = 320.,
        seed : int = 42):
    rng = np.random.default_rng(seed)
    time = np.array(time)
    dataset = []
    record_len = len(time)
    for i in range(num_id):
        v = 0
        while v < 0.1:
            v_cov1 = rng.standard_normal([])*v_cov1_std
            v_cov2 = rng.standard_normal([])*v_cov2_std
            v_cov3 = rng.standard_normal([])*v_cov3_std
            v = v_cov_fn(v_tv, v_cov1, v_cov2, v_cov3)

        k_a_eta = rng.standard_normal([])*k_a_std
        k_a = k_a_tv * np.exp(k_a_eta)

        k_e_eta = rng.standard_normal([])*k_e_std
        k_e = k_e_tv * np.exp(k_e_eta)
        dv_col = pred_gut_one_compartment_model(amt,v,k_a,k_e,time)
        dv_col = np.log(dv_col) * np.exp(rng.standard_normal(np.size(dv_col))*eps)
        id_col = np.array([i+1]).repeat(record_len)
        v_cov_cols = np.array([[v_cov1/v_cov1_std], [v_cov2/v_cov2_std], [v_cov3/v_cov3_std]]).repeat(record_len, -1)
        amt_col = np.array(amt).repeat(record_len)
        data = np.stack([id_col, amt_col, time, dv_col, *v_cov_cols])
        dataset.append(data)
    dataset = np.concatenate(dataset, 1)
    df =  pd.DataFrame(dataset.T)
    df.columns = ["ID","AMT","TIME","CONC","V_COV1","V_COV2","V_COV3"]
    return df
    

class TheoModel(MixedEffectsModel) :
    def __init__(
            self, 
            random_effect_configs: List[RandomEffectConfig], 
            iv_dim,
            iv_column_names,
            num_id: int = 1,
            total_steps: int = 15_000, 
            lr: float = 0.001, 
            eps=1e-3, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, eps, *args, **kwargs)
        self.v = BoundaryFixedEffect([30.], [6.], [180.])
        # self.v = lambda : 30.
        self.k_a = BoundaryFixedEffect([1.5], [0.3], [9])
        # self.k_a = lambda : 1.5
        self.k_e = BoundaryFixedEffect([0.1], [0.02], [0.6])
        # self.k_e = lambda : 0.1
        self.iv_column_names = iv_column_names
        self.iv_dim = iv_dim

        ffn_dims = (iv_dim, 6, 6, 6, 1)
        ffn_config = FFNConfig(ffn_dims, dropout=0.0, hidden_norm_layer=False, bias=True, hidden_act_fn="ReLU")
        self.v_ffn = FFN(config=ffn_config)
    
    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        time= time.unsqueeze(-1)
        random_effects = self.random_effects[0](id) # batch, feat.        
        k_a = self.k_a()*random_effects[:,0:1,None].exp() # batch, 1, 1
        k_e = self.k_e()*random_effects[:,1:2,None].exp() # batch, 1, 1
        if iv.size(-1) != 0 :
            v_cov = self.v_ffn(iv)
            v = self.v()*v_cov.exp()*random_effects[:,2:3,None].exp() # :batch, time
        # elif iv.size(-1) == 3 :
        #     v_cov = self.v_ffn(iv)
        #     v = self.v()*v_cov.exp()
        else :
            v = self.v()*random_effects[:,2:3,None].exp() # :batch, 1, 1 

        pred = (320 / v * k_a) / ((k_a - k_e).abs() + 1e-6) * ((-k_e*time).exp() - (-k_a*time).exp() + 1e-6)
        return pred.log().nan_to_num(nan=0.0, posinf=1000, neginf=0)
    
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("error", self.error_std_train)
        self.log("v", self.v().squeeze())
        self.log("k_a", self.k_a().squeeze())
        self.log("k_e", self.k_e().squeeze())
        omega = self.random_effects[0].covariance_matrix()
        self.log("omega_k_a", omega[0,0])
        self.log("omega_k_e", omega[1,1])
        # if self.iv_dim != 3 :
        self.log("omega_v", omega[2,2])

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=5e-3, 
            total_steps=self.hparams['total_steps'], 
            anneal_strategy='linear')
        return [optimizer], [lr_scheduler]
    
class PredictWriter(BasePredictionWriter):
    def __init__(
            self,
            output_path: Path, 
            write_interval: Literal['batch', 'epoch', 'batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)
        self.output_path = output_path

    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices: Sequence[int] | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None: 
        pred = prediction.cpu().numpy()
        np.save(self.output_path, pred)


    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Tensor, batch_indices: Sequence[Any]) -> None: ...
        

def main(
        dir:str,
        model_name:str,
        seed : int,
        num_id = 100,
        max_epochs = 3_000,
        iv_column_names = ['V_COV1'],
        lr = 1e-3,
        train = True):
    iv_len = len(iv_column_names)
    # if iv_len == 3 :
    #     random_effect_config = RandomEffectConfig(2, init_value=[[0.9,0.],[0.,0.9]])
    # else :
    random_effect_config = RandomEffectConfig(3, init_value=[[0.9,0.,0.],[0.,0.9,0.],[0.,0.,0.9]])
    batch_size = num_id
    df = make_dataset(num_id=num_id, seed=seed)
    
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
    seed_everything(seed, workers=True)
    

    # early_stop = EarlyStopping(monitor='train_loss', patience=10, min_delta=1)
    
    
    if train :
        model = TheoModel(
            random_effect_configs=[random_effect_config],
            iv_dim=len(iv_column_names), 
            iv_column_names = iv_column_names,
            total_steps=max_epochs,
            lr=lr, 
            num_id=id_len) 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{dir}/seed_{seed}", name=f"{model_name}")
        trainer = Trainer(
            max_epochs=max_epochs, 
            check_val_every_n_epoch=100, 
            logger=[tb_logger], 
            deterministic=True, 
            accumulate_grad_batches=int(num_id/batch_size))
        trainer.fit(model, datamodule=datamodule)
    else :
        model=TheoModel.load_from_checkpoint(f"lightning_logs/{dir}/seed_{seed}/{model_name}/version_0/checkpoints/epoch=19999-step=20000.ckpt")
        writer = PredictWriter(output_path=Path(f"lightning_logs/{dir}/seed_{seed}/{model_name}/pred.npy"))
        trainer = Trainer(
            max_epochs=1, 
            check_val_every_n_epoch=1, 
            logger=[],
            callbacks=[writer],
            deterministic=True, 
            accumulate_grad_batches=int(num_id/batch_size))
        trainer.predict(model, datamodule=datamodule)

train = True
if __name__ == "__main__":
    lr = 3e-3
    dir = "theo_cov_0206"
    max_epochs = 20_000
    num_id= 50
    for seed in range(50):
        try:
            main(dir, 'cov_1_2_3_model', seed=seed, iv_column_names=['V_COV1', 'V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, train= train)
            main(dir, 'cov_1_3_model', seed=seed, iv_column_names=['V_COV1', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
            main(dir, 'cov_1_2_model', seed=seed, iv_column_names=['V_COV1', 'V_COV2',], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
            main(dir, 'cov_2_3_model', seed=seed, iv_column_names=['V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
            main(dir, 'base_model', seed=seed, iv_column_names=[], max_epochs=max_epochs, lr=lr, num_id=num_id, train=train)
            main(dir, 'cov_1_model', seed=seed, iv_column_names=['V_COV1'], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
            main(dir, 'cov_2_model', seed=seed, iv_column_names=['V_COV2'], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
            main(dir, 'cov_3_model', seed=seed, iv_column_names=['V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, train = train)
        except:
            continue