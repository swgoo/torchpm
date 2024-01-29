import datetime
from torch import Tensor, tensor
from torchpm.data import List, Tensor, tensor
from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers

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
        v_std: float = 1.,
        k_a_tv: float = 1.5,
        k_a_std: float = 1.,
        k_e_tv: float = 0.1,
        k_e_std: float = 1.,
        eps : float = 0.05,
        num_id: int = 12,
        time: List[float] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,4.,8.,12.,24.],
        amt: float = 320.,
        seed : int = 42):
    rng = np.random.default_rng(seed)
    time = np.array(time)
    dataset = []
    record_len = len(time)
    for i in range(num_id):
        v = 0
        while v < 0.1 :
            v_cov1 = rng.standard_normal([])*v_cov1_std
            v_cov2 = rng.standard_normal([])*v_cov2_std
            v_cov3 = rng.standard_normal([])*v_cov3_std
            v_eta = rng.standard_normal([])*v_std
            v = v_cov_fn(v_tv, v_cov1, v_cov2, v_cov3)*np.exp(v_eta)

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
            lr: float = 0.001, 
            eps=1e-3, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, eps, *args, **kwargs)
        # self.v = BoundaryFixedEffect([30.], [10.], [50.])
        self.v = lambda : 30.
        # self.k_a = BoundaryFixedEffect([1.5], [1.], [5])
        self.k_a = lambda : 1.5
        # self.k_e = BoundaryFixedEffect([0.1], [0.05], [0.25])
        self.k_e = lambda : 0.1
        self.iv_column_names = iv_column_names

        ffn_dims = (iv_dim, 6, 4, 4, 2, 2, 1)
        ffn_config = FFNConfig(ffn_dims, dropout=0.0, hidden_norm_layer=False, bias=False, hidden_act_fn="SiLU")
        self.v_ffn = FFN(config=ffn_config)
    
    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        
        random_effects = self.random_effects[0](id) # batch, feat.
        # v = self.v()*random_effects[:,0].exp().unsqueeze(-1) # batch, time 
        v_cov = self.v_ffn(iv).squeeze(-1) if iv.size(-1) != 0 else tensor(0., device=self.device)
        v = self.v()*v_cov.exp()*random_effects[:,0].exp().unsqueeze(-1) # :batch, time 

        k_a = self.k_a()*random_effects[:,1].exp().unsqueeze(-1) # batch
        
        k_e = self.k_e()*random_effects[:,2].exp().unsqueeze(-1) # batch

        pred = ((320 / v * k_a) / (k_a - k_e) * ((-k_e*time).exp() - (-k_a*time).exp())).unsqueeze(-1)
        return pred.log()
    
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("error", self.error_std_train)
        # self.log("v", self.v().squeeze())
        # self.log("k_a", self.k_a().squeeze())
        # self.log("k_e", self.k_e().squeeze())
        omega = self.random_effects[0].covariance_matrix()
        self.log("omega_v", omega[0,0])
        self.log("omega_k_a", omega[1,1])
        self.log("omega_k_e", omega[2,2])
    
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        for para in self.v_ffn.parameters():
            loss += 1e-2*para.abs().sum() * batch['id'].size(0)/self.num_id
        return loss


    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=100)
        return [optimizer], [lr_scheduler]
    
def main(
        dir:str,
        model_name:str,
        seed : int,
        num_id = 100,
        max_epochs = 3_000,
        iv_column_names = ['V_COV1'],
        lr = 1e-3,):
    random_effect_config = RandomEffectConfig(3, init_value=[[0.1,0.,0.],[0.,0.1,0.],[0.,0.,0.1]])
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
    model = TheoModel(
        random_effect_configs=[random_effect_config],
        iv_dim=len(iv_column_names), 
        iv_column_names = iv_column_names,
        lr=lr, 
        num_id=id_len)

    # early_stop = EarlyStopping(monitor='train_loss', patience=10, min_delta=1)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{dir}/seed_{seed}", name=f"{model_name}")
    trainer = Trainer(
        max_epochs=max_epochs, 
        check_val_every_n_epoch=100, 
        logger=[tb_logger], 
        deterministic=True, 
        accumulate_grad_batches=int(num_id/batch_size))
    trainer.fit(model, datamodule=datamodule)
    print(seed)

if __name__ == "__main__":
    lr = 5e-3
    dir = "theo_cov_0129"
    max_epochs = 15_000
    num_id= 50
    for seed in range(11,62):
        try:
            main(dir, 'cov_3_model', seed, iv_column_names=['V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_1_2_3_model', seed, iv_column_names=['V_COV1', 'V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_1_3_model', seed, iv_column_names=['V_COV1', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_2_3_model', seed, iv_column_names=['V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_1_2_model', seed, iv_column_names=['V_COV1', 'V_COV2',], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_1_model', seed, iv_column_names=['V_COV1'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'cov_2_model', seed, iv_column_names=['V_COV2'], max_epochs=max_epochs, lr=lr, num_id=num_id)
            main(dir, 'base_model', seed, iv_column_names=[], max_epochs=max_epochs, lr=lr)
        except:
            continue