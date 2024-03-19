
from typing import Any, Callable, Literal, Optional, Sequence
from torch import Tensor
from torchpm.data import List, Tensor
from torchpm import *
from torchpm.data import *
import pandas as pd
import numpy as np

from lightning import LightningModule, Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import BasePredictionWriter
from torchpm import RandomEffectConfig

from lightning.pytorch import seed_everything

import typer

app = typer.Typer()


def v_fn(typical_value: Tensor, cov: Tensor):
    return typical_value*torch.exp(cov[0])+typical_value*cov[1]+cov[2]


def pred_gut_one_compartment_model(amt: np.array, v: np.array, k_a: np.array, k_e: np.array, time: np.array) -> np.array:
    return (amt / v * k_a) / (k_a - k_e) * (np.exp(-k_e*time) - np.exp(-k_a*time))


@dataclass
class ParameterConfig:
    v_tv : float = 30.
    v_std: float = 0.5
    v_min: float = 3.
    v_max: float = 300.

    k_a_tv: float = 1.5
    k_a_std: float = 0.5
    k_a_min: float = 0.3
    k_a_max: float = 7.5

    k_e_tv: float = 0.1
    k_e_std: float = 0.5
    k_e_min: float = 0.02
    k_e_max: float = 0.5

    sigma : float = 0.03

    v_cov_std : float = 0.5


def generate_dataset(
        seed : int,
        num_id: int,
        parameter_config : ParameterConfig,
        v_cov_fn: Optional[Callable],
        time: List[float] = [0.25,0.5,0.75,1.,2.,4.,8.,12.,24.],
        amt: float = 320.,
    )-> pd.DataFrame: 
    rng = np.random.default_rng(seed)
    time = np.array(time)
    dataset = []
    record_len = len(time)
    for i in range(num_id):
        
        v = parameter_config.v_min - 1
        while v < parameter_config.v_min or v > parameter_config.v_max:
            if v_cov_fn is None :
                v = parameter_config.v_tv*np.exp(v_eta)
            else :
                v_cov = torch.tensor(rng.standard_normal([3])*parameter_config.v_cov_std, dtype=torch.float)
                v_eta = torch.tensor(rng.standard_normal([])*parameter_config.v_std, dtype=torch.float)
                v = v_cov_fn(tensor(parameter_config.v_tv, dtype=torch.float), v_cov)*torch.exp(v_eta)
                v = v.numpy()
        
        k_a = parameter_config.k_a_min - 1
        while k_a < parameter_config.k_a_min or k_a > parameter_config.k_a_max:
            k_a_eta = rng.standard_normal([])*parameter_config.k_a_std
            k_a = parameter_config.k_a_tv * np.exp(k_a_eta)

        k_e = parameter_config.k_e_min - 1
        while k_e < parameter_config.k_e_min or k_e > parameter_config.k_e_max:
            k_e_eta = rng.standard_normal([])*parameter_config.k_e_std
            k_e = parameter_config.k_e_tv * np.exp(k_e_eta)

        dv_col = pred_gut_one_compartment_model(amt,v,k_a,k_e,time)
        dv_col = np.log(dv_col * np.exp(rng.standard_normal(np.size(dv_col))*parameter_config.sigma))

        id_col = np.array([i+1]).repeat(record_len)
        v_cov_cols = (v_cov/parameter_config.v_cov_std).unsqueeze(-1).repeat(1,record_len).numpy()
        # v_cov_cols = np.array([[v_cov[0]/v_cov_std], [v_cov[1]/v_cov_std], [v_cov[2]/v_cov_std]]).repeat(record_len, -1)

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
            parameter_config: ParameterConfig,
            iv_column_names : list[str] | None,
            num_id: int,
            total_steps: int, 
            cov_fn : Optional[Callable],
            lr: float = 1e-3, 
            eps=1e-5, 
            *args, **kwargs) -> None:
        super().__init__(random_effect_configs, num_id, lr, eps, *args, **kwargs)
        self.p_c = parameter_config
        self.v = BoundaryFixedEffect([self.p_c.v_tv], [self.p_c.v_min], [self.p_c.v_max])
        self.k_a = BoundaryFixedEffect([self.p_c.k_a_tv], [self.p_c.k_a_min], [self.p_c.k_a_max])
        self.k_e = BoundaryFixedEffect([self.p_c.k_e_tv], [self.p_c.k_e_min], [self.p_c.k_e_max])
        self.iv_column_names = iv_column_names
        
        self.iv_dim = len(iv_column_names)
        self.cov_fn = cov_fn

        if self.cov_fn is None and self.iv_dim > 0:
            ffn_dims = (self.iv_dim, 7, 7, 7, 1)
            ffn_config = FFNConfig(ffn_dims, dropout=0.0, hidden_norm_layer=False, bias=True, hidden_act_fn="ReLU")
            self.v_ffn = FFN(config=ffn_config)

    
    def forward(self, init: Tensor, time: Tensor, iv: Tensor, id: Tensor) -> Tensor:
        time= time.unsqueeze(-1)
        random_effects = self.random_effects[0](id) # batch, feat.        

        k_a = self.k_a()*random_effects[:,0:1,None].exp() # batch, 1, 1
        k_e = self.k_e()*random_effects[:,1:2,None].exp() # batch, 1, 1

        v_tv = self.v()
        if self.cov_fn is not None and self.iv_dim == 3:
            v = self.cov_fn(v_tv, iv.permute(2,0,1)*self.p_c.v_cov_std).unsqueeze(-1)*random_effects[:,2:3,None].exp()
        elif self.iv_dim > 0 :
            v_cov = self.v_ffn(iv)
            v = v_tv*v_cov.exp()*random_effects[:,2:3,None].exp() # :batch, time, 1
        else :
            v = v_tv*random_effects[:,2:3,None].exp() # :batch, time, 1

        pred = (320 / v * k_a) / ((k_a - k_e).abs() + 1e-6 ) * ((-k_e*time).exp() - (-k_a*time).exp())
        return pred.log().nan_to_num(nan=0.0, posinf=100, neginf=0)
    
    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("sigma", self.error_std_train)
        self.log("v", self.v().squeeze())
        self.log("k_a", self.k_a().squeeze())
        self.log("k_e", self.k_e().squeeze())
        omega = self.random_effects[0].covariance_matrix()
        self.log("omega_k_a", omega[0,0])
        self.log("omega_k_e", omega[1,1])
        self.log("omega_v", omega[2,2])
    
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
        

class Bootstrap :
    def __init__(self) -> None:
        pass

    def run(self):
        pass

def run(
        save_dir:str,
        model_name:str,
        seed : int,
        num_id: int,
        max_epochs: int,
        iv_column_names: List[str],
        v_cov_fn : Optional[Callable],
        lr : float):
    random_effect_config = RandomEffectConfig(3, init_value=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1]])
    batch_size = num_id
    df = generate_dataset(
        seed=seed,
        num_id=num_id,
        parameter_config=ParameterConfig(),
        v_cov_fn=v_fn
    )
    
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
        parameter_config=ParameterConfig(),
        random_effect_configs=[random_effect_config],
        iv_column_names = iv_column_names,
        total_steps=max_epochs,
        lr=lr, 
        num_id=id_len,
        cov_fn=v_cov_fn) 
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{save_dir}", name=f"{model_name}_{seed}")
    trainer = Trainer(
        max_epochs=max_epochs, 
        check_val_every_n_epoch=100, 
        logger=[tb_logger], 
        deterministic=True, 
        accumulate_grad_batches=int(num_id/batch_size))
    trainer.fit(model, datamodule=datamodule)
    writer = PredictWriter(output_path=Path(f"{save_dir}/{model_name}_{seed}_pred.npy"))
    trainer = Trainer(
        max_epochs=1, 
        check_val_every_n_epoch=1, 
        logger=[],
        callbacks=[writer],
        deterministic=True, 
        accumulate_grad_batches=int(num_id/batch_size))
    trainer.predict(model, datamodule=datamodule)


@app.command()
def main(
        save_dir : str,
        max_epochs : int = 10_000,
        lr : float = 1e-3,
        num_id : int = 30):
    seeds = list(range(102))
    seeds.remove(10)
    seeds.remove(14)
    for seed in seeds: 
        try:
            run(save_dir, 'cov_fn_model', seed=seed, iv_column_names=['V_COV1', 'V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=v_fn)
            run(save_dir, 'cov_1_2_3_model', seed=seed, iv_column_names=['V_COV1', 'V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'base_model', seed=seed, iv_column_names=[], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_1_3_model', seed=seed, iv_column_names=['V_COV1', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_1_2_model', seed=seed, iv_column_names=['V_COV1', 'V_COV2',], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_2_3_model', seed=seed, iv_column_names=['V_COV2', 'V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_1_model', seed=seed, iv_column_names=['V_COV1'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_2_model', seed=seed, iv_column_names=['V_COV2'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
            run(save_dir, 'cov_3_model', seed=seed, iv_column_names=['V_COV3'], max_epochs=max_epochs, lr=lr, num_id=num_id, v_cov_fn=None)
        except:
            continue
if __name__ == "__main__":
    # app()
    main("bs0221")