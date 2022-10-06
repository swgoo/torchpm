import unittest

from torchpm.para import *
from torchpm.data import *
from torchpm.models import *
from torchpm import predfunc
from tests.predfunc_test import *
from tests.para_test import *

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import TQDMProgressBar

import pytorch_lightning as pl

if __name__ == '__main__' :
    unittest.main()

class ModelsTest(unittest.TestCase) :
    def setUp(self) :
        self.symbolic_df = pd.read_csv('examples/THEO.csv')
        self.symbolic_dataset = PMDataset(self.symbolic_df)
        self.numeric_df = pd.read_csv('examples/THEO_ODE.csv')
        self.numeric_dataset = PMDataset(self.numeric_df)

    def test_simbolic_predfunc(self) :
        omega = ParaTest.get_omega()
        sigma = ParaTest.get_sigma()
        model_config = ModelConfig(
                SymbolicFunction,
                omega=omega,
                sigma=sigma)
        model = FOCEInter(model_config=model_config)

        dataloader = DataLoader(dataset=self.symbolic_dataset, batch_size=12)

        trainer = pl.Trainer(
            log_every_n_steps=1,
            accelerator='gpu',
            callbacks=[TQDMProgressBar(process_position =0)])
        trainer.fit(model, train_dataloaders=dataloader)

    def test_simulate(self) :
        pass

    def add_columns_and_normalize(self, df: pd.DataFrame, rng) :
        ids = df['ID'].unique()

        def get_correlated_value(cor : float, value : float, rng) :
            return cor*value + rng.normal(0, 1, 1)[0] * np.sqrt(1-cor**2)

        subsets : List[pd.DataFrame] = []
        for id in ids :
            mask = df['ID'] == id
            subsets.append(df[mask])

        for i, subset in enumerate(subsets) :
            bwt = subset['BWT'].iloc[0]
            cor_10 = get_correlated_value(0.1, bwt, rng)
            cor_50 = get_correlated_value(0.5, bwt, rng)
            cor_70 = get_correlated_value(0.7, bwt, rng)
            normal_random = rng.normal(0, 1, 1)[0]

            subset.insert(len(subset.columns), 'bwt_cor_0.1', cor_10)
            subset.insert(len(subset.columns), 'bwt_cor_0.5', cor_50)
            subset.insert(len(subset.columns), 'bwt_cor_0.7', cor_70)
            subset.insert(len(subset.columns), 'zero', 0.)

            subset.insert(len(subset.columns), 'normal', normal_random)
        
        df = pd.concat(subsets, axis=0, ignore_index=True)

        for column in ['BWT', 'bwt_cor_0.1', 'bwt_cor_0.5', 'bwt_cor_0.7'] :
            df[column] = (df[column]-df[column].mean()) / df[column].std()
        return df

    def test_transformer(self):

        df = self.symbolic_df
        rng = np.random.default_rng(42)
        df = self.add_columns_and_normalize(df, rng)

        dataset = PMDataset(df)

        omega = ParaTest.get_omega()
        sigma = ParaTest.get_sigma()
        model_config = ModelConfig(
                TransformerSymbolicFunction,
                omega=omega,
                sigma=sigma)
        model = FOCEInter(model_config=model_config, dataset=dataset)

        dataloader = DataLoader(dataset=dataset, batch_size=12)

        trainer = pl.Trainer(
            log_every_n_steps=1,
            accelerator='gpu',
            callbacks=[TQDMProgressBar(process_position =0)])
        trainer.fit(model, train_dataloaders=dataloader)


