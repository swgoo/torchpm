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
        dataframe = pd.read_csv('examples/THEO.csv')
        self.symbolic_dataset = PMDataset(dataframe)
        dataframe = pd.read_csv('examples/THEO_ODE.csv')
        self.numeric_dataset = PMDataset(dataframe)

    def test_simbolic_predfunc(self) :
        omega = ParaTest.get_omega()
        sigma = ParaTest.get_sigma()
        model_config = ModelConfig(
                self.symbolic_dataset,
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