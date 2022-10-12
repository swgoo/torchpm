from dataclasses import dataclass
from typing import Any
import pytorch_lightning as pl
from torchpm.data import *

@dataclass
class PMTrainerConfig :
    dataset : PMDataset
    def __call__(self) -> Dict[str, Any]:
        return {'accumulate_grad_batches' : self.dataset.len}     
    