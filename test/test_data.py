from dataclasses import field, fields
from torchpm.data import MixedEffectsTimeDataModule, MixedEffectsTimeData
from torch import tensor
class TestData:
    def test_dm(self, theo_datamodule : MixedEffectsTimeDataModule):
        theo_datamodule.setup('fit')
        for i in theo_datamodule.train_dataloader():
            print(i)
        return
    def test_field_name(self):
        obj = MixedEffectsTimeData(tensor(1), tensor([1.]), tensor([1.]), tensor([1.]), tensor([1.]))
        print(fields(obj))

    