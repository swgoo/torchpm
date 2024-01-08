import pytest

from torchpm.data import MixedEffectsTimeDataModule, MixedEffectsTimeDatasetConfig

@pytest.fixture(scope="class")
def theo_dataset_config() -> MixedEffectsTimeDatasetConfig:
    return MixedEffectsTimeDatasetConfig(
        dv_column_names=['CONC'],
        iv_column_names=['BWT'],
        init_column_names=['AMT'])

@pytest.fixture(scope="class")
def theo_datamodule(theo_dataset_config) -> MixedEffectsTimeDataModule:
    return MixedEffectsTimeDataModule(
            dataset_config=theo_dataset_config,
            train_data='test/THEO.csv')

