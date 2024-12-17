from dataclasses import dataclass

import torch


@dataclass
class BaseDatasetConfig:
    train_split_name: str = "train"
    valid_split_name: str = "validation"

    batch_size: int = 16
    num_workers: int = 16
    drop_last: bool = True


class BaseDataset(torch.utils.data.Dataset):
    pass
