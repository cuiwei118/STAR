from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=8, shuffle=True,
                   return_paths=False, my_collate=None, seed=None, persistent_workers=True):
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = _seed_worker

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=bool(persistent_workers) and num_workers > 0,
        collate_fn=my_collate,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_paths = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name

        return sample

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample


def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    has_paths = len(batch) > 0 and isinstance(batch[0], tuple)
    samples = [b[0] for b in batch] if has_paths else batch
    idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in samples]
    if has_paths:
        batch = [b for i, b in enumerate(batch) if idx[i]]
        kept_samples = [b[0] for b in batch]
        kept_paths = [b[1] for b in batch]
        return torch.utils.data.dataloader.default_collate(kept_samples), kept_paths
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
