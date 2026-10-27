from __future__ import annotations

import os
import pickle
from functools import lru_cache
import random

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset, Sampler
import torch


S2_BAND_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
    "CLM",
]


class _CompatUnpickler(pickle.Unpickler):
    """Load pickles created with NumPy variants that used `numpy._core` paths."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   preprocessed_dir=None, preprocessed_patch_dir=None, group_by_tile=False):
    dataset = SouthAfricaPatchDataset(
        csv_file=paths_file,
        root_dir=root_dir,
        transform=transform,
        preprocessed_dir=preprocessed_dir,
        preprocessed_patch_dir=preprocessed_patch_dir,
    )
    dataloader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=southafrica_collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    if group_by_tile:
        batch_sampler = TileBatchSampler(dataset.tile_ids, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def southafrica_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: southafrica_collate_fn([d[key] for d in batch]) for key in elem}
    if isinstance(elem, torch.Tensor):
        return torch.stack([x.clone() for x in batch], dim=0)
    if isinstance(elem, np.ndarray):
        return torch.stack([torch.as_tensor(np.ascontiguousarray(x)).clone() for x in batch], dim=0)
    if isinstance(elem, (int, np.integer)):
        return torch.tensor(batch, dtype=torch.int64)
    if isinstance(elem, (float, np.floating)):
        return torch.tensor(batch, dtype=torch.float32)
    return batch


class TileBatchSampler(Sampler):
    def __init__(self, tile_ids, batch_size, shuffle=True, drop_last=False):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_samples = len(tile_ids)
        self.tile_groups = {}
        for idx, tile_id in enumerate(tile_ids):
            self.tile_groups.setdefault(tile_id, []).append(idx)
        self.tile_keys = list(self.tile_groups.keys())

    def __iter__(self):
        tile_keys = list(self.tile_keys)
        pending = []
        if self.shuffle:
            random.shuffle(tile_keys)
        for tile_id in tile_keys:
            indices = list(self.tile_groups[tile_id])
            if self.shuffle:
                random.shuffle(indices)
            pending.extend(indices)
            while len(pending) >= self.batch_size:
                yield pending[:self.batch_size]
                pending = pending[self.batch_size:]
        if pending and not self.drop_last:
            yield pending

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


@lru_cache(maxsize=32)
def _load_preprocessed_tile(sample_path: str):
    with open(sample_path, "rb") as handle:
        return _CompatUnpickler(handle).load()


@lru_cache(maxsize=2048)
def _load_preprocessed_patch(sample_path: str):
    with open(sample_path, "rb") as handle:
        return _CompatUnpickler(handle).load()


@lru_cache(maxsize=None)
def _list_tile_dates(root_dir: str, source_split: str, tile_id: str):
    tile_dir = os.path.join(root_dir, source_split, "imagery", "s2", str(tile_id))
    valid_dates = []
    for date_str in sorted(os.listdir(tile_dir)):
        date_dir = os.path.join(tile_dir, date_str)
        if not os.path.isdir(date_dir):
            continue
        all_present = True
        for band in S2_BAND_ORDER:
            band_path = os.path.join(date_dir, f"{tile_id}_{date_str}_{band}_10m.tif")
            if not os.path.exists(band_path):
                all_present = False
                break
            try:
                with rasterio.open(band_path):
                    pass
            except rasterio.errors.RasterioIOError:
                all_present = False
                break
        if all_present:
            valid_dates.append(date_str)
    return tuple(valid_dates)


def _build_patch_cache_path(preprocessed_patch_dir: str, source_split: str, tile_id: str, y0: int, x0: int, patch_size: int):
    return os.path.join(
        preprocessed_patch_dir,
        source_split,
        str(tile_id),
        f"y{y0}_x{x0}_p{patch_size}.pkl",
    )


class SouthAfricaPatchDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, preprocessed_dir=None, preprocessed_patch_dir=None):
        data = pd.read_csv(csv_file)
        self.length = len(data)
        self.tile_ids = data["tile_id"].astype(str).to_numpy()
        if "source_split" in data.columns:
            self.source_splits = data["source_split"].astype(str).to_numpy()
        else:
            self.source_splits = np.full(self.length, "train", dtype=object)
        self.y0 = data["y0"].astype(np.int32).to_numpy()
        self.x0 = data["x0"].astype(np.int32).to_numpy()
        self.patch_sizes = data["patch_size"].astype(np.int32).to_numpy()
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessed_dir = preprocessed_dir
        self.preprocessed_patch_dir = preprocessed_patch_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        source_split = self.source_splits[idx]
        y0 = int(self.y0[idx])
        x0 = int(self.x0[idx])
        patch_size = int(self.patch_sizes[idx])

        patch_cache_path = None
        if self.preprocessed_patch_dir is not None:
            patch_cache_path = _build_patch_cache_path(
                self.preprocessed_patch_dir, source_split, tile_id, y0, x0, patch_size
            )

        patch_sample = None
        if patch_cache_path is not None and os.path.exists(patch_cache_path):
            try:
                patch_sample = _load_preprocessed_patch(patch_cache_path)
            except (EOFError, pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ValueError) as exc:
                print(
                    f"[WARN] Failed to load patch cache {patch_cache_path}: {exc}. "
                    "Falling back to tile cache/raw data."
                )

        if patch_sample is not None:
            spectral = patch_sample["spectral"]
            labels = patch_sample["labels"]
            doy = patch_sample["doy"]
        elif self.preprocessed_dir is not None:
            sample_path = os.path.join(self.preprocessed_dir, source_split, f"{tile_id}.pkl")
            tile_sample = _load_preprocessed_tile(sample_path)
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            spectral = tile_sample["spectral"][:, y0:y1, x0:x1, :]
            labels = tile_sample["labels"][y0:y1, x0:x1]
            doy = tile_sample["doy"]
        else:
            window = Window(x0, y0, patch_size, patch_size)
            label_path = os.path.join(self.root_dir, source_split, "labels", f"{tile_id}.tif")

            with rasterio.open(label_path) as src:
                labels = src.read(1, window=window)

            dates = _list_tile_dates(self.root_dir, source_split, tile_id)
            spectral_seq = []
            doy = []
            for date_str in dates:
                date_dir = os.path.join(self.root_dir, source_split, "imagery", "s2", tile_id, date_str)
                band_stack = []
                for band in S2_BAND_ORDER:
                    band_path = os.path.join(date_dir, f"{tile_id}_{date_str}_{band}_10m.tif")
                    with rasterio.open(band_path) as src:
                        band_stack.append(src.read(1, window=window))
                spectral_seq.append(np.stack(band_stack, axis=-1))
                # date format YYYY_MM_DD
                year, month, day = [int(x) for x in date_str.split("_")]
                doy.append(pd.Timestamp(year=year, month=month, day=day).dayofyear)
            spectral = np.stack(spectral_seq, axis=0)

        sample = {
            "spectral": np.ascontiguousarray(spectral),
            "doy": np.ascontiguousarray(np.asarray(doy, dtype=np.float32)),
            "labels": np.ascontiguousarray(labels),
            "tile_id": tile_id,
            "y0": y0,
            "x0": x0,
            "patch_size": patch_size,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
