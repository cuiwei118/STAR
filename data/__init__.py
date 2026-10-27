from typing import Callable, Dict

import torch

from data.MTLCC.dataloader import get_dataloader as get_mtlcc_dataloader
from data.MTLCC.dataloader import my_collate as mtlcc_collate
from data.MTLCC.data_transforms import MTLCC_transform
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.dataloader import my_collate as pastis_collate
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from data.SouthAfrica.dataloader import get_dataloader as get_southafrica_dataloader
from data.SouthAfrica.data_transforms import SouthAfrica_segmentation_transform

DATASET_INFO = {
    "MTLCC": "MTLCC",
    "PASTIS24": "PASTIS24",
    "PASTIS24_fold1": "PASTIS24",
    "SouthAfricaS2": "SouthAfricaS2",
}


def _family(name: str) -> str:
    if name == "MTLCC":
        return "MTLCC"
    if name == "PASTIS24" or name.startswith("PASTIS24_"):
        return "PASTIS24"
    if name == "SouthAfricaS2":
        return "SouthAfricaS2"
    raise ValueError(f"Unsupported dataset '{name}' in this public release.")


def get_dataloaders(config: dict) -> Dict[str, torch.utils.data.DataLoader]:
    dataloaders = {}
    seed = config.get("seed")
    for split, cfg in config["DATASETS"].items():
        family = _family(cfg["dataset"])
        if family == "MTLCC":
            transform = MTLCC_transform(config["MODEL"], {**cfg, "bidir_input": cfg.get("bidir_input", False)}, split == "train")
            loader, collate = get_mtlcc_dataloader, mtlcc_collate
        elif family == "PASTIS24":
            transform = PASTIS_segmentation_transform(config["MODEL"], cfg, split == "train")
            loader, collate = get_pastis_dataloader, pastis_collate
            dataloaders[split] = loader(
                paths_file=cfg["paths"], root_dir=cfg["base_dir"], transform=transform,
                batch_size=cfg.get("batch_size", 32), num_workers=cfg.get("num_workers", 4),
                shuffle=split == "train", return_paths=cfg.get("return_paths", False),
                my_collate=collate, seed=None if seed is None else int(seed),
                persistent_workers=cfg.get("persistent_workers", True),
            )
            continue
        if family == "MTLCC":
            dataloaders[split] = loader(
                paths_file=cfg["paths"], root_dir=cfg["base_dir"], transform=transform,
                batch_size=cfg.get("batch_size", 32), num_workers=cfg.get("num_workers", 4),
                shuffle=split == "train", return_paths=cfg.get("return_paths", False),
                my_collate=collate, seed=None if seed is None else int(seed),
                persistent_workers=cfg.get("persistent_workers", True),
            )
            continue
        transform = SouthAfrica_segmentation_transform(config["MODEL"], cfg, split == "train")
        dataloaders[split] = get_southafrica_dataloader(
            paths_file=cfg["paths"], root_dir=cfg["base_dir"], transform=transform,
            batch_size=cfg.get("batch_size", 32), num_workers=cfg.get("num_workers", 4),
            shuffle=split == "train", preprocessed_dir=cfg.get("preprocessed_dir"),
            preprocessed_patch_dir=cfg.get("preprocessed_patch_dir"),
            group_by_tile=cfg.get("group_by_tile_batches", False),
        )
    return dataloaders


def get_loss_data_input(config: dict):
    def loss_input_fn(sample, device):
        labels = sample["labels"]
        if labels.shape[-1] == 1:
            labels = labels[..., 0]
        labels = labels.to(device, non_blocking=True).to(torch.int64)
        unknown = sample.get("unk_masks")
        if unknown is None:
            return labels
        if unknown.shape[-1] == 1:
            unknown = unknown[..., 0]
        return labels, unknown.to(device, non_blocking=True).to(torch.bool)
    return loss_input_fn
