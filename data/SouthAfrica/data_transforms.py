from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

from utils.config_files_utils import get_params_values


SOUTH_AFRICA_REMAP = {
    0: 9,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
}
SOUTH_AFRICA_UNK = 9


def SouthAfrica_segmentation_transform(model_config, data_config, is_training):
    max_seq_len = int(data_config["max_seq_len"])
    hflip_prob = float(get_params_values(data_config, "hflip_prob", 0.5))
    vflip_prob = float(get_params_values(data_config, "vflip_prob", 0.5))
    temporal_random_sample = bool(get_params_values(data_config, "temporal_random_sample", False))
    bidir_input = bool(get_params_values(data_config, "bidir_input", False))

    transform_list = [
        ToTensor(),
        RemapLabels(SOUTH_AFRICA_REMAP),
        Normalize(),
        TileDOY(),
        ConcatSpectralAndDOY(),
    ]
    if bidir_input:
        transform_list.append(AddBackwardInputs())
    transform_list.append(
        TemporalSubsampleOrPad(
            max_seq_len=max_seq_len,
            random_sample=temporal_random_sample and is_training,
        )
    )
    if is_training:
        transform_list.append(HVFlip(hflip_prob=hflip_prob, vflip_prob=vflip_prob))
    transform_list.append(UnkMask(unk_class=SOUTH_AFRICA_UNK))
    return transforms.Compose(transform_list)


class ToTensor(object):
    def __call__(self, sample):
        spectral = torch.from_numpy(np.ascontiguousarray(sample["spectral"], dtype=np.float32)).clone()
        doy = torch.from_numpy(np.ascontiguousarray(sample["doy"], dtype=np.float32)).clone()
        labels = torch.from_numpy(np.ascontiguousarray(sample["labels"], dtype=np.int64)).clone()
        return {
            "spectral": spectral,
            "doy": doy,
            "labels": labels,
        }


class RemapLabels(object):
    def __init__(self, labels_dict):
        self.labels_dict = labels_dict

    def __call__(self, sample):
        labels = sample["labels"].clone()
        out = torch.full_like(labels, fill_value=self.labels_dict[0])
        for src, dst in self.labels_dict.items():
            out[labels == int(src)] = int(dst)
        sample["labels"] = out
        return sample


class Normalize(object):
    def __call__(self, sample):
        sample["spectral"] = sample["spectral"] / 255.0
        sample["doy"] = sample["doy"] / 365.0001
        return sample


class TileDOY(object):
    def __call__(self, sample):
        t, h, w, _ = sample["spectral"].shape
        sample["doy_tiled"] = sample["doy"].view(t, 1, 1, 1).repeat(1, h, w, 1)
        return sample


class ConcatSpectralAndDOY(object):
    def __call__(self, sample):
        sample["inputs"] = torch.cat([sample["spectral"], sample["doy_tiled"]], dim=-1)
        del sample["spectral"]
        del sample["doy"]
        del sample["doy_tiled"]
        return sample


class AddBackwardInputs(object):
    def __call__(self, sample):
        sample["inputs_backward"] = torch.flip(sample["inputs"], dims=(0,))
        return sample


class TemporalSubsampleOrPad(object):
    def __init__(self, max_seq_len, random_sample=False):
        self.max_seq_len = int(max_seq_len)
        self.random_sample = bool(random_sample)

    def __call__(self, sample):
        inputs = sample["inputs"]
        seq_len = int(inputs.shape[0])
        selected_inputs = inputs

        if seq_len > self.max_seq_len:
            if self.random_sample:
                idx = torch.randperm(seq_len)[: self.max_seq_len].sort()[0]
            else:
                idx = torch.linspace(0, seq_len - 1, steps=self.max_seq_len).round().long()
            selected_inputs = inputs[idx]
        elif seq_len < self.max_seq_len:
            pad_shape = (self.max_seq_len - seq_len, *inputs.shape[1:])
            pad = torch.zeros(pad_shape, dtype=inputs.dtype)
            selected_inputs = torch.cat([inputs, pad], dim=0)

        sample["inputs"] = selected_inputs

        if "inputs_backward" in sample:
            if seq_len > self.max_seq_len:
                sample["inputs_backward"] = torch.flip(selected_inputs, dims=(0,))
            elif seq_len < self.max_seq_len:
                backward = sample["inputs_backward"]
                pad_shape = (self.max_seq_len - seq_len, *backward.shape[1:])
                pad = torch.zeros(pad_shape, dtype=backward.dtype)
                sample["inputs_backward"] = torch.cat([backward, pad], dim=0)

        sample["seq_lengths"] = torch.tensor(min(seq_len, self.max_seq_len), dtype=torch.int64)
        return sample


class HVFlip(object):
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, sample):
        if torch.rand(1).item() < self.hflip_prob:
            sample["inputs"] = torch.flip(sample["inputs"], dims=(2,))
            if "inputs_backward" in sample:
                sample["inputs_backward"] = torch.flip(sample["inputs_backward"], dims=(2,))
            sample["labels"] = torch.flip(sample["labels"], dims=(1,))
        if torch.rand(1).item() < self.vflip_prob:
            sample["inputs"] = torch.flip(sample["inputs"], dims=(1,))
            if "inputs_backward" in sample:
                sample["inputs_backward"] = torch.flip(sample["inputs_backward"], dims=(1,))
            sample["labels"] = torch.flip(sample["labels"], dims=(0,))
        return sample


class UnkMask(object):
    def __init__(self, unk_class):
        self.unk_class = int(unk_class)

    def __call__(self, sample):
        sample["unk_masks"] = (sample["labels"] != self.unk_class).clone()
        sample["inputs"] = sample["inputs"].contiguous().clone()
        if "inputs_backward" in sample:
            sample["inputs_backward"] = sample["inputs_backward"].contiguous().clone()
        sample["labels"] = sample["labels"].contiguous().clone()
        return sample
