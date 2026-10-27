# STAR

Official PyTorch implementation of **Sketch and paint: A structural-temporal
aware refinement network for fine-grained crop mapping on satellite image
time series**, ISPRS Journal of Photogrammetry and Remote Sensing, 2026.
[Paper](https://doi.org/10.1016/j.isprsjprs.2026.09.009).

## 🧩 Framework

![STAR framework](docs/images/star-framework.jpg)

## 🌍 Datasets

| Dataset | Download | Expected format |
|---|---|---|
| PASTIS (France) | [PASTIS24](https://drive.google.com/drive/folders/1Lm0repzD_1NVcECsrwF8Q3bP2XEp2a0y) · [Original PASTIS](https://zenodo.org/records/5012942) | `pickle24x24/` |
| MTLCC (Germany) | [Zenodo](https://zenodo.org/records/5712933) | `data_IJGI18/datasets/full/240pkl/data16/` |
| South Africa | [Source Cooperative](https://source.coop/radiantearth/south-africa-crops-competition) | `preprocessed_tiles/` and `preprocessed_patches_24/` |

### Study areas

![Study areas in France and Germany](docs/images/study-areas-france-germany.png)

![Study area in South Africa](docs/images/study-area-south-africa.png)

## 📊 Results

![Quantitative comparison on PASTIS, Germany, and South Africa](docs/images/benchmark-results.png)

## 🛠️ Environment

Python 3.9, PyTorch 2.1.1, and CUDA-compatible dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Training

Run from the repository root after preparing the data:

```bash
python train.py --config configs/pastis.yaml --device 0 --seed 1234
python train.py --config configs/mtlcc.yaml --device 0 --seed 1234
python train.py --config configs/south_africa.yaml --device 0 --seed 1234
```

Each command trains one dataset. Use `--seed 66`, `--seed 1234`, or
`--seed 2026` for the three runs. Checkpoints and metrics are written to
`outputs/<dataset>/seed_<seed>/`. `best_val.pth` is selected by validation
macro mIoU.

> 📌 **Note:** The paper contains a proofreading oversight: STAR's initial learning-rate value should read **1 × 10⁻³ × 0.5 = 5 × 10⁻⁴**, rather than **1 × 10⁻³**. For reproduction, please follow the provided configuration files strictly.

## 🔎 Evaluation

For example, evaluate the Germany checkpoint saved by training above:

```bash
python evaluate.py \
  --config configs/mtlcc.yaml \
  --checkpoint outputs/mtlcc/seed_1234/best_val.pth \
  --device 0 --split test --seed 1234 \
  --save-json outputs/mtlcc/seed_1234/evaluation.json
```

## 📬 Contact

GitHub messages may not be checked regularly. For questions or issues, please
contact me by email:

cuiwei@whu.edu.cn

## 🙏 Acknowledgements

We thank the authors and maintainers of the following open-source projects
for sharing their implementations, tools, and datasets:

- [DeepSatModels / TSViT](https://github.com/michaeltrs/DeepSatModels).
- [U-TAE / PaPs](https://github.com/VSainteuf/utae-paps).
- [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark).
- [MTLCC](https://github.com/MarcCoru/MTLCC).

We also thank the providers of the
[South Africa crop dataset](https://source.coop/radiantearth/south-africa-crops-competition).

## 📖 Citation

Wei Cui, Kaimin Sun, Yu Duan, Xiao Huang, Tao He, and Fangyi Lv. *Sketch and
paint: A structural-temporal aware refinement network for fine-grained crop
mapping on satellite image time series*. ISPRS Journal of Photogrammetry and
Remote Sensing, 2026. https://doi.org/10.1016/j.isprsjprs.2026.09.009
