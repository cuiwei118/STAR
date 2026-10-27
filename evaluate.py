import sys
import os
import tempfile
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from data import get_dataloaders, get_loss_data_input
from metrics.loss_functions import get_loss
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_runtime_reproducibility(config: dict):
    repro_cfg = config.get("REPRODUCIBILITY", {})
    deterministic = bool(repro_cfg.get("deterministic", False))
    benchmark = bool(repro_cfg.get("benchmark", not deterministic))
    use_tf32 = bool(repro_cfg.get("tf32", not deterministic))
    float32_matmul_precision = str(
        repro_cfg.get("float32_matmul_precision", "high" if use_tf32 else "highest")
    )

    if deterministic and benchmark:
        benchmark = False
        print("[INFO] deterministic=True, forcing cudnn.benchmark=False.")

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(float32_matmul_precision)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)

    return {
        "deterministic": deterministic,
        "benchmark": benchmark,
        "tf32": use_tf32,
        "float32_matmul_precision": float32_matmul_precision,
    }


def configure_temp_dir(config: dict):
    repro_cfg = config.get("REPRODUCIBILITY", {})
    temp_dir = repro_cfg.get("temp_dir", None)
    if not temp_dir:
        return None

    if not os.path.isabs(temp_dir):
        temp_dir = os.path.join(os.getcwd(), temp_dir)

    os.makedirs(temp_dir, exist_ok=True)
    for env_name in ("TMPDIR", "TMP", "TEMP", "TEMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[env_name] = temp_dir
    tempfile.tempdir = temp_dir
    return temp_dir


def _get_colormap(config: dict, num_classes: int) -> np.ndarray:
    dataset_name = str(config.get("DATASETS", {}).get("test", {}).get("dataset", "")).lower()
    if "mtlcc" in dataset_name or num_classes == 17:
        return np.array([
            [0, 142, 192],
            [102, 194, 165],
            [252, 141, 98],
            [141, 160, 203],
            [231, 138, 195],
            [166, 216, 84],
            [255, 217, 47],
            [204, 153, 201],
            [255, 179, 128],
            [179, 205, 227],
            [188, 189, 220],
            [255, 204, 153],
            [255, 153, 153],
            [153, 153, 255],
            [153, 255, 204],
            [255, 255, 153],
            [204, 255, 153],
            [241, 241, 218],
        ], dtype=np.uint8)
    if "pastis" in dataset_name or num_classes == 20:
        return np.array([
            [241, 241, 218],
            [102, 194, 165],
            [252, 141, 98],
            [141, 160, 203],
            [231, 138, 195],
            [166, 216, 84],
            [255, 217, 47],
            [204, 153, 201],
            [255, 179, 128],
            [179, 205, 227],
            [188, 189, 220],
            [255, 204, 153],
            [255, 153, 153],
            [153, 153, 255],
            [153, 255, 204],
            [255, 255, 153],
            [204, 255, 153],
            [153, 204, 255],
            [255, 153, 204],
            [204, 153, 102],
        ], dtype=np.uint8)
    # Fallback: deterministic pseudo-palette for unknown datasets.
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        palette[idx] = ((37 * idx) % 255, (97 * idx) % 255, (167 * idx) % 255)
    return palette


def _decode_segmap(segmentation: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    color_image = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    max_idx = min(colormap.shape[0], int(segmentation.max()) + 1 if segmentation.size > 0 else 0)
    for cls_idx in range(max_idx):
        color_image[segmentation == cls_idx] = colormap[cls_idx]
    return color_image


def _overlay_error_crosses(color_img: np.ndarray, error_mask: np.ndarray, scale: int = 12, cross_color=(0, 0, 0)) -> Image.Image:
    pil = Image.fromarray(color_img.astype(np.uint8))
    up_size = (color_img.shape[1] * scale, color_img.shape[0] * scale)
    pil_up = pil.resize(up_size, resample=Image.NEAREST)
    draw = ImageDraw.Draw(pil_up)
    ys, xs = np.where(error_mask)
    cx_offset = scale // 2
    cy_offset = scale // 2
    dist = max(2, scale // 3)
    for y, x in zip(ys, xs):
        cx = x * scale + cx_offset
        cy = y * scale + cy_offset
        pts = [
            (min(up_size[0] - 1, cx + dist), min(up_size[1] - 1, cy + dist)),
            (max(0, cx - dist), min(up_size[1] - 1, cy + dist)),
            (min(up_size[0] - 1, cx + dist), max(0, cy - dist)),
            (max(0, cx - dist), max(0, cy - dist)),
        ]
        for px, py in pts:
            draw.line([(cx, cy), (px, py)], fill=tuple(int(c) for c in cross_color), width=1)
    return pil_up


def _sample_basename_from_path(path_str: str) -> str:
    p = Path(path_str)
    return p.stem


def _save_visualizations(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, output_dir: str, step: int, start_index: int, colormap: np.ndarray, sample_paths=None):
    pred_dir = os.path.join(output_dir, "pred")
    gt_dir = os.path.join(output_dir, "gt")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    bg_color = np.array([241, 241, 218], dtype=np.uint8)

    batch_size = predicted.shape[0]
    for i in range(batch_size):
        pred_np = predicted[i].cpu().numpy().astype(np.int64)
        target_np = target[i].cpu().numpy().astype(np.int64)
        pred_img = _decode_segmap(pred_np, colormap)
        gt_img = _decode_segmap(target_np, colormap)

        mask_bool = None
        if mask is not None:
            mask_i = np.squeeze(mask[i].cpu().numpy())
            mask_bool = mask_i.astype(bool)
            outside = ~mask_bool
            pred_img[outside] = bg_color
            gt_img[outside] = bg_color

        sample_idx = start_index + i
        stem = f"step{step}_sample{sample_idx}"
        if sample_paths is not None and i < len(sample_paths):
            stem = _sample_basename_from_path(str(sample_paths[i]))

        pred_path = os.path.join(pred_dir, f"{stem}_pred.png")
        gt_path = os.path.join(gt_dir, f"{stem}_gt.png")
        Image.fromarray(pred_img).save(pred_path)
        Image.fromarray(gt_img).save(gt_path)

        diff = np.any(pred_img != gt_img, axis=2)
        if mask_bool is not None:
            diff &= mask_bool
        overlay = _overlay_error_crosses(pred_img, diff, scale=12, cross_color=(0, 0, 0))
        overlay_path = os.path.join(overlay_dir, f"{stem}_pred_x.png")
        overlay.save(overlay_path)


def _init_region_f1_stats(num_classes: int):
    return {
        "boundary": [{"TP": 0, "FP": 0, "FN": 0} for _ in range(num_classes)],
        "interior": [{"TP": 0, "FP": 0, "FN": 0} for _ in range(num_classes)],
    }


def _accumulate_region_f1_stats(
    predicted_np: np.ndarray,
    target_np: np.ndarray,
    mask_np: np.ndarray,
    num_classes: int,
    contour_width: int,
    kernel_size: int,
    stats: dict,
):
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    for cls_idx in range(num_classes):
        gt_cls = target_np == cls_idx
        pred_cls = predicted_np == cls_idx

        if mask_np is not None:
            gt_cls = gt_cls & mask_np
            pred_cls = pred_cls & mask_np

        if not np.any(gt_cls) and not np.any(pred_cls):
            continue

        gt_eroded = binary_erosion(gt_cls, structure=structure, iterations=contour_width)
        pred_eroded = binary_erosion(pred_cls, structure=structure, iterations=contour_width)

        gt_boundary = gt_cls ^ gt_eroded
        pred_boundary = pred_cls ^ pred_eroded
        gt_interior = gt_eroded
        pred_interior = pred_eroded

        for region_name, pred_region, gt_region in (
            ("boundary", pred_boundary, gt_boundary),
            ("interior", pred_interior, gt_interior),
        ):
            region_stats = stats[region_name][cls_idx]
            region_stats["TP"] += int(np.logical_and(pred_region, gt_region).sum())
            region_stats["FP"] += int(np.logical_and(pred_region, ~gt_region).sum())
            region_stats["FN"] += int(np.logical_and(~pred_region, gt_region).sum())


def _finalize_region_f1_stats(stats: dict, num_classes: int):
    result = {}
    for region_name in ("boundary", "interior"):
        class_f1 = np.full(num_classes, np.nan, dtype=np.float64)
        for cls_idx in range(num_classes):
            tp = stats[region_name][cls_idx]["TP"]
            fp = stats[region_name][cls_idx]["FP"]
            fn = stats[region_name][cls_idx]["FN"]
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            class_f1[cls_idx] = 2 * precision * recall / (precision + recall + 1e-8)
        result[region_name] = {
            "class_F1": class_f1.tolist(),
            "mean_F1": float(np.nanmean(class_f1)) if np.any(~np.isnan(class_f1)) else float("nan"),
        }
    return result


def evaluate(
    net,
    evalloader,
    loss_fn,
    loss_input_fn,
    num_classes,
    device,
    use_amp=False,
    visualize=False,
    visualize_dir=None,
    visualize_max_samples=0,
    config=None,
    compute_region_f1=False,
    contour_width=1,
    region_kernel_size=3,
):
    predicted_all = []
    labels_all = []
    losses_all = []
    valid_pixels_total = 0
    pixels_total = 0
    visualized_samples = 0
    colormap = _get_colormap(config or {}, num_classes) if visualize else None
    region_f1_stats = _init_region_f1_stats(num_classes) if compute_region_f1 else None

    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(evalloader):
            sample_paths = None
            if isinstance(sample, (tuple, list)) and len(sample) == 2 and isinstance(sample[1], (list, tuple)):
                sample_paths = sample[1]
                sample = sample[0]
            inputs = sample['inputs'].to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(inputs)
                    logits = logits.permute(0, 2, 3, 1)
                    _, predicted = torch.max(logits.data, -1)
                    ground_truth = loss_input_fn(sample, device)
                    loss = loss_fn['all'](logits, ground_truth)
            else:
                logits = net(inputs)
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)

            target, mask = ground_truth
            if visualize and visualize_dir is not None:
                remaining = visualize_max_samples - visualized_samples if visualize_max_samples > 0 else predicted.shape[0]
                if remaining > 0:
                    take = min(predicted.shape[0], remaining)
                    _save_visualizations(
                        predicted=predicted[:take],
                        target=target[:take],
                        mask=None if mask is None else mask[:take],
                        output_dir=visualize_dir,
                        step=step,
                        start_index=visualized_samples,
                        colormap=colormap,
                        sample_paths=None if sample_paths is None else sample_paths[:take],
                    )
                    visualized_samples += take

            if mask is not None:
                valid_pixels_total += int(mask.sum().item())
                pixels_total += int(mask.numel())
                pred_batch = predicted.view(-1)[mask.view(-1)].cpu().numpy()
                lab_batch = target.view(-1)[mask.view(-1)].cpu().numpy()
            else:
                pixels_total += int(target.numel())
                valid_pixels_total += int(target.numel())
                pred_batch = predicted.view(-1).cpu().numpy()
                lab_batch = target.view(-1).cpu().numpy()

            if pred_batch.shape[0] != lab_batch.shape[0]:
                print(
                    f"[WARN] skipped batch {step}: predicted length {pred_batch.shape[0]} != "
                    f"labels length {lab_batch.shape[0]}"
                )
                continue

            if compute_region_f1:
                predicted_np = predicted.cpu().numpy().astype(np.int64)
                target_np = target.cpu().numpy().astype(np.int64)
                if target_np.ndim == 4 and target_np.shape[-1] == 1:
                    target_np = np.squeeze(target_np, axis=-1)
                mask_np = None
                if mask is not None:
                    mask_np = mask.cpu().numpy()
                    if mask_np.ndim == 4 and mask_np.shape[-1] == 1:
                        mask_np = np.squeeze(mask_np, axis=-1)
                    mask_np = mask_np.astype(bool)
                for sample_idx in range(predicted_np.shape[0]):
                    _accumulate_region_f1_stats(
                        predicted_np=predicted_np[sample_idx],
                        target_np=target_np[sample_idx],
                        mask_np=None if mask_np is None else mask_np[sample_idx],
                        num_classes=num_classes,
                        contour_width=contour_width,
                        kernel_size=region_kernel_size,
                        stats=region_f1_stats,
                    )

            predicted_all.append(pred_batch)
            labels_all.append(lab_batch)
            losses_all.append(loss.view(-1).cpu().detach().numpy())

    if not predicted_all:
        raise RuntimeError("No valid batches were accumulated during evaluation.")

    print("finished iterating over dataset after step %d" % step)
    if pixels_total > 0:
        print(f"[DEBUG] Eval valid pixel ratio: {valid_pixels_total}/{pixels_total} = {valid_pixels_total / pixels_total:.6f}")
    print("calculating metrics...")

    predicted_classes = np.concatenate(predicted_all)
    target_classes = np.concatenate(labels_all)
    losses = np.concatenate(losses_all)

    eval_metrics = get_classification_metrics(
        predicted=predicted_classes,
        labels=target_classes,
        n_classes=num_classes,
        unk_masks=None,
    )

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

    un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)
    class_support = np.array([(target_classes == cls_idx).sum() for cls_idx in range(num_classes)], dtype=np.int64)

    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(
        "Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
        "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s"
        % (
            losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
            micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)
        )
    )
    print("Per-class evaluation details:")
    for cls_idx in range(num_classes):
        print(
            "  class %d | support: %d | IoU: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Accuracy: %.4f"
            % (
                cls_idx, int(class_support[cls_idx]), class_IOU[cls_idx], class_F1[cls_idx],
                class_precision[cls_idx], class_recall[cls_idx], class_acc[cls_idx]
            )
        )
    region_f1_metrics = None
    if compute_region_f1:
        region_f1_metrics = _finalize_region_f1_stats(region_f1_stats, num_classes)
        print("Boundary/Internal F1 details:")
        for cls_idx in range(num_classes):
            boundary_f1 = region_f1_metrics["boundary"]["class_F1"][cls_idx]
            interior_f1 = region_f1_metrics["interior"]["class_F1"][cls_idx]
            if np.isnan(boundary_f1) and np.isnan(interior_f1):
                continue
            print(
                "  class %d | Boundary F1: %.4f | Internal F1: %.4f"
                % (cls_idx, boundary_f1, interior_f1)
            )
        print(
            "  mean   | Boundary F1: %.4f | Internal F1: %.4f"
            % (
                region_f1_metrics["boundary"]["mean_F1"],
                region_f1_metrics["interior"]["mean_F1"],
            )
        )
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

    class_loss_full = np.zeros(num_classes, dtype=np.float64)
    for cls_idx, loss_value in zip(un_labels.tolist(), class_loss.tolist()):
        class_loss_full[int(cls_idx)] = float(loss_value)

    metrics_dict = {
        "macro": {
            "Loss": float(losses.mean()),
            "Accuracy": float(macro_acc),
            "Precision": float(macro_precision),
            "Recall": float(macro_recall),
            "F1": float(macro_F1),
            "IOU": float(macro_IOU),
        },
        "micro": {
            "Loss": float(losses.mean()),
            "Accuracy": float(micro_acc),
            "Precision": float(micro_precision),
            "Recall": float(micro_recall),
            "F1": float(micro_F1),
            "IOU": float(micro_IOU),
        },
        "class": {
            "Loss": class_loss_full.tolist(),
            "Accuracy": class_acc.tolist(),
            "Precision": class_precision.tolist(),
            "Recall": class_recall.tolist(),
            "F1": class_F1.tolist(),
            "IOU": class_IOU.tolist(),
            "Support": class_support.tolist(),
        },
    }
    if region_f1_metrics is not None:
        metrics_dict["region_F1"] = {
            "contour_width": int(contour_width),
            "kernel_size": int(region_kernel_size),
            "boundary": region_f1_metrics["boundary"],
            "internal": region_f1_metrics["interior"],
        }
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on a chosen split.")
    parser.add_argument("--config", required=True, help="configuration (.yaml) file to use")
    parser.add_argument("--checkpoint", required=True, help="checkpoint (.pth) file to evaluate")
    parser.add_argument("--device", type=str, required=True, help="device ids to use, e.g. '0', '0,1', or 'cpu'")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="dataset split to evaluate")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision (AMP) inference")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--save-json", type=str, default=None, help="optional path to save metrics as JSON")
    parser.add_argument("--visualize", action="store_true", help="save prediction/gt/overlay visualizations during evaluation")
    parser.add_argument("--visualize-dir", type=str, default=None, help="directory for visualization outputs; defaults next to save-json or ./eval_visualizations")
    parser.add_argument("--visualize-max-samples", type=int, default=32, help="maximum number of samples to visualize; <=0 means all")
    parser.add_argument("--compute-region-f1", action="store_true", help="compute optional boundary/internal F1 metrics")
    parser.add_argument("--contour-width", type=int, default=1, help="erosion iterations used to define internal region; boundary = parcel - eroded(parcel)")
    parser.add_argument("--region-kernel-size", type=int, default=3, help="odd kernel size for morphology when computing boundary/internal F1")
    args = parser.parse_args()
    if args.region_kernel_size < 1 or args.region_kernel_size % 2 == 0:
        raise ValueError("--region-kernel-size must be a positive odd integer.")

    config = read_yaml(args.config)
    if args.split not in config.get("DATASETS", {}):
        parser.error(f"Split '{args.split}' is not defined in {args.config}.")
    if args.device.strip().lower() == "cpu":
        device_ids = [0]
        allow_cpu = True
    else:
        device_ids = [int(d) for d in args.device.split(",")]
        allow_cpu = False
    config["local_device_ids"] = device_ids
    config["seed"] = int(args.seed)
    if args.visualize:
        config["DATASETS"][args.split]["return_paths"] = True

    set_seed(args.seed)
    runtime_cfg = configure_runtime_reproducibility(config)
    temp_dir = configure_temp_dir(config)
    device = get_device(device_ids, allow_cpu=allow_cpu)

    loader_config = {**config, "DATASETS": {args.split: config["DATASETS"][args.split]}}
    dataloaders = get_dataloaders(loader_config)
    split_loader = dataloaders[args.split]

    print(f"[INFO] Evaluating split: {args.split}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Dataset paths: {config['DATASETS'][args.split]['paths']}")
    print(f"[INFO] Base dir: {config['DATASETS'][args.split]['base_dir']}")
    print(f"[INFO] Runtime reproducibility: {runtime_cfg}")
    if temp_dir is not None:
        print(f"[INFO] Runtime temp dir: {temp_dir}")

    net = get_model(config, device)
    load_from_checkpoint(net, args.checkpoint, partial_restore=False, device=device)
    net.to(device)
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    loss_input_fn = get_loss_data_input(config)
    loss_config = dict(config)
    loss_config["SOLVER"] = config.get("EVALUATION", config.get("SOLVER", {}))
    loss_fn = {
        "all": get_loss(loss_config, device, reduction=None),
        "mean": get_loss(loss_config, device, reduction="mean"),
    }

    metrics_dict = evaluate(
        net=net,
        evalloader=split_loader,
        loss_fn=loss_fn,
        loss_input_fn=loss_input_fn,
        num_classes=config["MODEL"]["num_classes"],
        device=device,
        use_amp=args.amp,
        visualize=args.visualize,
        visualize_dir=(
            args.visualize_dir
            if args.visualize_dir is not None
            else (
                os.path.join(os.path.dirname(args.save_json), "visualizations")
                if args.save_json
                else os.path.join(os.getcwd(), "eval_visualizations")
            )
        ),
        visualize_max_samples=args.visualize_max_samples,
        config=config,
        compute_region_f1=args.compute_region_f1,
        contour_width=args.contour_width,
        region_kernel_size=args.region_kernel_size,
    )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Metrics JSON saved to: {args.save_json}")
