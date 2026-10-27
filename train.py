import sys
import os
import json
import ast
import tempfile
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os,random
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from tqdm import tqdm

def set_seed(seed=1234): #42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


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


def _parse_steps_config(steps_cfg):
    if steps_cfg is None:
        return None
    if isinstance(steps_cfg, int):
        return steps_cfg
    if isinstance(steps_cfg, (list, tuple)):
        if not steps_cfg:
            return None
        return int(steps_cfg[-1])
    if isinstance(steps_cfg, str):
        steps_cfg = steps_cfg.strip()
        if not steps_cfg:
            return None
        if steps_cfg.isdigit():
            return int(steps_cfg)
        try:
            parsed = ast.literal_eval(steps_cfg)
        except (ValueError, SyntaxError):
            return None
        return _parse_steps_config(parsed)
    return None


def resolve_total_train_steps(config: dict, steps_per_epoch: int) -> int:
    solver_cfg = config.get("SOLVER", {})
    explicit_max_steps = get_params_values(solver_cfg, "max_steps", None)
    if explicit_max_steps is not None:
        return max(1, int(explicit_max_steps))

    parsed_steps = _parse_steps_config(get_params_values(solver_cfg, "steps", None))
    if parsed_steps is not None:
        return max(1, int(parsed_steps))

    num_epochs = int(get_params_values(solver_cfg, "num_epochs", 1))
    return max(1, num_epochs * max(1, steps_per_epoch))


def resolve_scheduler_total_steps(config: dict, steps_per_epoch: int, total_train_steps: int) -> int:
    solver_cfg = config.get("SOLVER", {})
    scheduler_steps = get_params_values(solver_cfg, "scheduler_steps", None)
    parsed_scheduler_steps = _parse_steps_config(scheduler_steps)
    if parsed_scheduler_steps is not None:
        return max(1, int(parsed_scheduler_steps))
    return max(1, int(total_train_steps))


def build_step_aligned_scheduler_config(config: dict, steps_per_epoch: int, scheduler_total_steps: int) -> dict:
    scheduler_config = dict(config)
    scheduler_config["SOLVER"] = dict(config["SOLVER"])
    scheduler_config["SOLVER"]["num_epochs"] = float(scheduler_total_steps) / max(1, steps_per_epoch)
    return scheduler_config

def train_and_evaluate(net, dataloaders, config, device, lin_cls=False, use_amp=False):
    def _get_state_dict_to_save(model, device_ids):
        return model.module.state_dict() if len(device_ids) > 1 else model.state_dict()

    def _update_confusion_matrix(cm, predicted_np, labels_np, num_classes):
        valid = (labels_np >= 0) & (labels_np < num_classes)
        if valid.ndim > 1:
            valid = valid.reshape(-1)
        predicted_np = predicted_np.reshape(-1)[valid]
        labels_np = labels_np.reshape(-1)[valid]
        encoded = labels_np * num_classes + predicted_np
        cm += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        return cm

    def _macro_iou_from_confusion_matrix(cm):
        cm = cm.astype(np.float64, copy=False)
        diag = np.diag(cm)
        rowsum = cm.sum(axis=1)
        colsum = cm.sum(axis=0)
        denom = rowsum + colsum - diag
        iou = np.divide(diag, denom, out=np.full_like(diag, np.nan, dtype=np.float64), where=denom > 0)
        return float(np.nanmean(iou))

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn,epoch, scaler=None, use_amp=False):
        import time
        optimizer.zero_grad()
        t1 = time.time()
        inputs = sample['inputs'].to(device, non_blocking=True)
        ground_truth = loss_input_fn(sample, device)
        # ground_truth is expected to already include unknown mask generated by
        # transforms (Add2UnkClass/UnkMask). We rely on that mask so do not
        # perform additional train-time keep_classes masking here.
        t2 = time.time()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                outputs = outputs.permute(0, 2, 3, 1)
                loss = loss_fn['mean'](outputs, ground_truth)
            scaler.scale(loss).backward()
            t3 = time.time()
            scaler.step(optimizer)
            scaler.update()
            t4 = time.time()
        else:
            outputs = net(inputs)
            t3 = time.time()
            outputs = outputs.permute(0, 2, 3, 1)
            loss = loss_fn['mean'](outputs, ground_truth)
            loss.backward()
            t4 = time.time()
            optimizer.step()
        t5 = time.time()
        timings = {
            'data_ms': (t2 - t1) * 1000.0,
            'forward_ms': (t3 - t2) * 1000.0,
            'backward_ms': (t4 - t3) * 1000.0,
            'optimizer_ms': (t5 - t4) * 1000.0,
            'batch_ms': (t5 - t1) * 1000.0,
        }
        return outputs, ground_truth, loss, timings

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        loss_sum = 0.0
        loss_count = 0
        valid_pixels_total = 0
        pixels_total = 0

        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits= net(sample['inputs'].to(device))

                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                # Convert to numpy arrays and ensure per-batch lengths match.
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
                batch_loss_tensor = loss.reshape(-1).detach()
                if mask is not None and batch_loss_tensor.numel() == mask.numel():
                    batch_loss_tensor = batch_loss_tensor[mask.view(-1)]
                finite_mask = torch.isfinite(batch_loss_tensor)
                if finite_mask.numel() > 0:
                    batch_loss_tensor = batch_loss_tensor[finite_mask]
                batch_loss = batch_loss_tensor.cpu().numpy()

                if pred_batch.shape[0] != lab_batch.shape[0]:
                    continue
                predicted_all.append(pred_batch)
                labels_all.append(lab_batch)
                if batch_loss.size > 0:
                    losses_all.append(batch_loss)
                    loss_sum += float(batch_loss.sum())
                    loss_count += int(batch_loss.size)

        print("finished iterating over dataset after step %d" % step)
        if pixels_total > 0:
            print(f"[DEBUG] Eval valid pixel ratio: {valid_pixels_total}/{pixels_total} = {valid_pixels_total / pixels_total:.6f}")
        print("calculating metrics...")
        if not predicted_all or not labels_all:
            nan = float('nan')
            empty_class = np.full(num_classes, np.nan, dtype=np.float64)
            empty_support = np.zeros(num_classes, dtype=np.int64)
            return (
                np.arange(num_classes, dtype=np.int64),
                {
                    "macro": {"Loss": nan, "Accuracy": nan, "Precision": nan, "Recall": nan, "F1": nan, "IOU": nan},
                    "micro": {"Loss": nan, "Accuracy": nan, "Precision": nan, "Recall": nan, "F1": nan, "IOU": nan},
                    "class": {"Loss": empty_class, "Accuracy": empty_class, "Precision": empty_class,
                              "Recall": empty_class, "F1": empty_class, "IOU": empty_class, "Support": empty_support},
                },
            )

        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all) if losses_all else np.array([], dtype=np.float64)

        eval_metrics = get_classification_metrics(
            predicted=predicted_classes,
            labels=target_classes,
            n_classes=num_classes,
            unk_masks=None,
        )
        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        if losses.size > 0:
            un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)
            mean_loss = float(loss_sum / max(1, loss_count))
        else:
            un_labels = np.array([], dtype=np.int64)
            class_loss = np.array([], dtype=np.float64)
            mean_loss = float('nan')

        class_support = np.array([(target_classes == cls_idx).sum() for cls_idx in range(num_classes)], dtype=np.int64)
        valid_classes = class_support > 0
        class_loss_full = np.full(num_classes, np.nan, dtype=np.float64)
        for cls_idx, loss_value in zip(un_labels.tolist(), class_loss.tolist()):
            class_loss_full[int(cls_idx)] = float(loss_value)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(
            "Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
            "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
            (
                mean_loss, micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
                micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)
            )
        )
        print("Per-class evaluation details:")
        for cls_idx in range(num_classes):
            if class_support[cls_idx] <= 0:
                continue
            print(
                "  class %d | support: %d | IoU: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f | Accuracy: %.4f" %
                (
                    cls_idx, int(class_support[cls_idx]), class_IOU[cls_idx], class_F1[cls_idx],
                    class_precision[cls_idx], class_recall[cls_idx], class_acc[cls_idx]
                )
            )
        print("Valid classes: %d/%d" % (int(valid_classes.sum()), num_classes))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (
            np.arange(num_classes, dtype=np.int64),
            {
                "macro": {"Loss": mean_loss, "Accuracy": macro_acc, "Precision": macro_precision,
                          "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                "micro": {"Loss": mean_loss, "Accuracy": micro_acc, "Precision": micro_precision,
                          "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                "class": {"Loss": class_loss_full, "Accuracy": class_acc, "Precision": class_precision,
                          "Recall": class_recall, "F1": class_F1, "IOU": class_IOU, "Support": class_support},
            },
        )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    test_eval_steps = int(get_params_values(config['CHECKPOINT'], "test_eval_steps", 0) or 0)
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    results_path = get_params_values(config.get('RESULTS', {}), "save_path", save_path)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    total_train_steps = resolve_total_train_steps(config, num_steps_train)
    scheduler_total_steps = resolve_scheduler_total_steps(config, num_steps_train, total_train_steps)
    max_epochs = max(int(num_epochs), (total_train_steps + max(1, num_steps_train) - 1) // max(1, num_steps_train))

    # Early stopping defaults to validation loss, which is a safer signal than
    # a noisy target-year IoU during cross-year experiments.
    early_stop_cfg = get_params_values(config['SOLVER'], "early_stopping", {})
    early_stop_enabled = get_params_values(early_stop_cfg, "enabled", True)
    patience = int(get_params_values(early_stop_cfg, "patience", 20))
    early_stop_monitor = get_params_values(early_stop_cfg, "monitor", "Loss")
    early_stop_mode = get_params_values(early_stop_cfg, "mode", "min")
    early_stop_min_delta = float(get_params_values(early_stop_cfg, "min_delta", 1e-4))
    patience_counter = 0
    if early_stop_mode == "min":
        best_monitor_value = float("inf")
    else:
        best_monitor_value = float("-inf")
    test_best_monitor = str(get_params_values(config['CHECKPOINT'], "test_best_monitor", "IOU"))
    test_best_mode = str(get_params_values(config['CHECKPOINT'], "test_best_mode", "max")).lower()
    test_best_min_delta = float(get_params_values(config['CHECKPOINT'], "test_best_min_delta", 1e-4))
    if test_best_mode == "min":
        best_test_value = float("inf")
    else:
        best_test_value = float("-inf")
    should_stop = False

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    print("current learn rate: ", lr)
    print(f"training will run for up to {total_train_steps} optimizer steps")
    print(f"lr scheduler is parameterized for {scheduler_total_steps} steps")
    print(
        f"eval/save cadence: eval_steps={eval_steps}, test_eval_steps={test_eval_steps}, "
        f"save_steps={save_steps}, train_metrics_steps={train_metrics_steps}"
    )
    print(f"best checkpoint monitor: {early_stop_monitor} ({early_stop_mode})")
    if str(early_stop_monitor).upper() != "IOU" or str(early_stop_mode).lower() != "max":
        print("[WARN] Current config is not selecting best checkpoint by val IOU. "
              "For peak mIoU hunting, set SOLVER.early_stopping.monitor=IOU and mode=max.")


    net.to(device)
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)
    if results_path and (not os.path.exists(results_path)):
        os.makedirs(results_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    scheduler_config = build_step_aligned_scheduler_config(config, num_steps_train, scheduler_total_steps)
    scheduler = build_scheduler(scheduler_config, optimizer, num_steps_train)
    writer = SummaryWriter(save_path)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    best_monitor_step = None
    best_monitor_epoch = None
    best_test_step = None
    best_test_epoch = None
    last_val_macro = {'Loss': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'),
                      'Recall': float('nan'), 'F1': float('nan'), 'IOU': float('nan')}
    global_step = start_global - 1

    for epoch in range(start_epoch, start_epoch + max_epochs):
        net.train()
        if should_stop:
            print(f"Early stop triggered at the end of epoch {epoch-1}.")
            break
        if global_step >= total_train_steps:
            print(f"Reached target training budget ({total_train_steps} steps) at the end of epoch {epoch-1}.")
            break
        import time
        epoch_start_time = time.time()
        batch_count = 0
        total_data_ms = 0.0
        total_forward_ms = 0.0
        total_backward_ms = 0.0
        total_optimizer_ms = 0.0
        total_batch_ms = 0.0
        train_metrics_acc = {'Loss': 0.0, 'IOU': 0.0, 'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        train_metrics_count = 0

        use_tqdm = bool(get_params_values(config['SOLVER'], "use_tqdm", True))
        disable_tqdm = (not use_tqdm) or (not sys.stdout.isatty())
        with tqdm(total=num_steps_train, desc=f"Epoch {epoch}", ncols=120, disable=disable_tqdm) as pbar:
            for step, sample in enumerate(dataloaders['train']):
                if should_stop or global_step >= total_train_steps:
                    break
                global_step += 1
                abs_step = global_step
                logits, ground_truth, loss, timings = train_step(
                    net, sample, loss_fn, optimizer, device,
                    loss_input_fn=loss_input_fn, epoch=epoch,
                    scaler=scaler, use_amp=use_amp
                )
                batch_count += 1
                total_data_ms += timings['data_ms']
                total_forward_ms += timings['forward_ms']
                total_backward_ms += timings['backward_ms']
                total_optimizer_ms += timings['optimizer_ms']
                total_batch_ms += timings['batch_ms']
                if len(ground_truth) == 2:
                    labels, unk_masks = ground_truth
                else:
                    labels = ground_truth
                    unk_masks = None
                # accumulate per-batch train metrics to report once per epoch
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                for k in list(train_metrics_acc.keys()):
                    train_metrics_acc[k] += float(batch_metrics.get(k, 0.0))
                train_metrics_count += 1
                if scheduler is not None:
                    scheduler.step_update(abs_step)

                if (train_metrics_steps > 0 and abs_step % train_metrics_steps == 0) or abs_step == total_train_steps:
                    write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                    print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                          "batch recall: %.4f, batch F1: %.4f" %
                          (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                           batch_metrics['Recall'], batch_metrics['F1']))

                if (save_steps > 0 and abs_step % save_steps == 0) or abs_step == total_train_steps:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                    else:
                        torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

                should_eval_now = ((eval_steps > 0 and abs_step % eval_steps == 0) or abs_step == total_train_steps)
                if should_eval_now:
                    print(f"[DEBUG] START EVAL step={abs_step} epoch={epoch}", flush=True)
                    try:
                        eval_metrics = evaluate(net, dataloaders['val'], loss_fn, config)
                        print(f"[DEBUG] END EVAL step={abs_step} epoch={epoch}", flush=True)
                        val_macro = eval_metrics[1]['macro']
                        val_micro = eval_metrics[1]['micro']
                    except Exception as e:
                        print(f"Validation failed at step {abs_step}: {e}", flush=True)
                        eval_metrics = None
                        val_macro = {'Loss': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'),
                                     'Recall': float('nan'), 'F1': float('nan'), 'IOU': float('nan')}
                        val_micro = dict(val_macro)

                    last_val_macro = val_macro
                    print(f"[DEBUG] START WRITE_EVAL_SUMMARIES step={abs_step} epoch={epoch}", flush=True)
                    write_mean_summaries(writer, val_micro, abs_step, mode="eval_micro", optimizer=None)
                    write_mean_summaries(writer, val_macro, abs_step, mode="eval_macro", optimizer=None)
                    if eval_metrics is not None:
                        write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                              optimizer=None)
                    print(f"[DEBUG] END WRITE_EVAL_SUMMARIES step={abs_step} epoch={epoch}", flush=True)

                    print(f"Eval @ step {abs_step}: Val loss {val_macro['Loss']:.7f}, IOU {val_macro['IOU']:.4f}, "
                          f"Accuracy {val_macro['Accuracy']:.4f}, F1 {val_macro['F1']:.4f}", flush=True)

                    current_monitor_value = float(val_macro.get(early_stop_monitor, float("nan")))
                    if early_stop_mode == "min":
                        improved = current_monitor_value < (best_monitor_value - early_stop_min_delta)
                    else:
                        improved = current_monitor_value > (best_monitor_value + early_stop_min_delta)

                    if improved:
                        print(f"[DEBUG] START SAVE_BEST step={abs_step} epoch={epoch}", flush=True)
                        best_monitor_value = current_monitor_value
                        best_monitor_step = abs_step
                        best_monitor_epoch = epoch
                        patience_counter = 0
                        state_dict_to_save = _get_state_dict_to_save(net, local_device_ids)
                        torch.save(state_dict_to_save, "%s/best_val.pth" % (save_path))
                        torch.save(state_dict_to_save, "%s/best.pth" % (save_path))
                        print(f"Step {abs_step}: New best {early_stop_monitor} = {best_monitor_value:.6f}. "
                              f"Model saved as best_val.pth and best.pth. Patience counter reset.", flush=True)
                        print(f"[DEBUG] END SAVE_BEST step={abs_step} epoch={epoch}", flush=True)
                    else:
                        patience_counter += 1
                        print(f"Step {abs_step}: {early_stop_monitor} did not improve "
                              f"(current: {current_monitor_value:.6f}, best: {best_monitor_value:.6f}). "
                              f"Patience: {patience_counter}/{patience}", flush=True)
                        if early_stop_enabled and patience_counter >= patience:
                            print(f"Early stopping condition met at step {abs_step} "
                                  f"(monitor={early_stop_monitor}, mode={early_stop_mode}, patience={patience}).",
                                  flush=True)
                            should_stop = True

                    net.train()

                should_test_eval_now = (
                    'test' in dataloaders and (
                        (test_eval_steps > 0 and abs_step % test_eval_steps == 0) or
                        (test_eval_steps > 0 and abs_step == total_train_steps)
                    )
                )
                if should_test_eval_now:
                    print(f"[DEBUG] START TEST_EVAL step={abs_step} epoch={epoch}", flush=True)
                    test_eval_metrics = None
                    try:
                        test_eval_metrics = evaluate(net, dataloaders['test'], loss_fn, config)
                        print(f"[DEBUG] END TEST_EVAL step={abs_step} epoch={epoch}", flush=True)
                        test_macro = test_eval_metrics[1]['macro']
                        test_micro = test_eval_metrics[1]['micro']
                    except Exception as e:
                        print(f"Test evaluation failed at step {abs_step}: {e}", flush=True)
                        test_macro = {'Loss': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'),
                                      'Recall': float('nan'), 'F1': float('nan'), 'IOU': float('nan')}
                        test_micro = dict(test_macro)

                    print(f"[DEBUG] START WRITE_TEST_SUMMARIES step={abs_step} epoch={epoch}", flush=True)
                    write_mean_summaries(writer, test_micro, abs_step, mode="test_micro", optimizer=None)
                    write_mean_summaries(writer, test_macro, abs_step, mode="test_macro", optimizer=None)
                    if test_eval_metrics is not None:
                        write_class_summaries(writer, [test_eval_metrics[0], test_eval_metrics[1]['class']], abs_step,
                                              mode="test", optimizer=None)
                    print(f"[DEBUG] END WRITE_TEST_SUMMARIES step={abs_step} epoch={epoch}", flush=True)
                    print(
                        f"Test @ step {abs_step}: Test loss {test_macro['Loss']:.7f}, IOU {test_macro['IOU']:.4f}, "
                        f"Accuracy {test_macro['Accuracy']:.4f}, F1 {test_macro['F1']:.4f}",
                        flush=True
                    )
                    current_test_value = float(test_macro.get(test_best_monitor, float("nan")))
                    if test_best_mode == "min":
                        test_improved = current_test_value < (best_test_value - test_best_min_delta)
                    else:
                        test_improved = current_test_value > (best_test_value + test_best_min_delta)

                    if test_improved:
                        best_test_value = current_test_value
                        best_test_step = abs_step
                        best_test_epoch = epoch
                        state_dict_to_save = _get_state_dict_to_save(net, local_device_ids)
                        torch.save(state_dict_to_save, "%s/best_test.pth" % (save_path))
                        print(
                            f"Step {abs_step}: New best test {test_best_monitor} = {best_test_value:.6f}. "
                            f"Model saved as best_test.pth.",
                            flush=True
                        )
                    net.train()
                pbar.update(1)

        if train_metrics_count > 0:
            avg_train_metrics = {k: train_metrics_acc[k] / train_metrics_count for k in train_metrics_acc}
        else:
            avg_train_metrics = {'Loss': float('nan'), 'IOU': float('nan'), 'Accuracy': float('nan'), 'Precision': float('nan'), 'Recall': float('nan'), 'F1': float('nan')}

        print(f"[DEBUG] START WRITE_TRAIN_EPOCH_SUMMARY epoch={epoch}", flush=True)
        write_mean_summaries(writer, {'IOU': avg_train_metrics['IOU'], 'Accuracy': avg_train_metrics['Accuracy'],
                                                                    'Precision': avg_train_metrics['Precision'], 'Recall': avg_train_metrics['Recall'],
                                                                    'F1': avg_train_metrics['F1'], 'Loss': avg_train_metrics['Loss']},
                                                            global_step, mode="train_epoch", optimizer=optimizer)
        print(f"[DEBUG] END WRITE_TRAIN_EPOCH_SUMMARY epoch={epoch}", flush=True)

        epoch_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch} summary: Train loss {avg_train_metrics['Loss']:.7f}, IOU {avg_train_metrics['IOU']:.4f} | "
                    f"Last val loss {last_val_macro['Loss']:.7f}, IOU {last_val_macro['IOU']:.4f} | "
                    f"epoch_time {epoch_elapsed:.1f}s", flush=True)

        if should_stop or global_step >= total_train_steps:
            break

    best_ckpt_path = os.path.join(save_path, "best.pth")
    if os.path.isfile(best_ckpt_path):
        state_dict = torch.load(best_ckpt_path, map_location=device)
        if len(local_device_ids) > 1:
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print(f"Loaded best checkpoint from {best_ckpt_path} for final test evaluation.")
    else:
        print(f"best.pth not found at {best_ckpt_path}. Running final test with current model weights.")

    test_metrics_payload = None
    if 'test' in dataloaders:
        print("Running final test evaluation...")
        test_eval = evaluate(net, dataloaders['test'], loss_fn, config)
        test_metrics_payload = {
            "labels": [int(x) for x in test_eval[0]],
            "metrics": {
                split_name: {
                    key: value.tolist() if hasattr(value, "tolist") else float(value) if isinstance(value, (np.floating, np.integer)) else value
                    for key, value in split_metrics.items()
                }
                for split_name, split_metrics in test_eval[1].items()
            },
            "best_monitor": {
                "name": early_stop_monitor,
                "mode": early_stop_mode,
                "value": float(best_monitor_value),
                "step": int(best_monitor_step) if best_monitor_step is not None else None,
                "epoch": int(best_monitor_epoch) if best_monitor_epoch is not None else None,
            },
            "best_test_monitor": {
                "name": test_best_monitor,
                "mode": test_best_mode,
                "value": float(best_test_value),
                "step": int(best_test_step) if best_test_step is not None else None,
                "epoch": int(best_test_epoch) if best_test_epoch is not None else None,
            },
        }
        test_metrics_file = os.path.join(results_path, "test_metrics.json")
        with open(test_metrics_file, "w") as f:
            json.dump(test_metrics_payload, f, indent=2)
        print(f"Saved final test metrics to {test_metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', type=str, help='gpu ids to use')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP) training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible repeated runs')
    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin
    use_amp = args.amp
    seed = int(args.seed)
    set_seed(seed)
    device = get_device(device_ids, allow_cpu=False)
    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['seed'] = seed
    runtime_cfg = configure_runtime_reproducibility(config)
    temp_dir = configure_temp_dir(config)
    if 'CHECKPOINT' in config and 'save_path' in config['CHECKPOINT']:
        base_save_path = config['CHECKPOINT']['save_path']
        config['CHECKPOINT']['save_path'] = os.path.join(base_save_path, f"seed_{seed}")
    if 'RESULTS' in config and 'save_path' in config['RESULTS']:
        base_results_path = config['RESULTS']['save_path']
        config['RESULTS']['save_path'] = os.path.join(base_results_path, f"seed_{seed}")
    # 打印实际用到的数据根路径，便于确认数据是否在SSD
    # 兼容 base_dir 和 basedir
    base_dir = None
    if 'base_dir' in config['DATASETS']['train']:
        base_dir = config['DATASETS']['train']['base_dir']
    elif 'basedir' in config['DATASETS']['train']:
        base_dir = config['DATASETS']['train']['basedir']
    print(f'【DEBUG】train base_dir/basedir: {base_dir}')
    if base_dir is None:
        print('【DEBUG】train base_dir/basedir 未找到，建议将yaml中的basedir统一改为base_dir')
    if temp_dir is not None:
        print(f"[INFO] Runtime temp dir: {temp_dir}")
    dataloaders = get_dataloaders(config)
    # 更详细的 debug：打印 train 配置和 DATASET_INFO 的键，便于定位问题
    try:
        print('【DEBUG】config[DATASETS][train]:', config.get('DATASETS', {}).get('train'))
        from data import DATASET_INFO
        print('【DEBUG】DATASET_INFO keys:', list(DATASET_INFO.keys()))
        ds_name = config.get('DATASETS', {}).get('train', {}).get('dataset')
        print('【DEBUG】train dataset name:', ds_name)
        if ds_name in DATASET_INFO:
            print('【DEBUG】DATASET_INFO[ds_name]:', DATASET_INFO[ds_name])
        else:
            print('【DEBUG】dataset name not found in DATASET_INFO')
    except Exception as e:
        print('【DEBUG】error while printing DATASET_INFO:', e)
    print(f'【DEBUG】seed: {seed}')
    print(f'【DEBUG】reproducibility: {config.get("REPRODUCIBILITY", {})}')
    print(f'【DEBUG】runtime_applied: {runtime_cfg}')
    print(f'【DEBUG】save_path: {config.get("CHECKPOINT", {}).get("save_path")}')
    net = get_model(config, device)
    train_and_evaluate(net, dataloaders, config, device, lin_cls=lin_cls, use_amp=use_amp)
