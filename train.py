import sys
import time
import atexit
from datetime import datetime

sys.path.append('./')

import os
import copy
import numpy as np
import torch

sys.path.append('./')
import argparse
from modules import dataset, network, loss, transform
from utils import yaml_config_hook, metric, initialization_utils


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Stable implementation defaults are kept here so dataset YAML files only
# describe experiment-specific choices.  They are merged before argparse is
# built, therefore every option remains available as a command-line override.
COMMON_CONFIG_DEFAULTS = {
    "image_size": 7,
    "dim_emebeding": 512,
    "projection_dim": 128,
    "cluster_freeze_epochs": 0,
    "cluster_coverage_min_samples": 1,
    "reseed_empty_clusters": True,
    "reseed_undercovered_clusters": False,
    "cluster_reseed_max_attempts": 5,
    "cluster_reseed_candidate_pool": 256,
    "cluster_repair_strategy": "legacy",
    "cluster_repair_cooldown_epochs": 0,
    "cluster_stats_refresh_every_epoch": False,
    "post_epoch_cluster_guard": False,
    "shared_pretrain_checkpoint": "",
    "deterministic_stage_seeding": False,
    "kmeans_seed": -1,
    "joint_seed": -1,
    "kmeans_init_sample_size": 8192,
    "weight_decay": 0.0005,
    "cluster_lambda_start_ratio": 1.0,
    "cluster_lambda_ramp_epochs": 1,
    "balance_coef": 0.0,
    "center_orthogonal_coef": 0.0,
    "small_cluster_protection_weight": 0.0,
    "small_cluster_floor_factor": 0.5,
    "small_cluster_min_ratio": 0.0,
    "ema_anchor_momentum": 0.995,
    "reliability_graph_alpha": 0.5,
    "reliability_graph_steps": 1,
    "reliability_graph_topk": 0,
    "rac_connection_mode": "projection",
    "rac_encoder_mix": 0.0,
    "clu_min_confidence": 0.0,
    "clu_min_samples_per_cluster": 1,
    "use_cluster_balanced_clu": False,
    "preserve_high_confidence_on_incomplete_coverage": False,
    "save_best_oa_checkpoint": True,
    "dataloader_prefetch_factor": 4,
    "initialization_batch_size": 512,
    "auxiliary_batch_size": 512,
    "enable_tf32": False,
    "use_amp": False,
    "amp_dtype": "float16",
    "eval_every_epochs": 1,
}

def build_effective_config(raw_config):
    """Merge fixed implementation defaults, then explicit experiment values."""
    effective = dict(COMMON_CONFIG_DEFAULTS)
    effective.update(raw_config)
    return effective


def apply_runtime_metadata(args):
    if not hasattr(args, "cluster_protection_profile"):
        args.cluster_protection_profile = str(args.dataset)


def build_pretrain_signature(args):
    """Describe representation-learning choices that make a checkpoint reusable."""
    return {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "modality_files": list(getattr(args, "modality_files", [])),
        "image_size": int(args.image_size),
        "dim_emebeding": int(args.dim_emebeding),
        "projection_dim": int(getattr(args, "projection_dim", 128)),
        "batch_size": int(args.batch_size),
        "pretrain_epoch": int(getattr(args, "pretrain_epoch", 10)),
        "pretrain_learning_rate": float(args.pretrain_learning_rate),
        "contrastive_param": float(args.contrastive_param),
        "crop_scale_min": float(getattr(args, "crop_scale_min", 0.7)),
        "mask_pixel_prob": float(getattr(args, "mask_pixel_prob", 0.1)),
        "mask_band_prob": float(getattr(args, "mask_band_prob", 0.1)),
        "use_reliability_fusion": bool(
            getattr(args, "use_reliability_fusion", False)
        ),
        "reliability_min_weight": float(
            getattr(args, "reliability_min_weight", 0.0)
        ),
        "reliability_temperature": float(
            getattr(args, "reliability_temperature", 1.0)
        ),
        "reliability_preserve_scale": bool(
            getattr(args, "reliability_preserve_scale", False)
        ),
        "reliability_uniform_init": bool(
            getattr(args, "reliability_uniform_init", False)
        ),
        "modality_aware_augmentation": bool(
            getattr(args, "modality_aware_augmentation", False)
        ),
        "use_group_permute_bands": bool(
            getattr(args, "use_group_permute_bands", True)
        ),
    }


def validate_augsburg_pretrain_checkpoint(checkpoint, expected_signature, path):
    """Reject stale Augsburg checkpoints after a three-view pipeline change."""
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Augsburg pretrain checkpoint has no metadata: {}. "
            "Use a new checkpoint path and retrain.".format(path)
        )
    actual_signature = checkpoint.get("pretrain_signature")
    if actual_signature != expected_signature:
        raise ValueError(
            "Augsburg pretrain checkpoint signature mismatch: {}. "
            "Expected {!r}, found {!r}. Use a new checkpoint path and retrain."
            .format(path, expected_signature, actual_signature)
        )


def configure_cuda_acceleration(args):
    """Configure optional CUDA math acceleration without changing model logic."""
    amp_dtype_name = str(getattr(args, "amp_dtype", "float16")).strip().lower()
    amp_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if amp_dtype_name not in amp_dtypes:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")

    cuda_available = DEVICE.type == "cuda"
    tf32_enabled = bool(getattr(args, "enable_tf32", False)) and cuda_available
    if tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    amp_enabled = bool(getattr(args, "use_amp", False)) and cuda_available
    amp_dtype = amp_dtypes[amp_dtype_name]
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    amp_scaler = (
        torch.amp.GradScaler("cuda") if scaler_enabled else None
    )
    print(
        "CUDA acceleration: tf32={} amp={} amp_dtype={} grad_scaler={}".format(
            tf32_enabled, amp_enabled, amp_dtype_name, scaler_enabled,
        )
    )
    return amp_enabled, amp_dtype, amp_scaler


def autocast_context(device, enabled, dtype):
    """Return an autocast context that is a no-op outside CUDA AMP runs."""
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def move_modalities(modalities, device):
    """Move a multimodal batch without blocking the host on pinned CUDA copies."""
    return [
        modality.to(device, non_blocking=(device.type == "cuda"))
        for modality in modalities
    ]


def validate_experiment_args(args):
    required = (
        "dataset", "dataset_root", "model_path", "batch_size", "image_size",
        "joint_train_epoch", "dim_emebeding", "lr_scale",
        "weight_decay", "contrastive_param", "lambda_clu", "lambda_rel_max",
    )
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise ValueError("Missing required configuration keys: {}".format(", ".join(missing)))
    if args.batch_size <= 1:
        raise ValueError("batch_size must be greater than 1 for contrastive learning")
    if args.image_size < 3:
        raise ValueError("image_size must be at least 3")
    if args.joint_train_epoch <= 0:
        raise ValueError("joint_train_epoch must be positive")
    if args.dim_emebeding <= 0 or args.lr_scale <= 0:
        raise ValueError("dim_emebeding and lr_scale must be positive")
    if args.lambda_clu < 0 or args.lambda_rel_max < 0:
        raise ValueError("lambda_clu and lambda_rel_max must be non-negative")
    if getattr(args, "use_reliability_contrastive", False) and args.lambda_rel_max > 1:
        raise ValueError(
            "lambda_rel_max is a reliability interpolation strength and must be <= 1"
        )
    if getattr(args, "reliability_graph_sigma", 0.1) <= 0:
        raise ValueError("reliability_graph_sigma must be positive")
    graph_alpha = getattr(args, "reliability_graph_alpha", 0.5)
    if not 0 <= graph_alpha <= 1:
        raise ValueError("reliability_graph_alpha must be in [0, 1]")
    if getattr(args, "reliability_graph_steps", 1) < 1:
        raise ValueError("reliability_graph_steps must be at least 1")
    if getattr(args, "reliability_graph_topk", 0) < 0:
        raise ValueError("reliability_graph_topk must be non-negative")
    rac_connection_mode = str(
        getattr(args, "rac_connection_mode", "projection")
    ).strip().lower()
    if rac_connection_mode not in {"projection", "encoder", "hybrid"}:
        raise ValueError(
            "rac_connection_mode must be one of: projection, encoder, hybrid"
        )
    rac_encoder_mix = float(getattr(args, "rac_encoder_mix", 0.0))
    if not 0.0 <= rac_encoder_mix <= 1.0:
        raise ValueError("rac_encoder_mix must be in [0, 1]")
    if int(getattr(args, "kmeans_seed", -1)) < -1:
        raise ValueError("kmeans_seed must be -1 (inherit seed) or non-negative")
    if int(getattr(args, "joint_seed", -1)) < -1:
        raise ValueError("joint_seed must be -1 (inherit seed) or non-negative")
    if int(getattr(args, "kmeans_init_sample_size", 8192)) <= 0:
        raise ValueError("kmeans_init_sample_size must be positive")
    if int(getattr(args, "cluster_repair_cooldown_epochs", 2)) < 0:
        raise ValueError("cluster_repair_cooldown_epochs must be non-negative")
    if int(getattr(args, "cluster_reseed_candidate_pool", 256)) < 2:
        raise ValueError("cluster_reseed_candidate_pool must be at least 2")
    if int(getattr(args, "eval_every_epochs", 1)) <= 0:
        raise ValueError("eval_every_epochs must be positive")
    if int(getattr(args, "dataloader_prefetch_factor", 4)) <= 0:
        raise ValueError("dataloader_prefetch_factor must be positive")
    if int(getattr(args, "initialization_batch_size", 512)) <= 0:
        raise ValueError("initialization_batch_size must be positive")
    if int(getattr(args, "auxiliary_batch_size", 512)) <= 0:
        raise ValueError("auxiliary_batch_size must be positive")
    repair_strategy = str(
        getattr(args, "cluster_repair_strategy", "legacy")
    ).strip().lower()
    if repair_strategy not in {"legacy", "split_largest"}:
        raise ValueError(
            "cluster_repair_strategy must be 'legacy' or 'split_largest'"
        )
    if getattr(args, "dec_frequency_power", 1.0) < 0:
        raise ValueError("dec_frequency_power must be non-negative")
    if getattr(args, "small_cluster_protection_weight", 0.0) < 0:
        raise ValueError("small_cluster_protection_weight must be non-negative")
    floor_factor = getattr(args, "small_cluster_floor_factor", 0.5)
    min_ratio = getattr(args, "small_cluster_min_ratio", 0.0)
    if not 0 <= floor_factor <= 1:
        raise ValueError("small_cluster_floor_factor must be between 0 and 1")
    if not 0 <= min_ratio < 1:
        raise ValueError("small_cluster_min_ratio must be in [0, 1)")
    reliability_temperature = float(
        getattr(args, "reliability_temperature", 1.0)
    )
    if reliability_temperature <= 0:
        raise ValueError("reliability_temperature must be positive")
    reliability_min_weight = float(
        getattr(args, "reliability_min_weight", 0.0)
    )
    configured_modalities = getattr(args, "modality_files", None)
    if configured_modalities:
        max_min_weight = 1.0 / len(configured_modalities)
        if not 0.0 <= reliability_min_weight < max_min_weight:
            raise ValueError(
                "reliability_min_weight must be in [0, 1 / n_modalities)"
            )
    if float(getattr(args, "init_anchor_weight_end", 0.0)) < 0:
        raise ValueError("init_anchor_weight_end must be non-negative")


def has_full_cluster_coverage(cluster_counts, min_samples=1):
    """Return whether every configured cluster is represented globally."""
    if cluster_counts is None or cluster_counts.numel() == 0:
        return False
    return bool(torch.all(cluster_counts >= max(1, int(min_samples))).item())


def compute_cluster_lambda(epoch, target_lambda, ramp_epochs=1, start_ratio=1.0):
    """Linearly ramp the DEC weight from a safe fraction to its target value."""
    target_lambda = max(0.0, float(target_lambda))
    ramp_epochs = max(1, int(ramp_epochs))
    start_ratio = min(1.0, max(0.0, float(start_ratio)))
    if ramp_epochs == 1:
        return target_lambda
    progress = min(1.0, max(0.0, (int(epoch) - 1) / float(ramp_epochs - 1)))
    return target_lambda * (start_ratio + progress * (1.0 - start_ratio))


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_dataloader(dataset_obj, batch_size, shuffle, drop_last, workers,
                     generator=None, persistent_workers=True,
                     prefetch_factor=4):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": workers,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = initialization_utils.seed_dataloader_worker
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    # Pin memory for faster CPU-to-GPU transfer
    if torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True
    return torch.utils.data.DataLoader(dataset_obj, **loader_kwargs)


def shutdown_persistent_workers(data_loader):
    """Release an already-started persistent loader before a stage rebuild."""
    iterator = getattr(data_loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
        data_loader._iterator = None


class TeeStream:
    """Write console output to both the original stream and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def setup_experiment_log(dataset_name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(dataset_name))
    log_path = os.path.abspath(os.path.join(log_dir, f"{safe_dataset}_{timestamp}.log"))
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_log():
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_log)
    print(f"Experiment log: {log_path}")
    return log_path


def pretrain(model, loss_op, train_loader, optimizer, epoch, total_epochs,
             amp_enabled=False, amp_dtype=torch.float16, amp_scaler=None,
             monitor_reliability=False):
    model.train()
    loss_epoch = torch.zeros((), device=DEVICE)
    gate_sum = None
    gate_sq_sum = None
    gate_samples = 0
    for step, ((x_1, x_2), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = move_modalities(x_1, DEVICE)
        x_list_2 = move_modalities(x_2, DEVICE)
        with autocast_context(DEVICE, amp_enabled, amp_dtype):
            z_1, z_2 = model.forward_pretrain(x_list_1, x_list_2)
            loss_ = loss_op(z_1, z_2)
        if amp_scaler is None:
            loss_.backward()
            optimizer.step()
        else:
            amp_scaler.scale(loss_).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        if monitor_reliability:
            reliability = getattr(
                model.embedding_layer, "last_modality_reliability", None
            )
            if reliability is not None:
                reliability = reliability.detach()
                batch_sum = reliability.sum(dim=0)
                batch_sq_sum = reliability.square().sum(dim=0)
                gate_sum = batch_sum if gate_sum is None else gate_sum + batch_sum
                gate_sq_sum = (
                    batch_sq_sum
                    if gate_sq_sum is None
                    else gate_sq_sum + batch_sq_sum
                )
                gate_samples += reliability.size(0)
        loss_epoch += loss_.detach()
        if step % 50 == 0:
            print(f"Pretrain [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}]\t loss: {loss_.item():.6f}")
    monitor = {}
    if gate_samples:
        gate_mean = gate_sum / gate_samples
        gate_var = gate_sq_sum / gate_samples - gate_mean.square()
        monitor["gate_mean"] = gate_mean.cpu().numpy()
        monitor["gate_std"] = gate_var.clamp_min(0).sqrt().cpu().numpy()
    return (loss_epoch / len(train_loader)).item(), monitor


@torch.no_grad()
def update_ema_model(ema_model, model, momentum):
    """Update a label-independent EMA teacher initialized at the KMeans state."""
    momentum = min(1.0, max(0.0, float(momentum)))
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        source_value = model_state[name].detach().to(
            device=ema_value.device, dtype=ema_value.dtype
        )
        if torch.is_floating_point(ema_value):
            ema_value.mul_(momentum).add_(source_value, alpha=1.0 - momentum)
        else:
            ema_value.copy_(source_value)


def train(model, loss_op, train_loader, optimizer, ema_model=None, ema_momentum=0.995,
          amp_enabled=False, amp_dtype=torch.float16, amp_scaler=None):
    model.train()
    loss_epoch = torch.zeros((), device=DEVICE)
    relation_totals = {}
    gate_sum = None
    gate_sq_sum = None
    gate_samples = 0
    active_cluster_mask = loss_op.clustering_loss.get_active_cluster_mask(
        model.clustering_head.cluster_centers.size(0),
        model.clustering_head.cluster_centers.device,
    )
    inactive_center_rows = torch.nonzero(
        ~active_cluster_mask, as_tuple=False
    ).view(-1)
    if inactive_center_rows.numel() == 0:
        inactive_center_rows = None
    for step, ((x_1, x_2), y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = move_modalities(x_1, DEVICE)
        x_list_2 = move_modalities(x_2, DEVICE)
        with autocast_context(DEVICE, amp_enabled, amp_dtype):
            h1 = h2 = None
            if getattr(loss_op, "requires_encoder_features", False):
                y1, y2, z1, z2, h1, h2 = model(
                    x_list_1, x_list_2,
                    return_features=True, return_embeddings=True,
                )
            else:
                y1, y2, z1, z2 = model(x_list_1, x_list_2, return_features=True)
            anchor_y1 = anchor_y2 = teacher_z1 = teacher_z2 = None
            if ema_model is not None:
                ema_model.eval()
                with torch.no_grad():
                    anchor_y1, anchor_y2, teacher_z1, teacher_z2 = ema_model(
                        x_list_1, x_list_2, return_features=True
                    )
            loss_, loss_ins, loss_clu = loss_op(
                y1, y2, cluster_center=model.clustering_head.cluster_centers,
                z_1=z1, z_2=z2,
                h_1=h1, h_2=h2,
                anchor_y_1=anchor_y1, anchor_y_2=anchor_y2,
                teacher_z_1=teacher_z1, teacher_z_2=teacher_z2,
            )
            loss_anchor = (
                loss_op.anchor_weight
                * loss_op.last_relation_stats.get("anchor_loss", y1.new_tensor(0.0))
            )

        if amp_scaler is None:
            loss_.backward()
        else:
            amp_scaler.scale(loss_).backward()
        center_grad = model.clustering_head.cluster_centers.grad
        inactive_center_snapshot = None
        if inactive_center_rows is not None:
            inactive_center_snapshot = (
                model.clustering_head.cluster_centers.detach()[
                    inactive_center_rows
                ].clone()
            )
        if center_grad is not None and inactive_center_rows is not None:
            center_grad[~active_cluster_mask] = 0.0
        if amp_scaler is None:
            optimizer.step()
        else:
            amp_scaler.step(optimizer)
            amp_scaler.update()
        if inactive_center_rows is not None:
            # Adam's weight decay and momentum can move a row even after its
            # data gradient is zeroed. Restore cooling centers exactly and
            # clear only their optimizer moments.
            with torch.no_grad():
                model.clustering_head.cluster_centers[inactive_center_rows].copy_(
                    inactive_center_snapshot
                )
            reset_cluster_optimizer_rows(
                optimizer,
                model.clustering_head.cluster_centers,
                inactive_center_rows.tolist(),
            )
        if ema_model is not None:
            update_ema_model(ema_model, model, ema_momentum)
        for name, value in getattr(loss_op, "last_relation_stats", {}).items():
            if torch.is_tensor(value):
                value = value.detach()
            else:
                value = loss_.new_tensor(float(value))
            relation_totals[name] = relation_totals.get(name, value.new_zeros(())) + value
        reliability = getattr(model.embedding_layer, "last_modality_reliability", None)
        if reliability is not None:
            reliability = reliability.detach()
            batch_sum = reliability.sum(dim=0)
            batch_sq_sum = reliability.square().sum(dim=0)
            gate_sum = batch_sum if gate_sum is None else gate_sum + batch_sum
            gate_sq_sum = batch_sq_sum if gate_sq_sum is None else gate_sq_sum + batch_sq_sum
            gate_samples += reliability.size(0)
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t loss: "
                  f"{loss_.item():.6f}\t"
                  f"CL:{loss_ins.item():.6f}\t"
                  f"CLU: {loss_clu.item():.6f}\t"
                  f"ANCH: {loss_anchor.item():.6f}")
        loss_epoch += loss_.detach()
    monitor = {}
    if relation_totals:
        monitor.update({
            name: (value / len(train_loader)).item()
            for name, value in relation_totals.items()
        })
    if gate_samples:
        gate_mean = gate_sum / gate_samples
        gate_var = gate_sq_sum / gate_samples - gate_mean.square()
        monitor["gate_mean"] = gate_mean.cpu().numpy()
        monitor["gate_std"] = gate_var.clamp_min(0).sqrt().cpu().numpy()
    return loss_epoch.item(), monitor


def inference(test_loader, model, device, is_labeled_pixel,
              amp_enabled=False, amp_dtype=torch.float16):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = move_modalities(x, device)
        with torch.no_grad(), autocast_context(device, amp_enabled, amp_dtype):
            pred = model.forward_cluster(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(y_pred_vector.shape))
    if is_labeled_pixel:
        y_eval, y_pred_eval = labels_vector, y_pred_vector
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]
        y_eval, y_pred_eval = y, y_pred
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(y, y_pred)
    cluster_counts = np.bincount(y_pred_eval.astype(int), minlength=len(np.unique(y_eval)))
    cluster_ratio = cluster_counts / cluster_counts.sum()
    cluster_entropy = -(cluster_ratio[cluster_ratio > 0] * np.log(cluster_ratio[cluster_ratio > 0])).sum()
    cluster_entropy /= np.log(len(cluster_counts))
    print('OA = {:.4f} AA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(
        acc, np.mean(ca), kappa, nmi, ari, pur))
    print('Cluster counts = {} ratios = {} normalized_entropy = {:.4f}'.format(
        cluster_counts.tolist(), np.round(cluster_ratio, 4).tolist(), cluster_entropy))
    return acc, kappa, nmi, ari, pur, ca


@torch.no_grad()
def compute_global_cluster_stats(model, dataset_train, device, batch_size=512, workers=2,
                                 amp_enabled=False, amp_dtype=torch.float16,
                                 prefetch_factor=4):
    """Compute unsupervised soft frequencies and hard counts over train samples."""
    model.eval()
    saved_transform = dataset_train.transform
    dataset_train.transform = None
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
    loader = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)
    global_freq = None
    hard_counts = None
    for x, y in loader:
        x_list = move_modalities(x, device)
        with autocast_context(device, amp_enabled, amp_dtype):
            h = model.encode(x_list)
            prob = model.clustering_head(h)
        freq = prob.float().sum(dim=0)
        global_freq = freq if global_freq is None else global_freq + freq
        pred = prob.argmax(dim=1)
        counts = torch.bincount(pred, minlength=prob.size(1))
        hard_counts = counts if hard_counts is None else hard_counts + counts
    dataset_train.transform = saved_transform
    model.train()
    return global_freq.cpu(), hard_counts.cpu()


@torch.no_grad()
def _sample_cluster_features(
        model, dataset_train, cluster_id, device,
        batch_size=512, workers=2, sample_size=4096, prefetch_factor=4):
    """Collect a deterministic priority sample from one predicted cluster."""
    sample_size = max(2, int(sample_size))
    sampled_features = None
    sampled_priorities = None
    generator = torch.Generator().manual_seed(1729 + int(cluster_id))
    was_training = model.training
    saved_transform = dataset_train.transform
    model.eval()
    dataset_train.transform = None
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    try:
        loader = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)
        for x, _ in loader:
            x_list = move_modalities(x, device)
            embedding = model.encode(x_list)
            prediction = model.clustering_head(embedding).argmax(dim=1)
            batch_features = embedding[prediction.eq(int(cluster_id))].detach().cpu()
            if batch_features.numel() == 0:
                continue
            batch_priorities = torch.rand(
                batch_features.size(0), generator=generator
            )
            if sampled_features is None:
                sampled_features = batch_features
                sampled_priorities = batch_priorities
            else:
                sampled_features = torch.cat(
                    [sampled_features, batch_features], dim=0
                )
                sampled_priorities = torch.cat(
                    [sampled_priorities, batch_priorities], dim=0
                )
            if sampled_priorities.numel() > sample_size:
                sampled_priorities, keep_index = torch.topk(
                    sampled_priorities, k=sample_size
                )
                sampled_features = sampled_features[keep_index]
    finally:
        dataset_train.transform = saved_transform
        model.train(was_training)
    return sampled_features


@torch.no_grad()
def _split_cluster_features(features, original_center, iterations=12):
    """Split a populated cluster into two dense subclusters with local 2-means."""
    if features is None or features.size(0) < 2:
        raise RuntimeError("At least two donor-cluster features are required")
    features = features.detach().float().cpu()
    original_center = original_center.detach().float().cpu()

    first_center = features.mean(dim=0)
    farthest_index = (
        (features - first_center).square().sum(dim=1).argmax().item()
    )
    split_centers = torch.stack([first_center, features[farthest_index]], dim=0)
    assignments = None
    for _ in range(max(1, int(iterations))):
        distance = torch.cdist(features, split_centers).square()
        assignments = distance.argmin(dim=1)
        if assignments.unique().numel() < 2:
            farthest_index = distance[:, 0].argmax().item()
            split_centers[1] = features[farthest_index]
            assignments = torch.cdist(features, split_centers).square().argmin(dim=1)
        if assignments.unique().numel() < 2:
            raise RuntimeError(
                "Donor-cluster features are degenerate and cannot be split"
            )
        updated = torch.stack([
            features[assignments.eq(cluster_id)].mean(dim=0)
            for cluster_id in range(2)
        ])
        if torch.allclose(updated, split_centers, rtol=1e-5, atol=1e-6):
            split_centers = updated
            break
        split_centers = updated

    partition_counts = torch.bincount(assignments, minlength=2)
    if torch.any(partition_counts == 0):
        raise RuntimeError("Local 2-means failed to create two populated partitions")

    # Preserve the donor identity with the child closest to its previous center.
    donor_child = int(
        (split_centers - original_center).square().sum(dim=1).argmin().item()
    )
    repaired_child = 1 - donor_child
    return (
        split_centers[donor_child],
        split_centers[repaired_child],
        partition_counts,
    )


@torch.no_grad()
def _legacy_reseed_undercovered_cluster_centers(
        model, dataset_train, cluster_counts, min_samples, device,
        batch_size=512, workers=2, candidate_pool_size=256,
        prefetch_factor=4):
    """Original farthest-feature repair retained for experiment reproduction."""
    if cluster_counts is None or cluster_counts.numel() == 0:
        return []

    min_samples = max(1, int(min_samples))
    undercovered = torch.nonzero(
        cluster_counts < min_samples, as_tuple=False
    ).view(-1)
    if undercovered.numel() == 0:
        return []

    centers = model.clustering_head.cluster_centers
    healthy = torch.nonzero(
        cluster_counts >= min_samples, as_tuple=False
    ).view(-1).to(centers.device)
    if healthy.numel() == 0:
        raise RuntimeError("Cannot reseed clusters because no healthy center remains")

    candidate_pool_size = max(int(candidate_pool_size), int(undercovered.numel()))
    reference_centers = centers.detach()[healthy]
    candidate_features = None
    candidate_scores = None
    was_training = model.training
    saved_transform = dataset_train.transform
    model.eval()
    dataset_train.transform = None
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    try:
        loader = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)
        for x, _ in loader:
            x_list = move_modalities(x, device)
            embedding = model.encode(x_list)
            min_distance = torch.cdist(
                embedding, reference_centers
            ).square().min(dim=1).values
            keep = min(candidate_pool_size, embedding.size(0))
            batch_scores, batch_index = torch.topk(min_distance, k=keep)
            batch_features = embedding[batch_index].detach().cpu()
            batch_scores = batch_scores.detach().cpu()

            if candidate_features is None:
                candidate_features = batch_features
                candidate_scores = batch_scores
            else:
                candidate_features = torch.cat(
                    [candidate_features, batch_features], dim=0
                )
                candidate_scores = torch.cat(
                    [candidate_scores, batch_scores], dim=0
                )
            if candidate_scores.numel() > candidate_pool_size:
                candidate_scores, keep_index = torch.topk(
                    candidate_scores, k=candidate_pool_size
                )
                candidate_features = candidate_features[keep_index]
    finally:
        dataset_train.transform = saved_transform
        model.train(was_training)

    if candidate_features is None or candidate_features.size(0) < undercovered.numel():
        raise RuntimeError("Not enough feature candidates to reseed under-covered clusters")

    selected_features = []
    selection_references = reference_centers.detach().cpu()
    for _ in undercovered.tolist():
        distance = torch.cdist(
            candidate_features, selection_references
        ).square().min(dim=1).values
        selected_index = int(distance.argmax().item())
        selected = candidate_features[selected_index]
        selected_features.append(selected)
        selection_references = torch.cat(
            [selection_references, selected.unsqueeze(0)], dim=0
        )

    replacement = torch.stack(selected_features).to(
        centers.device, dtype=centers.dtype
    )
    centers[undercovered.to(centers.device)] = replacement
    return undercovered.tolist()


@torch.no_grad()
def reseed_undercovered_cluster_centers(
        model, dataset_train, cluster_counts, min_samples, device,
        batch_size=512, workers=2, candidate_pool_size=4096,
        prefetch_factor=4):
    """Repair low-coverage centers by splitting the largest healthy clusters."""
    if cluster_counts is None or cluster_counts.numel() == 0:
        return {
            "repaired_clusters": [],
            "donor_clusters": [],
            "touched_clusters": [],
        }

    min_samples = max(1, int(min_samples))
    undercovered = torch.nonzero(
        cluster_counts < min_samples, as_tuple=False
    ).view(-1)
    if undercovered.numel() == 0:
        return {
            "repaired_clusters": [],
            "donor_clusters": [],
            "touched_clusters": [],
        }

    centers = model.clustering_head.cluster_centers
    effective_counts = cluster_counts.detach().cpu().clone()
    repaired_clusters = []
    donor_clusters = []
    unavailable_donors = set(undercovered.tolist())

    for repaired_cluster in undercovered.tolist():
        donor_order = torch.argsort(effective_counts, descending=True).tolist()
        donor_cluster = None
        donor_features = None
        for candidate in donor_order:
            if candidate in unavailable_donors or effective_counts[candidate] < 2:
                continue
            candidate_features = _sample_cluster_features(
                model, dataset_train, candidate, device,
                batch_size=batch_size, workers=workers,
                sample_size=candidate_pool_size,
                prefetch_factor=prefetch_factor,
            )
            if candidate_features is not None and candidate_features.size(0) >= 2:
                donor_cluster = int(candidate)
                donor_features = candidate_features
                break
        if donor_cluster is None:
            raise RuntimeError(
                "Cannot repair clusters because no splittable donor cluster remains"
            )

        donor_center, repaired_center, partition_counts = _split_cluster_features(
            donor_features, centers[donor_cluster]
        )
        centers[donor_cluster].copy_(
            donor_center.to(centers.device, dtype=centers.dtype)
        )
        centers[repaired_cluster].copy_(
            repaired_center.to(centers.device, dtype=centers.dtype)
        )
        repaired_clusters.append(int(repaired_cluster))
        donor_clusters.append(donor_cluster)
        unavailable_donors.add(donor_cluster)
        effective_counts[donor_cluster] = int(partition_counts.max().item())
        effective_counts[repaired_cluster] = int(partition_counts.min().item())

    touched_clusters = sorted(set(repaired_clusters + donor_clusters))
    return {
        "repaired_clusters": repaired_clusters,
        "donor_clusters": donor_clusters,
        "touched_clusters": touched_clusters,
    }


@torch.no_grad()
def reset_cluster_optimizer_rows(optimizer, cluster_centers, cluster_ids):
    """Clear Adam moments only for centers changed by a cluster repair."""
    if not cluster_ids:
        return
    state = optimizer.state.get(cluster_centers)
    if not state:
        return
    rows = torch.as_tensor(
        sorted(set(map(int, cluster_ids))),
        device=cluster_centers.device,
        dtype=torch.long,
    )
    for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(state_name)
        if torch.is_tensor(value) and value.shape == cluster_centers.shape:
            value.index_fill_(0, rows, 0.0)


def cluster_repair_succeeded(
        previous_counts, repaired_counts, repaired_clusters, min_samples,
        donor_clusters=None):
    """Require every repaired cluster to meet the requested hard-count floor."""
    if not repaired_clusters:
        return False
    required = max(1, int(min_samples))
    for cluster_id in repaired_clusters:
        before = int(previous_counts[int(cluster_id)].item())
        after = int(repaired_counts[int(cluster_id)].item())
        if after < required or after <= before:
            return False
    for cluster_id in donor_clusters or []:
        if int(repaired_counts[int(cluster_id)].item()) < required:
            return False
    return True


@torch.no_grad()
def repair_cluster_centers_with_rollback(
        model, dataset_train, cluster_counts, min_samples, device, optimizer,
        batch_size=512, workers=2, candidate_pool_size=4096,
        ema_model=None, repair_strategy="split_largest", prefetch_factor=4):
    """Route legacy repairs or run density-aware repair with rollback."""
    repair_strategy = str(repair_strategy).strip().lower()
    centers = model.clustering_head.cluster_centers
    if repair_strategy == "legacy":
        repaired_clusters = _legacy_reseed_undercovered_cluster_centers(
            model, dataset_train, cluster_counts, min_samples, device,
            batch_size=batch_size, workers=workers,
            candidate_pool_size=candidate_pool_size,
            prefetch_factor=prefetch_factor,
        )
        # Preserve the original optimizer behavior for exact legacy reruns.
        optimizer.state.pop(centers, None)
        repaired_freq, repaired_counts = compute_global_cluster_stats(
            model, dataset_train, device,
            batch_size=batch_size, workers=workers,
            prefetch_factor=prefetch_factor,
        )
        return (
            repaired_freq,
            repaired_counts,
            {
                "strategy": "legacy",
                "repaired_clusters": repaired_clusters,
                "donor_clusters": [],
                "touched_clusters": repaired_clusters,
            },
            True,
        )
    if repair_strategy != "split_largest":
        raise ValueError("Unknown cluster repair strategy: {}".format(
            repair_strategy
        ))

    center_snapshot = centers.detach().clone()
    previous_counts = cluster_counts.detach().cpu().clone()
    repair = reseed_undercovered_cluster_centers(
        model, dataset_train, cluster_counts, min_samples, device,
        batch_size=batch_size, workers=workers,
        candidate_pool_size=candidate_pool_size,
        prefetch_factor=prefetch_factor,
    )
    repair["strategy"] = "split_largest"
    repaired_freq, repaired_counts = compute_global_cluster_stats(
        model, dataset_train, device,
        batch_size=batch_size, workers=workers,
        prefetch_factor=prefetch_factor,
    )
    succeeded = cluster_repair_succeeded(
        previous_counts,
        repaired_counts,
        repair["repaired_clusters"],
        min_samples,
        donor_clusters=repair["donor_clusters"],
    )
    if succeeded:
        reset_cluster_optimizer_rows(
            optimizer, centers, repair["touched_clusters"]
        )
        if ema_model is not None and repair["touched_clusters"]:
            source_rows = torch.as_tensor(
                repair["touched_clusters"],
                device=centers.device,
                dtype=torch.long,
            )
            ema_centers = ema_model.clustering_head.cluster_centers
            target_rows = source_rows.to(ema_centers.device)
            ema_centers[target_rows].copy_(
                centers[source_rows].to(
                    device=ema_centers.device,
                    dtype=ema_centers.dtype,
                )
            )
        return repaired_freq, repaired_counts, repair, True

    centers.copy_(center_snapshot)
    restored_freq, restored_counts = compute_global_cluster_stats(
        model, dataset_train, device,
        batch_size=batch_size, workers=workers,
        prefetch_factor=prefetch_factor,
    )
    return restored_freq, restored_counts, repair, False


@torch.no_grad()
def compute_global_cluster_freq(model, dataset_train, device, batch_size=512, workers=2):
    """Compute global cluster soft-assignment frequencies over the full training set.

    Returns a tensor of shape [K] where entry k = Σ_i q_ik, the sum of soft
    cluster probabilities for cluster k across every training sample.

    Using these global frequencies as the DEC target-distribution denominator
    instead of the noisy per-batch sum is the key fix that prevents fine-tuning
    from degrading the KMeans initialization (the original DEC paper intent).
    """
    model.eval()
    saved_transform = dataset_train.transform
    dataset_train.transform = None          # use raw, unaugmented features
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(dataset_train, **loader_kwargs)
    global_freq = None
    for x, y in loader:
        x_list = move_modalities(x, device)
        h = model.encode(x_list)
        prob = model.clustering_head(h)      # [B, K]
        freq = prob.sum(dim=0).cpu()         # [K]
        global_freq = freq if global_freq is None else global_freq + freq
    dataset_train.transform = saved_transform
    model.train()
    return global_freq  # [K]


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="config.yaml")
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_args.config)
    config = build_effective_config(yaml_config_hook(config_args.config))
    for k, v in config.items():
        value_type = str2bool if isinstance(v, bool) else type(v)
        parser.add_argument(f"--{k}", default=v, type=value_type)
    args = parser.parse_args()
    apply_runtime_metadata(args)
    validate_experiment_args(args)
    amp_enabled, amp_dtype, amp_scaler = configure_cuda_acceleration(args)
    torch.autograd.set_detect_anomaly(getattr(args, "detect_anomaly", False))
    setup_experiment_log(args.dataset)
    pretrain_path = args.model_path + '/pretrain'
    joint_train_path = args.model_path + '/joint-train'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    if not os.path.exists(joint_train_path):
        os.makedirs(joint_train_path)
    initialization_utils.set_global_random_seed(seed=args.seed)

    root = args.dataset_root

    # prepare data
    if hasattr(args, "modality_files"):
        if not hasattr(args, "ground_truth_file"):
            raise ValueError(
                "ground_truth_file is required when modality_files is configured"
            )
        if not args.modality_files:
            raise ValueError("modality_files must contain at least one view")
        img_path = tuple(
            os.path.join(root, modality_file)
            for modality_file in args.modality_files
        )
        gt_path = os.path.join(root, args.ground_truth_file)
        missing_paths = [
            path for path in (*img_path, gt_path)
            if not os.path.isfile(path)
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Configured dataset files do not exist: {}".format(
                    ", ".join(missing_paths)
                )
            )
        print(
            "Configured {} views for {}: {}".format(
                len(img_path), args.dataset,
                ", ".join(os.path.basename(path) for path in img_path),
            )
        )
    elif args.dataset == "Houston":
        im_1, im_2 = 'data_HS_LR', 'data_MS_HR'
        gt_ = '2013Houston_gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Trento":
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'Trento-GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Augsburg":
        im_1, im_2, im_3 = 'data_HS_LR', 'data_SAR_HR', 'data_DSM'
        gt_ = 'Augsburg_gt'
        img_path = (
            root + im_1 + '.mat',
            root + im_2 + '.mat',
            root + im_3 + '.mat',
        )
    elif args.dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "MDSA":
        im_1, im_2, im_3, im_4 = (
            'MDAS-Sub1-HSI',
            'MDAS-Sub1-MSI',
            'MDAS-Sub1-SAR',
            'MDAS-Sub1-DSM',
        )
        gt_ = 'MDAS-Sub1-GT'
        img_path = (
            root + im_1 + '.mat',
            root + im_2 + '.mat',
            root + im_3 + '.mat',
            root + im_4 + '.mat',
        )
    else:
        raise NotImplementedError
    if not hasattr(args, "modality_files"):
        gt_path = root + gt_ + '.mat'
    train_transform = transform.Transforms(
        size=args.image_size,
        crop_scale=(getattr(args, "crop_scale_min", 0.7), 1.0),
        mask_pixel_prob=getattr(args, "mask_pixel_prob", 0.1),
        mask_band_prob=getattr(args, "mask_band_prob", 0.1),
        modality_aware_augmentation=getattr(
            args, "modality_aware_augmentation", False
        ),
        use_group_permute_bands=getattr(
            args, "use_group_permute_bands", True
        ),
    )
    dataset_train = dataset.MultiModalDataset(gt_path, *img_path, patch_size=(args.image_size, args.image_size),
                                              transform=train_transform,
                                              is_labeled=False,
                                              sampling_strategy=getattr(args, "train_sampling_strategy", "all"),
                                              max_samples=getattr(args, "train_max_samples", 0),
                                              sampling_seed=args.seed)
    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print(args)
    loader_prefetch_factor = int(
        getattr(args, "dataloader_prefetch_factor", 4)
    )
    initialization_batch_size = int(
        getattr(args, "initialization_batch_size", 512)
    )
    auxiliary_batch_size = int(
        getattr(args, "auxiliary_batch_size", 512)
    )
    print(
        'DataLoader memory profile: workers={} prefetch_factor={} '
        'initialization_batch_size={} auxiliary_batch_size={}'.format(
            args.workers, loader_prefetch_factor,
            initialization_batch_size, auxiliary_batch_size,
        )
    )

    deterministic_stages = bool(
        getattr(args, "deterministic_stage_seeding", False)
    )
    data_loader_train = build_dataloader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, drop_last=True, workers=args.workers,
        prefetch_factor=loader_prefetch_factor,
    )

    # # test loader
    if dataset_train.sample_indices is None and not args.is_labeled_pixel:
        dataset_test = dataset_train.with_transform(None)
        print('Evaluation dataset reuses preprocessed training tensors.')
    else:
        dataset_test = dataset.MultiModalDataset(
            gt_path, *img_path,
            patch_size=(args.image_size, args.image_size),
            transform=None, is_labeled=args.is_labeled_pixel,
            scalers=dataset_train.scalers,
        )
        print('Evaluation dataset reuses training normalization statistics.')
    data_loader_test = build_dataloader(
        dataset_test, batch_size=auxiliary_batch_size,
        shuffle=False, drop_last=False, workers=args.workers,
        prefetch_factor=loader_prefetch_factor,
    )

    # initialize model
    model = network.Net(dataset_train.n_modality, dataset_train.in_channels,
                        (args.image_size, args.image_size), 32, class_num, args.dim_emebeding,
                        use_reliability_fusion=getattr(args, "use_reliability_fusion", False),
                        projection_dim=getattr(args, "projection_dim", 128),
                        reliability_min_weight=getattr(
                            args, "reliability_min_weight", 0.0),
                        reliability_temperature=getattr(
                            args, "reliability_temperature", 1.0),
                        reliability_preserve_scale=getattr(
                            args, "reliability_preserve_scale", False),
                        reliability_uniform_init=getattr(
                            args, "reliability_uniform_init", False))
    # print(model)
    # summary(model, (args.in_channel, args.image_size, args.image_size), device='cpu')
    model = model.to(DEVICE)
    if getattr(args, "use_reliability_fusion", False):
        print('Using direct adaptive feature fusion: min_weight={:.3f} '
              'temperature={:.3f} preserve_scale={} uniform_init={}'.format(
                  float(getattr(args, "reliability_min_weight", 0.0)),
                  float(getattr(args, "reliability_temperature", 1.0)),
                  bool(getattr(args, "reliability_preserve_scale", False)),
                  bool(getattr(args, "reliability_uniform_init", False)),
              ))

    # from thop import profile
    # inputs = [torch.randn(1, 63, 7, 7).to(DEVICE), torch.randn(1, 2, 7, 7).to(DEVICE)]
    # flops, params = profile(model, (inputs, inputs))
    # print('flops: ', flops, 'params: ', params)

    # Stage 1: contrastive representation pretraining.
    pretrain_epochs = getattr(args, "pretrain_epoch", 10) if getattr(args, "is_pretrain", True) else 0
    pretrain_lr = args.pretrain_learning_rate
    joint_lr = args.joint_learning_rate
    shared_pretrain_checkpoint = str(
        getattr(args, "shared_pretrain_checkpoint", "")
    ).strip()
    pretrain_signature = build_pretrain_signature(args)
    print('Learning rates: pretrain_lr={:.6g}, joint_lr={:.6g}'.format(pretrain_lr, joint_lr))
    if shared_pretrain_checkpoint and os.path.exists(shared_pretrain_checkpoint):
        checkpoint = torch.load(shared_pretrain_checkpoint, map_location=DEVICE)
        if args.dataset == "Augsburg":
            validate_augsburg_pretrain_checkpoint(
                checkpoint, pretrain_signature, shared_pretrain_checkpoint
            )
        model.load_state_dict(checkpoint["net"] if "net" in checkpoint else checkpoint)
        print('Loaded shared pretrain checkpoint: {}'.format(shared_pretrain_checkpoint))
    elif pretrain_epochs > 0:
        print('start feature pretraining ...')
        pretrain_optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if 'clustering_head' not in n],
            lr=pretrain_lr, weight_decay=args.weight_decay
        )
        pretrain_loss = loss.PretrainLoss(args.batch_size, args.contrastive_param, DEVICE).to(DEVICE)
        for epoch in range(1, pretrain_epochs + 1):
            epoch_loss, pretrain_monitor = pretrain(
                model, pretrain_loss, data_loader_train, pretrain_optimizer,
                epoch, pretrain_epochs,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                amp_scaler=amp_scaler,
                monitor_reliability=(args.dataset == "Augsburg"),
            )
            print(f"Pretrain Epoch [{epoch}/{pretrain_epochs}]\t Loss: {epoch_loss:.6f}")
            if "gate_mean" in pretrain_monitor:
                print('Pretrain reliability monitor: mean={} std={}'.format(
                    np.round(pretrain_monitor["gate_mean"], 4).tolist(),
                    np.round(pretrain_monitor["gate_std"], 4).tolist(),
                ))
        if shared_pretrain_checkpoint:
            checkpoint_dir = os.path.dirname(shared_pretrain_checkpoint)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "net": model.state_dict(),
                "epoch": pretrain_epochs,
                "seed": args.seed,
                "pretrain_signature": pretrain_signature,
                "note": "Shared label-independent pretrain checkpoint for loss-weight grid search.",
            }, shared_pretrain_checkpoint)
            print('Saved shared pretrain checkpoint: {}'.format(shared_pretrain_checkpoint))

    # Stage 2: initialize cluster centers from unaugmented pretrained features.
    print('initializing cluster centers with MiniBatchKMeans ...')
    configured_kmeans_seed = int(getattr(args, "kmeans_seed", -1))
    kmeans_seed = args.seed if configured_kmeans_seed < 0 else configured_kmeans_seed
    configured_joint_seed = int(getattr(args, "joint_seed", -1))
    joint_seed = args.seed if configured_joint_seed < 0 else configured_joint_seed
    kmeans_generator = (
        initialization_utils.make_torch_generator(kmeans_seed)
        if deterministic_stages else None
    )
    kmeans_init_sample_size = int(
        getattr(args, "kmeans_init_sample_size", 8192)
    )
    original_train_transform = dataset_train.transform
    dataset_train.transform = None
    data_loader_init = build_dataloader(
        dataset_train, batch_size=initialization_batch_size,
        shuffle=True, drop_last=False, workers=args.workers,
        generator=kmeans_generator,
        persistent_workers=not deterministic_stages,
        prefetch_factor=loader_prefetch_factor,
    )
    try:
        centers, *_ = initialization_utils.init_centers(
            model, data_loader_init, class_num, DEVICE, False, seed=kmeans_seed,
            evaluate_labels=False, init_sample_size=kmeans_init_sample_size,
        )
    except Exception as e:
        print(f"ERROR: KMeans initialization failed: {str(e)}")
        print("This may happen if the feature space is degenerate or batch size is too small.")
        raise
    finally:
        dataset_train.transform = original_train_transform
    with torch.no_grad():
        model.clustering_head.cluster_centers.copy_(centers.to(DEVICE, dtype=model.clustering_head.cluster_centers.dtype))
    center_fingerprint = initialization_utils.tensor_sha256(
        model.clustering_head.cluster_centers
    )
    print('KMeans reproducibility: deterministic={} seed={} '
          'init_sample_size={} center_sha256={}'.format(
              deterministic_stages,
              kmeans_seed,
              kmeans_init_sample_size,
              center_fingerprint[:16],
          ))
    initial_cluster_freq, initial_cluster_counts = compute_global_cluster_stats(
        model, dataset_train, DEVICE,
        batch_size=auxiliary_batch_size, workers=args.workers,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        prefetch_factor=loader_prefetch_factor,
    )
    initial_cluster_ratio = initial_cluster_counts.float() / initial_cluster_counts.sum().clamp_min(1)
    print('Initial unsupervised hard cluster counts = {} ratios = {}'.format(
        initial_cluster_counts.tolist(),
        np.round(initial_cluster_ratio.numpy(), 4).tolist(),
    ))
    initialization_metadata = {
        'deterministic_stage_seeding': deterministic_stages,
        'kmeans_seed': int(kmeans_seed),
        'joint_seed': int(joint_seed),
        'kmeans_init_sample_size': int(kmeans_init_sample_size),
        'center_sha256': center_fingerprint,
        'initial_cluster_counts': initial_cluster_counts.tolist(),
    }
    if deterministic_stages:
        # Re-enter joint training from the same main-process and worker RNG
        # state regardless of whether pretraining ran or a checkpoint was loaded.
        shutdown_persistent_workers(data_loader_train)
        initialization_utils.set_global_random_seed(seed=joint_seed)
        data_loader_train = build_dataloader(
            dataset_train, batch_size=args.batch_size,
            shuffle=True, drop_last=True, workers=args.workers,
            generator=initialization_utils.make_torch_generator(joint_seed),
            persistent_workers=True,
            prefetch_factor=loader_prefetch_factor,
        )
        print('Joint-stage reproducibility: seed={} train_loader_rebuilt=True'.format(
            joint_seed
        ))
    # Stage 3: joint optimizer / loss.
    reliability_names = ('embedding_layer.reliability_fusion',)
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if 'clustering_head' not in n and not any(key in n for key in reliability_names)],
         'lr': joint_lr, 'weight_decay': args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(key in n for key in reliability_names)],
         'lr': joint_lr, 'weight_decay': 0.0},
        {"params": [model.clustering_head.cluster_centers],
         'lr': joint_lr * args.lr_scale, 'weight_decay': args.weight_decay}
    ]
    grouped_parameters = [group for group in grouped_parameters if len(group["params"]) > 0]
    optimizer = torch.optim.Adam(grouped_parameters, lr=joint_lr)

    # # ===== joint training ==========
    loss_op_joint = loss.JointLoss(args.batch_size,  # class_num,  #
                                   instance_temperature=args.contrastive_param,
                                   lambda_clu=args.lambda_clu,
                                   instance_weight=getattr(args, "joint_contrastive_weight", 1.0),
                                   anchor_weight=getattr(args, "init_anchor_weight", 0.0),
                                   balance_coef=getattr(args, "balance_coef", 0.0),
                                   center_orthogonal_coef=getattr(args, "center_orthogonal_coef", 0.0),
                                   small_cluster_protection_weight=getattr(
                                       args, "small_cluster_protection_weight", 0.0),
                                   small_cluster_floor_factor=getattr(
                                       args, "small_cluster_floor_factor", 0.5),
                                   small_cluster_min_ratio=getattr(
                                       args, "small_cluster_min_ratio", 0.0),
                                   device=DEVICE,
                                   use_reliability_contrastive=getattr(
                                       args, "use_reliability_contrastive",
                                       getattr(args, "use_confidence_relation", False)),
                                   lambda_rel=0.0,
                                   frequency_power=getattr(args, "dec_frequency_power", 0.5),
                                   use_high_confidence_clu=getattr(args, "use_high_confidence_clu", False),
                                   clu_reliable_ratio=getattr(args, "clu_reliable_ratio", 1.0),
                                   clu_min_confidence=getattr(args, "clu_min_confidence", 0.0),
                                   clu_min_samples_per_cluster=getattr(args, "clu_min_samples_per_cluster", 1),
                                   use_cluster_balanced_clu=getattr(
                                       args, "use_cluster_balanced_clu", False),
                                   reliability_graph_sigma=getattr(
                                       args, "reliability_graph_sigma", 0.2),
                                   reliability_graph_alpha=getattr(
                                       args, "reliability_graph_alpha", 0.5),
                                   reliability_graph_steps=getattr(
                                       args, "reliability_graph_steps", 1),
                                   reliability_graph_topk=getattr(
                                       args, "reliability_graph_topk", 16),
                                   rac_connection_mode=getattr(
                                       args, "rac_connection_mode", "projection"),
                                   rac_encoder_mix=getattr(
                                       args, "rac_encoder_mix", 0.0))
    loss_op_joint.set_reference_cluster_freq(initial_cluster_freq)
    print('Joint objective: L_total = w_ins(t) * L_RAC + lambda_clu * L_clu '
          '+ w_anchor * L_anchor')
    print('Loss weights: instance={:.6f}->{:.6f}, lambda_clu={:.6f}, '
          'graph_strength_max={:.6f}, anchor={:.6f}->{:.6f}'.format(
        getattr(args, "joint_contrastive_weight", 1.0),
        getattr(args, "joint_contrastive_weight_end", 1.0),
        args.lambda_clu,
        getattr(args, "reliability_graph_strength_max", args.lambda_rel_max),
        getattr(args, "init_anchor_weight", 0.0),
        getattr(
            args, "init_anchor_weight_end",
            getattr(args, "init_anchor_weight", 0.0),
        ),
    ))
    print('High-confidence CLU profile: dataset={} enabled={} ratio={:.2f} '
          'min_conf={:.2f} min_per_cluster={}'.format(
              getattr(args, "cluster_protection_profile", args.dataset),
              getattr(args, "use_high_confidence_clu", False),
              getattr(args, "clu_reliable_ratio", 1.0),
              getattr(args, "clu_min_confidence", 0.0),
              getattr(args, "clu_min_samples_per_cluster", 1),
          ))
    print('RAC connection: mode={} configured_encoder_mix={:.3f} '
          'effective_encoder_mix={:.3f}'.format(
              loss_op_joint.rac_connection_mode,
              float(getattr(args, "rac_encoder_mix", 0.0)),
              loss_op_joint.rac_encoder_mix,
          ))
    loss_history = []
    score_history = []
    best_oa = float('-inf')
    best_epoch = None
    best_scores = None
    best_checkpoint_path = os.path.join(joint_train_path, "best_oa_checkpoint.tar")
    eval_every_epochs = int(getattr(args, "eval_every_epochs", 1))
    print('Label evaluation cadence: every {} epoch(s), including the final epoch.'.format(
        eval_every_epochs
    ))

    anchor_weight = float(getattr(args, "init_anchor_weight", 0.0))
    anchor_weight_end = float(getattr(
        args, "init_anchor_weight_end", anchor_weight
    ))
    ema_momentum = float(getattr(args, "ema_anchor_momentum", 0.995))
    ema_model = None
    reliability_enabled = getattr(
        args, "use_reliability_contrastive",
        getattr(args, "use_confidence_relation", False)
    )
    if max(anchor_weight, anchor_weight_end) > 0 or reliability_enabled:
        ema_model = copy.deepcopy(model).to(DEVICE).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        print('Using EMA target branch: anchor_weight={:.6f}, momentum={:.6f}, '
              'reliability_graph={}'.format(anchor_weight, ema_momentum, reliability_enabled))

    # CosineAnnealingLR: smooth decay from learning_rate → 1% of learning_rate over all epochs.
    # Replaces StepLR (step=5, gamma=0.5) which decayed to 6.25% of lr by epoch 10
    # and caused unstable early-epoch performance drops.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.joint_train_epoch, eta_min=joint_lr * 0.01
    )
    lambda_rel_max = float(getattr(
        args, "reliability_graph_strength_max", args.lambda_rel_max
    ))
    if not 0.0 <= lambda_rel_max <= 1.0:
        raise ValueError("reliability_graph_strength_max must be in [0, 1]")
    print('Checkpoint policy: save final epoch and best-OA evaluation checkpoint.')
    print('start fine-tuning ...')
    start_time = time.time()
    # Note: cluster center freezing is removed as it can cause instability,
    # especially for datasets like Augsburg where hard-label collapse is a concern.
    # The global frequency update strategy and high-confidence sample selection
    # provide sufficient stability during early epochs.

    # How often to refresh the global DEC target frequencies (in epochs).
    # Will be dynamically adjusted based on epoch number (see below).
    # target_update_freq is now computed adaptively inside the epoch loop
    global_cluster_freq = None   # will be populated before epoch 1
    global_cluster_counts = None
    last_target_update_epoch = None
    cluster_freeze_epochs = max(0, int(getattr(args, "cluster_freeze_epochs", 0)))
    coverage_min_samples = max(1, int(getattr(args, "cluster_coverage_min_samples", 1)))
    # Configurations may opt in to repairing merely small clusters; truly empty
    # clusters are guarded separately and repaired by default.
    reseed_undercovered = bool(getattr(args, "reseed_undercovered_clusters", False))
    reseed_max_attempts = max(0, int(getattr(args, "cluster_reseed_max_attempts", 5)))
    reseed_candidate_pool = max(
        2, int(getattr(args, "cluster_reseed_candidate_pool", 256))
    )
    repair_strategy = str(
        getattr(args, "cluster_repair_strategy", "legacy")
    ).strip().lower()
    repair_cooldown_epochs = max(
        0, int(getattr(args, "cluster_repair_cooldown_epochs", 0))
    )
    cluster_repair_cooldown = {}
    print(
        'Cluster repair policy: strategy={} cooldown_epochs={} '
        'balanced_clu={} preserve_high_confidence_on_incomplete_coverage={}'.format(
            repair_strategy,
            repair_cooldown_epochs,
            bool(getattr(args, "use_cluster_balanced_clu", False)),
            bool(getattr(
                args,
                "preserve_high_confidence_on_incomplete_coverage",
                False,
            )),
        )
    )
    # Empty clusters are always repaired.  ``reseed_undercovered_clusters``
    # additionally lets a configuration repair clusters that are non-empty but
    # below ``cluster_coverage_min_samples``.
    force_reseed_empty = bool(getattr(args, "reseed_empty_clusters", True))
    post_epoch_cluster_guard = bool(
        getattr(args, "post_epoch_cluster_guard", False)
    )
    cluster_lambda_ramp_epochs = max(
        1, int(getattr(args, "cluster_lambda_ramp_epochs", 1))
    )
    cluster_lambda_start_ratio = float(
        getattr(args, "cluster_lambda_start_ratio", 1.0)
    )
    instance_weight_start = float(
        getattr(args, "joint_contrastive_weight", 1.0)
    )
    instance_weight_end = float(
        getattr(args, "joint_contrastive_weight_end", instance_weight_start)
    )

    for epoch in range(1, args.joint_train_epoch + 1):
        # ----- Global DEC target frequency update -----
        # Adaptive update frequency: more frequent in early epochs when clusters change rapidly
        # Early epochs (1-10): update every epoch for stability
        # Middle epochs (11-15): update every 2 epochs
        # Late epochs (16+): update every 3 epochs to reduce computation
        if getattr(args, "cluster_stats_refresh_every_epoch", False):
            target_update_freq = 1
        elif epoch <= 10:
            target_update_freq = 1
        elif epoch <= 15:
            target_update_freq = 2
        else:
            target_update_freq = 3

        # Refresh every target_update_freq epochs (first update always happens
        # at epoch 1 so the very first training step already uses global targets).
        if (last_target_update_epoch is None
                or epoch - last_target_update_epoch >= target_update_freq):
            global_cluster_freq, global_cluster_counts = compute_global_cluster_stats(
                model, dataset_train, DEVICE,
                batch_size=auxiliary_batch_size, workers=args.workers,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                prefetch_factor=loader_prefetch_factor,
            )
            print(f'[Epoch {epoch}] Updated global cluster freq: '
                  f'{global_cluster_freq.round().int().tolist()} '
                  f'(total={int(global_cluster_freq.sum())})')
            print(f'[Epoch {epoch}] Updated hard cluster counts: '
                  f'{global_cluster_counts.tolist()} '
                  f'(total={int(global_cluster_counts.sum())})')
            last_target_update_epoch = epoch

        cluster_coverage_ok = has_full_cluster_coverage(
            global_cluster_counts, coverage_min_samples
        )
        has_empty_cluster = bool(torch.any(global_cluster_counts == 0).item())
        should_reseed = (
            (not cluster_coverage_ok and reseed_undercovered)
            or (has_empty_cluster and force_reseed_empty)
        )
        repair_min_samples = coverage_min_samples if reseed_undercovered else 1
        repair_attempt = 0
        while should_reseed and repair_attempt < reseed_max_attempts:
            (global_cluster_freq, global_cluster_counts,
             repair, repair_succeeded) = repair_cluster_centers_with_rollback(
                model, dataset_train, global_cluster_counts,
                repair_min_samples, DEVICE, optimizer,
                batch_size=auxiliary_batch_size, workers=args.workers,
                candidate_pool_size=reseed_candidate_pool,
                ema_model=ema_model,
                repair_strategy=repair_strategy,
                prefetch_factor=loader_prefetch_factor,
            )
            repair_attempt += 1
            print('[Epoch {}] Cluster repair strategy={} donors={} targets={} '
                  '(attempt {}/{}, succeeded={})'.format(
                      epoch,
                      repair["strategy"],
                      repair["donor_clusters"],
                      repair["repaired_clusters"],
                      repair_attempt,
                      reseed_max_attempts,
                      repair_succeeded,
                  ))
            print(f'[Epoch {epoch}] Post-repair hard cluster counts: '
                  f'{global_cluster_counts.tolist()} '
                  f'(total={int(global_cluster_counts.sum())})')
            if not repair_succeeded:
                print('[Epoch {}] Cluster repair made no verified improvement; '
                      'centers were rolled back.'.format(epoch))
                break
            if repair["strategy"] == "split_largest" and repair_cooldown_epochs > 0:
                for cluster_id in repair["repaired_clusters"]:
                    cluster_repair_cooldown[int(cluster_id)] = max(
                        cluster_repair_cooldown.get(int(cluster_id), 0),
                        repair_cooldown_epochs,
                    )
            cluster_coverage_ok = has_full_cluster_coverage(
                global_cluster_counts, coverage_min_samples
            )
            has_empty_cluster = bool(torch.any(global_cluster_counts == 0).item())
            should_reseed = (
                (not cluster_coverage_ok and reseed_undercovered)
                or (has_empty_cluster and force_reseed_empty)
            )
        if should_reseed:
            print('[Epoch {}] WARNING: cluster coverage is still incomplete '
                  'after {} repair attempts: {}'.format(
                      epoch, repair_attempt, global_cluster_counts.tolist()
                  ))
        loss_op_joint.set_global_freq(global_cluster_freq)

        current_cluster_lambda = compute_cluster_lambda(
            epoch, args.lambda_clu,
            ramp_epochs=cluster_lambda_ramp_epochs,
            start_ratio=cluster_lambda_start_ratio,
        )
        loss_op_joint.set_lambda_clu(current_cluster_lambda)
        if args.joint_train_epoch <= 1:
            current_instance_weight = instance_weight_end
        else:
            instance_progress = (epoch - 1) / float(args.joint_train_epoch - 1)
            current_instance_weight = (
                instance_weight_start
                + instance_progress * (instance_weight_end - instance_weight_start)
            )
        loss_op_joint.set_instance_weight(current_instance_weight)
        if args.joint_train_epoch <= 1:
            current_anchor_weight = anchor_weight_end
        else:
            anchor_progress = (epoch - 1) / float(args.joint_train_epoch - 1)
            current_anchor_weight = (
                anchor_weight
                + anchor_progress * (anchor_weight_end - anchor_weight)
            )
        loss_op_joint.set_anchor_weight(current_anchor_weight)

        centers_frozen = epoch <= cluster_freeze_epochs
        model.clustering_head.cluster_centers.requires_grad_(not centers_frozen)
        print('Cluster stability guard: centers_frozen={} full_coverage={} min_samples={}'.format(
            centers_frozen, cluster_coverage_ok, coverage_min_samples
        ))
        print('Cluster lambda: {:.6f} (target={:.6f})'.format(
            current_cluster_lambda, args.lambda_clu
        ))
        print('Joint contrastive weight: {:.6f}'.format(current_instance_weight))

        warmup_epochs = getattr(args, "relation_warmup_epochs", 0)
        ramp_epochs = max(1, getattr(args, "relation_ramp_epochs", 1))
        if epoch < warmup_epochs:
            current_relation_lambda = 0.0
        else:
            ramp_progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
            current_relation_lambda = lambda_rel_max * (ramp_progress ** 1.25)
        loss_op_joint.set_lambda_rel(current_relation_lambda)

        use_high_confidence_clu = getattr(args, "use_high_confidence_clu", False)
        preserve_high_confidence = bool(getattr(
            args, "preserve_high_confidence_on_incomplete_coverage", False
        ))
        clu_target_ratio = getattr(args, "clu_reliable_ratio", 1.0)
        clu_warmup_epochs = int(getattr(args, "clu_warmup_epochs", 0))
        clu_ramp_epochs = max(1, int(getattr(args, "clu_ramp_epochs", 1)))
        active_cluster_mask = torch.ones(
            global_cluster_counts.numel(), dtype=torch.bool
        )
        cooling_clusters = sorted([
            cluster_id for cluster_id, remaining
            in cluster_repair_cooldown.items()
            if remaining > 0
        ])
        if cooling_clusters:
            active_cluster_mask[cooling_clusters] = False
        loss_op_joint.set_active_clusters(active_cluster_mask)

        if (not use_high_confidence_clu
                or epoch <= clu_warmup_epochs
                or (not cluster_coverage_ok and not preserve_high_confidence)):
            current_clu_enabled = False
            current_clu_ratio = 1.0
        else:
            clu_progress = min(1.0, (epoch - clu_warmup_epochs) / clu_ramp_epochs)
            current_clu_enabled = True
            current_clu_ratio = 1.0 + clu_progress * (clu_target_ratio - 1.0)
        loss_op_joint.set_high_confidence_clu(current_clu_enabled, current_clu_ratio)

        print('Reliability graph strength: {:.6f}'.format(current_relation_lambda))
        print('High-confidence CLU schedule: enabled={} ratio={:.4f}'.format(
                  current_clu_enabled,
                  current_clu_ratio,
              ))
        print('Cluster repair cooldown: active_clusters={} cooling_clusters={}'.format(
            int(active_cluster_mask.sum().item()),
            cooling_clusters,
        ))

        loss_epoch, monitor = train(
            model, loss_op_joint, data_loader_train, optimizer,
            ema_model=ema_model, ema_momentum=ema_momentum,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            amp_scaler=amp_scaler,
        )
        for cluster_id in list(cluster_repair_cooldown):
            remaining = cluster_repair_cooldown[cluster_id] - 1
            if remaining <= 0:
                del cluster_repair_cooldown[cluster_id]
            else:
                cluster_repair_cooldown[cluster_id] = remaining
        print(f"Epoch [{epoch}/{args.joint_train_epoch}]\t Loss: {loss_epoch / len(data_loader_train)}")
        if getattr(loss_op_joint, "uses_features", False):
            print(
                'Reliability contrastive stats: strength={:.4f} '
                'graph_mean={:.6f} offdiag_mean={:.6f} offdiag_max={:.6f} '
                'offdiag_density={:.6f} neighbors_per_anchor={:.2f} '
                'false_negative_mass={:.4f} base_loss={:.6f} rac_loss={:.6f}'.format(
                    monitor.get("reliability_graph_strength", 0.0),
                    monitor.get("reliability_graph_mean", 0.0),
                    monitor.get("reliability_graph_offdiag_mean", 0.0),
                    monitor.get("reliability_graph_offdiag_max", 0.0),
                    monitor.get("reliability_graph_offdiag_density", 0.0),
                    monitor.get("reliability_graph_neighbors_per_anchor", 0.0),
                    monitor.get("false_negative_mass_per_anchor", 0.0),
                    monitor.get("standard_instance_loss", 0.0),
                    monitor.get("reliability_contrastive_loss", 0.0),
                )
            )
            print(
                'RAC connection stats: projection_loss={:.6f} '
                'encoder_loss={:.6f} encoder_mix={:.3f}'.format(
                    monitor.get("rac_projection_loss", 0.0),
                    monitor.get("rac_encoder_loss", 0.0),
                    monitor.get("rac_encoder_mix", 0.0),
                )
            )
        if getattr(args, "use_high_confidence_clu", False):
            print(
                'High-confidence CLU stats: '
                'selected_ratio={:.4f} stable_ratio={:.4f} '
                'selected_clusters={:.1f} active_clusters={:.1f} '
                'effective_ratio={:.4f}'.format(
                    monitor.get("clu_high_confidence_selected_ratio", 0.0),
                    monitor.get("clu_high_confidence_stable_ratio", 0.0),
                    monitor.get("clu_high_confidence_selected_clusters", 0.0),
                    monitor.get("clu_active_clusters", 0.0),
                    monitor.get("clu_effective_sample_ratio", 0.0),
                )
            )
        if getattr(args, "small_cluster_protection_weight", 0.0) > 0:
            print(
                'Small-cluster protection stats: floor_loss={:.6f} '
                'protected_clusters={:.1f} min_batch_ratio={:.4f} '
                'min_floor_ratio={:.4f}'.format(
                    monitor.get("small_cluster_floor_loss", 0.0),
                    monitor.get("small_cluster_protected_count", 0.0),
                    monitor.get("small_cluster_min_batch_ratio", 0.0),
                    monitor.get("small_cluster_min_floor_ratio", 0.0),
                )
            )
        if current_anchor_weight > 0:
            print('Initialization anchor stats: anchor_weight={:.6f} '
                  'anchor_loss={:.6f} ema_momentum={:.6f}'.format(
                      current_anchor_weight,
                      monitor.get("anchor_loss", 0.0),
                      ema_momentum,
                  ))
        if "gate_mean" in monitor:
            print('Reliability monitor: mean={} std={}'.format(
                np.round(monitor["gate_mean"], 4).tolist(),
                np.round(monitor["gate_std"], 4).tolist()))
        loss_history.append(loss_epoch / len(data_loader_train))

        # A cluster can collapse during the epoch even when coverage was valid
        # before the first optimizer step.  Check again before evaluation so an
        # empty cluster is repaired immediately instead of surviving until the
        # next epoch.
        if post_epoch_cluster_guard:
            global_cluster_freq, global_cluster_counts = compute_global_cluster_stats(
                model, dataset_train, DEVICE,
                batch_size=auxiliary_batch_size, workers=args.workers,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                prefetch_factor=loader_prefetch_factor,
            )
            post_has_empty = bool(torch.any(global_cluster_counts == 0).item())
            post_repair_attempt = 0
            while (post_has_empty and force_reseed_empty
                   and post_repair_attempt < reseed_max_attempts):
                (global_cluster_freq, global_cluster_counts,
                 repair, repair_succeeded) = repair_cluster_centers_with_rollback(
                    model, dataset_train, global_cluster_counts, 1, DEVICE,
                    optimizer,
                    batch_size=auxiliary_batch_size, workers=args.workers,
                    candidate_pool_size=reseed_candidate_pool,
                    ema_model=ema_model,
                    repair_strategy=repair_strategy,
                    prefetch_factor=loader_prefetch_factor,
                )
                post_repair_attempt += 1
                print('[Epoch {}] Post-train cluster repair strategy={} '
                      'donors={} targets={} '
                      '(attempt {}/{}, succeeded={})'.format(
                          epoch,
                          repair["strategy"],
                          repair["donor_clusters"],
                          repair["repaired_clusters"],
                          post_repair_attempt,
                          reseed_max_attempts,
                          repair_succeeded,
                      ))
                print(f'[Epoch {epoch}] Post-train repaired hard cluster counts: '
                      f'{global_cluster_counts.tolist()} '
                      f'(total={int(global_cluster_counts.sum())})')
                if not repair_succeeded:
                    print('[Epoch {}] Post-train cluster repair failed validation; '
                          'centers were rolled back.'.format(epoch))
                    break
                if (repair["strategy"] == "split_largest"
                        and repair_cooldown_epochs > 0):
                    for cluster_id in repair["repaired_clusters"]:
                        cluster_repair_cooldown[int(cluster_id)] = max(
                            cluster_repair_cooldown.get(int(cluster_id), 0),
                            repair_cooldown_epochs,
                        )
                post_has_empty = bool(torch.any(global_cluster_counts == 0).item())
            loss_op_joint.set_global_freq(global_cluster_freq)
            last_target_update_epoch = epoch
            if post_has_empty:
                print('[Epoch {}] WARNING: empty clusters remain before evaluation: '
                      '{}'.format(epoch, global_cluster_counts.tolist()))

        should_evaluate = (
            epoch % eval_every_epochs == 0
            or epoch == args.joint_train_epoch
        )
        if should_evaluate:
            print('Label evaluation after epoch {}/{}:'.format(
                epoch, args.joint_train_epoch
            ))
            acc, kappa, nmi, ari, pur, ca = inference(
                data_loader_test, model, DEVICE,
                is_labeled_pixel=args.is_labeled_pixel,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            )
            current_scores = [acc, np.mean(ca), kappa, nmi, ari, pur]
            score_history.append(current_scores)
            if acc > best_oa:
                best_oa = float(acc)
                best_epoch = epoch
                best_scores = np.asarray(current_scores, dtype=float)
                if getattr(args, "save_best_oa_checkpoint", True):
                    torch.save({
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'oa': best_oa,
                        'metrics': dict(zip(
                            ["OA", "AA", "Kappa", "NMI", "ARI", "Purity"],
                            map(float, best_scores),
                        )),
                        'initialization': initialization_metadata,
                        'train_loss': float(loss_history[-1]),
                        'note': 'Best-OA evaluation checkpoint; labels are used only for model selection.',
                    }, best_checkpoint_path)
                    print('  *** New best OA={:.4f} at epoch {} - checkpoint saved ***'.format(
                        best_oa, best_epoch
                    ))
                else:
                    print('  *** New best OA={:.4f} at epoch {} ***'.format(
                        best_oa, best_epoch
                    ))
        else:
            print('Skipping label evaluation at epoch {}; next evaluation follows the {}-epoch cadence.'.format(
                epoch, eval_every_epochs
            ))
        lr_scheduler.step()
    running_time = time.time() - start_time
    final_checkpoint_state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.joint_train_epoch,
        'train_loss': float(loss_history[-1]),
        'initialization': initialization_metadata,
        'relation_stats': {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                           for k, v in monitor.items()},
        'note': 'Final unsupervised checkpoint; labels were not used for model selection.',
    }
    final_checkpoint_path = os.path.join(joint_train_path, "final_checkpoint.tar")
    torch.save(final_checkpoint_state, final_checkpoint_path)
    print('Saved final checkpoint: epoch={} train_loss={:.6f}'.format(
        final_checkpoint_state['epoch'], final_checkpoint_state.get('train_loss', 0.0)))
    if best_epoch is not None and getattr(args, "save_best_oa_checkpoint", True):
        print('Saved best-OA checkpoint: epoch={} OA={:.4f} -> {}'.format(
            best_epoch, best_oa, best_checkpoint_path
        ))
    print(f'fine tuning time: {running_time:.3f} s')
    print(loss_history)

    epoch_scores = np.asarray(score_history, dtype=float)
    mean_scores = epoch_scores.mean(axis=0)
    std_scores = epoch_scores.std(axis=0)
    metric_names = ["OA", "AA", "Kappa", "NMI", "ARI", "Purity"]
    print("=" * 72)
    print("LABEL-METRIC SUMMARY OVER {} EPOCHS".format(len(epoch_scores)))
    print("Best epoch result (selected by OA): epoch={}".format(best_epoch))
    for name, value in zip(metric_names, best_scores):
        print("  {:<7s} {:.4f}".format(name, value))
    print("Average result over all evaluated epochs:")
    for name, mean_value, std_value in zip(metric_names, mean_scores, std_scores):
        print("  {:<7s} mean={:.4f} std={:.4f}".format(name, mean_value, std_value))
    print("Average training loss: {:.6f}".format(float(np.mean(loss_history))))
    print("Saved checkpoint: {}".format(final_checkpoint_path))
    print("=" * 72)
