# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


# ============================================================
# Local imports
# ============================================================

from convnextv2 import ConvNeXtV2
from ema_improved import EMA
from fpn import FPN
from pfnm import PFNM
from fcos_head import FCOSHead

from dataset import (
    ConnectorDataset,
)

from loss import (
    FCOSLoss,
)


# ============================================================
# Global configuration
# ============================================================

CLASS_NAMES = [
    "C1-2",
    "C1-4",
    "C1-6",
    "C2-2",
    "C2-4",
    "C2-6",
    "C3-2",
    "C3-4",
    "C3-6",
    "C4-2",
    "C4-3",
    "C4-6",
]


NUM_CLASSES = 12

IMAGE_SIZE = 640

BATCH_SIZE = 8

NUM_WORKERS = 4

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 5

MIN_LR = 1e-6

GRAD_CLIP_NORM = 10.0

SEED = 42

SAVE_INTERVAL = 1

EARLY_STOPPING_PATIENCE = 20

USE_AMP = True


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int = SEED,
):
    """
    设置随机种子。
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


# ============================================================
# Device
# ============================================================

def get_device(
    device_name: Optional[str] = None,
):
    """
    获取训练设备。
    """

    if device_name is not None:

        device_name = (
            device_name.lower()
        )

        if device_name == "cuda":

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "指定 CUDA，但当前环境没有可用 GPU。"
                )

            return torch.device(
                "cuda"
            )

        return torch.device(
            device_name
        )

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# Process information
# ============================================================

def build_default_process_info(
    batch_size: int,
    device: torch.device,
):
    """
    构建默认工艺信息。

    PFNM 所使用的信息包括：

        assembly stage
        process parameter range
        contact requirement
        environmental condition

    如果 dataset.py 已经返回工艺信息，
    train.py 会优先使用 dataset 中的数据。

    此处只是提供统一的 fallback。
    """

    return None


# ============================================================
# Batch parsing
# ============================================================

def move_tensor_to_device(
    value,
    device: torch.device,
):
    """
    将 Tensor 或嵌套结构移动到 GPU。
    """

    if torch.is_tensor(value):

        return value.to(
            device,
            non_blocking=True,
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            key: move_tensor_to_device(
                item,
                device,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            move_tensor_to_device(
                item,
                device,
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return tuple(
            move_tensor_to_device(
                item,
                device,
            )
            for item in value
        )

    return value


def parse_batch(
    batch,
    device: torch.device,
):
    """
    兼容不同 Dataset 返回形式。

    支持：

        {
            "image": ...,
            "boxes": ...,
            "labels": ...
        }

    或：

        {
            "images": ...,
            "targets": ...
        }

    或：

        (images, targets)

    或：

        (images, boxes, labels)
    """

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        batch,
        dict,
    ):

        images = None

        for key in (
            "image",
            "images",
            "img",
            "imgs",
        ):

            if key in batch:

                images = batch[key]

                break

        if images is None:

            raise KeyError(
                "Dataset batch 中没有找到 image/images。"
            )

        targets = None

        for key in (
            "targets",
            "target",
            "annotations",
            "annotation",
        ):

            if key in batch:

                targets = batch[key]

                break

        process_info = None

        for key in (
            "process_info",
            "process_infos",
            "factory_info",
        ):

            if key in batch:

                process_info = batch[key]

                break

        # 如果没有 targets，
        # 尝试直接使用 boxes + labels。
        if targets is None:

            boxes = batch.get(
                "boxes",
                None,
            )

            labels = batch.get(
                "labels",
                None,
            )

            if boxes is not None or labels is not None:

                targets = {
                    "boxes": boxes,
                    "labels": labels,
                }

        return (
            move_tensor_to_device(
                images,
                device,
            ),
            move_tensor_to_device(
                targets,
                device,
            ),
            move_tensor_to_device(
                process_info,
                device,
            ),
        )

    # --------------------------------------------------------
    # Tuple / list
    # --------------------------------------------------------

    if isinstance(
        batch,
        (
            tuple,
            list,
        ),
    ):

        if len(batch) == 2:

            images = batch[0]

            targets = batch[1]

            return (
                move_tensor_to_device(
                    images,
                    device,
                ),
                move_tensor_to_device(
                    targets,
                    device,
                ),
                None,
            )

        if len(batch) == 3:

            images = batch[0]

            boxes = batch[1]

            labels = batch[2]

            targets = {
                "boxes": boxes,
                "labels": labels,
            }

            return (
                move_tensor_to_device(
                    images,
                    device,
                ),
                move_tensor_to_device(
                    targets,
                    device,
                ),
                None,
            )

    raise TypeError(
        "无法识别 Dataset 返回的数据结构："
        f"{type(batch)}"
    )


# ============================================================
# Model
# ============================================================

class ProcessAwareFCOS(
    nn.Module
):
    """
    第二章完整检测网络。

    Backbone:
        ConvNeXt V2

    Attention:
        EMA

    Neck:
        SimAM-FPN

    Process module:
        PFNM

    Detection head:
        FCOS
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        self.backbone = (
            ConvNeXtV2()
        )

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        self.ema = EMA()

        # ----------------------------------------------------
        # FPN
        # ----------------------------------------------------

        self.fpn = FPN()

        # ----------------------------------------------------
        # PFNM
        # ----------------------------------------------------

        self.pfnm = PFNM()

        # ----------------------------------------------------
        # FCOS head
        # ----------------------------------------------------

        self.head = FCOSHead(
            num_classes=num_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        process_info=None,
    ):
        """
        前向传播。
        """

        # ====================================================
        # ConvNeXt V2
        # ====================================================

        features = (
            self.backbone(
                x
            )
        )

        # ====================================================
        # EMA
        # ====================================================

        features = (
            self.ema(
                features
            )
        )

        # ====================================================
        # SimAM-FPN
        # ====================================================

        pyramid_features = (
            self.fpn(
                features
            )
        )

        # ====================================================
        # PFNM
        # ====================================================

        if process_info is not None:

            try:

                pyramid_features = (
                    self.pfnm(
                        pyramid_features,
                        process_info,
                    )
                )

            except TypeError:

                pyramid_features = (
                    self.pfnm(
                        pyramid_features
                    )
                )

        else:

            pyramid_features = (
                self.pfnm(
                    pyramid_features
                )
            )

        # ====================================================
        # FCOS head
        # ====================================================

        predictions = (
            self.head(
                pyramid_features
            )
        )

        return predictions


# ============================================================
# Model information
# ============================================================

def count_parameters(
    model: nn.Module,
):
    """
    统计参数量。
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


def print_model_summary(
    model: nn.Module,
):
    """
    打印模型摘要。
    """

    total, trainable = (
        count_parameters(
            model
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Model Summary"
    )

    print(
        "=" * 70
    )

    print(
        f"Total parameters     : "
        f"{total / 1e6:.2f} M"
    )

    print(
        f"Trainable parameters : "
        f"{trainable / 1e6:.2f} M"
    )

    print(
        "=" * 70
    )


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
):
    """
    构建 AdamW。
    """

    decay_parameters = []

    no_decay_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:

            continue

        if (
            parameter.ndim <= 1
            or name.endswith(
                ".bias"
            )
            or "norm" in name.lower()
        ):

            no_decay_parameters.append(
                parameter
            )

        else:

            decay_parameters.append(
                parameter
            )

    optimizer = AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-8,
    )

    return optimizer


# ============================================================
# Learning rate
# ============================================================

def build_scheduler(
    optimizer,
    epochs: int,
    min_lr: float = MIN_LR,
):
    """
    Cosine Annealing 学习率。
    """

    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
    )

    return scheduler


# ============================================================
# Warmup
# ============================================================

def adjust_learning_rate_warmup(
    optimizer,
    base_lr: float,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    warmup_epochs: int,
):
    """
    线性 warmup。
    """

    if warmup_epochs <= 0:

        return

    current_step = (
        epoch
        * steps_per_epoch
        + step
        + 1
    )

    warmup_steps = (
        warmup_epochs
        * steps_per_epoch
    )

    if current_step > warmup_steps:

        return

    warmup_ratio = (
        current_step
        / max(
            warmup_steps,
            1,
        )
    )

    warmup_lr = (
        base_lr
        * warmup_ratio
    )

    for param_group in (
        optimizer.param_groups
    ):

        param_group[
            "lr"
        ] = warmup_lr


# ============================================================
# Loss handling
# ============================================================

def build_loss_function():
    """
    构建 FCOS loss。

    当前优先使用：

        FCOSLoss()

    如果 loss.py 中使用 num_classes 参数，
    则自动尝试对应接口。
    """

    constructors = [
        lambda: FCOSLoss(),
        lambda: FCOSLoss(
            num_classes=NUM_CLASSES
        ),
        lambda: FCOSLoss(
            num_classes=NUM_CLASSES,
            strides=(
                8,
                16,
                32,
                64,
            ),
        ),
    ]

    last_error = None

    for constructor in constructors:

        try:

            return constructor()

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        "无法创建 FCOSLoss。"
        f"\n最后错误：{last_error}"
    )


def parse_loss_output(
    loss_output,
):
    """
    统一 loss.py 返回格式。

    支持：

        total_loss

    或：

        {
            "loss": ...,
            "cls_loss": ...,
            "reg_loss": ...,
            "centerness_loss": ...
        }

    或：

        (
            total_loss,
            cls_loss,
            reg_loss,
            centerness_loss,
        )
    """

    if torch.is_tensor(
        loss_output
    ):

        return (
            loss_output,
            {
                "loss": loss_output,
            },
        )

    if isinstance(
        loss_output,
        dict,
    ):

        total_loss = None

        for key in (
            "loss",
            "total_loss",
            "total",
        ):

            if key in loss_output:

                total_loss = (
                    loss_output[key]
                )

                break

        if total_loss is None:

            raise KeyError(
                "loss.py 返回 dict，"
                "但没有 loss/total_loss。"
            )

        return (
            total_loss,
            loss_output,
        )

    if isinstance(
        loss_output,
        (
            tuple,
            list,
        ),
    ):

        if len(
            loss_output
        ) == 0:

            raise ValueError(
                "loss.py 返回空 tuple/list。"
            )

        total_loss = (
            loss_output[0]
        )

        loss_dict = {
            "loss": total_loss,
        }

        if len(
            loss_output
        ) > 1:

            loss_dict[
                "cls_loss"
            ] = loss_output[1]

        if len(
            loss_output
        ) > 2:

            loss_dict[
                "reg_loss"
            ] = loss_output[2]

        if len(
            loss_output
        ) > 3:

            loss_dict[
                "centerness_loss"
            ] = loss_output[3]

        return (
            total_loss,
            loss_dict,
        )

    raise TypeError(
        "无法识别 loss 输出类型："
        f"{type(loss_output)}"
    )


# ============================================================
# Loss forward
# ============================================================

def calculate_loss(
    criterion,
    predictions,
    targets,
):
    """
    调用 loss.py。

    优先：

        criterion(predictions, targets)

    如果接口不同，
    自动尝试常见形式。
    """

    attempts = [
        lambda: criterion(
            predictions,
            targets,
        ),
        lambda: criterion(
            predictions=predictions,
            targets=targets,
        ),
        lambda: criterion(
            outputs=predictions,
            targets=targets,
        ),
    ]

    last_error = None

    for attempt in attempts:

        try:

            result = attempt()

            return parse_loss_output(
                result
            )

        except (
            TypeError,
            AttributeError,
            KeyError,
        ) as exc:

            last_error = exc

    raise RuntimeError(
        "FCOSLoss 调用失败。"
        f"\n最后错误：{last_error}"
    )


# ============================================================
# Loss statistics
# ============================================================

class AverageMeter:
    """
    平均值统计器。
    """

    def __init__(self):
        self.reset()

    def reset(self):

        self.sum = 0.0

        self.count = 0

    @property
    def avg(self):

        if self.count == 0:

            return 0.0

        return (
            self.sum
            / self.count
        )

    def update(
        self,
        value: float,
        n: int = 1,
    ):

        self.sum += (
            float(value)
            * n
        )

        self.count += n


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    grad_clip_norm: float,
    use_amp: bool,
):
    """
    单个 epoch 训练。
    """

    model.train()

    total_meter = (
        AverageMeter()
    )

    cls_meter = (
        AverageMeter()
    )

    reg_meter = (
        AverageMeter()
    )

    centerness_meter = (
        AverageMeter()
    )

    start_time = (
        time.time()
    )

    steps_per_epoch = max(
        len(loader),
        1,
    )

    for step, batch in enumerate(
        loader
    ):

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        adjust_learning_rate_warmup(
            optimizer=optimizer,
            base_lr=base_lr,
            epoch=epoch,
            step=step,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
        )

        # ----------------------------------------------------
        # Parse batch
        # ----------------------------------------------------

        images, targets, process_info = (
            parse_batch(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with autocast(
            device_type=device.type,
            enabled=(
                use_amp
                and device.type
                == "cuda"
            ),
        ):

            predictions = (
                model(
                    images,
                    process_info=process_info,
                )
            )

            (
                total_loss,
                loss_dict,
            ) = calculate_loss(
                criterion,
                predictions,
                targets,
            )

        # ----------------------------------------------------
        # Loss check
        # ----------------------------------------------------

        if not torch.isfinite(
            total_loss
        ):

            print(
                "\nWARNING: "
                "non-finite loss detected."
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            continue

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if (
            use_amp
            and device.type
            == "cuda"
        ):

            scaler.scale(
                total_loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            if grad_clip_norm > 0:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            total_loss.backward()

            if grad_clip_norm > 0:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            images.shape[0]
            if torch.is_tensor(
                images
            )
            else 1
        )

        total_meter.update(
            total_loss.detach().item(),
            batch_size,
        )

        if "cls_loss" in loss_dict:

            value = (
                loss_dict[
                    "cls_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            cls_meter.update(
                value,
                batch_size,
            )

        if "reg_loss" in loss_dict:

            value = (
                loss_dict[
                    "reg_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            reg_meter.update(
                value,
                batch_size,
            )

        if "centerness_loss" in loss_dict:

            value = (
                loss_dict[
                    "centerness_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            centerness_meter.update(
                value,
                batch_size,
            )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        if (
            step == 0
            or (step + 1) % 20 == 0
            or step + 1
            == len(loader)
        ):

            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            print(
                f"\r"
                f"Epoch "
                f"[{epoch + 1:03d}/"
                f"{total_epochs:03d}] "
                f"Step "
                f"[{step + 1:04d}/"
                f"{len(loader):04d}] "
                f"Loss "
                f"{total_meter.avg:.4f} "
                f"LR "
                f"{current_lr:.2e}",
                end="",
            )

    elapsed = (
        time.time()
        - start_time
    )

    print()

    return {
        "loss": total_meter.avg,
        "cls_loss": cls_meter.avg,
        "reg_loss": reg_meter.avg,
        "centerness_loss": centerness_meter.avg,
        "time": elapsed,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_amp: bool,
):
    """
    单个 epoch 验证。
    """

    model.eval()

    total_meter = (
        AverageMeter()
    )

    cls_meter = (
        AverageMeter()
    )

    reg_meter = (
        AverageMeter()
    )

    centerness_meter = (
        AverageMeter()
    )

    start_time = (
        time.time()
    )

    for batch in loader:

        images, targets, process_info = (
            parse_batch(
                batch,
                device,
            )
        )

        with autocast(
            device_type=device.type,
            enabled=(
                use_amp
                and device.type
                == "cuda"
            ),
        ):

            predictions = (
                model(
                    images,
                    process_info=process_info,
                )
            )

            (
                total_loss,
                loss_dict,
            ) = calculate_loss(
                criterion,
                predictions,
                targets,
            )

        batch_size = (
            images.shape[0]
            if torch.is_tensor(
                images
            )
            else 1
        )

        total_meter.update(
            total_loss.detach().item(),
            batch_size,
        )

        if "cls_loss" in loss_dict:

            value = (
                loss_dict[
                    "cls_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            cls_meter.update(
                value,
                batch_size,
            )

        if "reg_loss" in loss_dict:

            value = (
                loss_dict[
                    "reg_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            reg_meter.update(
                value,
                batch_size,
            )

        if "centerness_loss" in loss_dict:

            value = (
                loss_dict[
                    "centerness_loss"
                ]
            )

            if torch.is_tensor(
                value
            ):

                value = (
                    value.detach().item()
                )

            centerness_meter.update(
                value,
                batch_size,
            )

    elapsed = (
        time.time()
        - start_time
    )

    return {
        "loss": total_meter.avg,
        "cls_loss": cls_meter.avg,
        "reg_loss": reg_meter.avg,
        "centerness_loss": centerness_meter.avg,
        "time": elapsed,
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    history: Dict[str, list],
):
    """
    保存完整训练状态。
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "best_metric": best_metric,
        "history": history,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }

    torch.save(
        checkpoint,
        str(path),
    )


# ============================================================
# Resume
# ============================================================

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device=None,
):
    """
    恢复训练。
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=False,
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=False,
        )

    else:

        model.load_state_dict(
            checkpoint,
            strict=False,
        )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        )
        is not None
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state_dict"
        )
        is not None
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    best_metric = float(
        checkpoint.get(
            "best_metric",
            float("inf"),
        )
    )

    history = checkpoint.get(
        "history",
        {},
    )

    return (
        epoch,
        best_metric,
        history,
    )


# ============================================================
# Dataset builder
# ============================================================

def build_dataset(
    root: str,
    split: str,
    image_size: int,
):
    """
    创建 ConnectorDataset。

    尝试兼容常见 Dataset 构造方式。
    """

    constructors = [
        lambda: ConnectorDataset(
            root=root,
            split=split,
            image_size=image_size,
        ),
        lambda: ConnectorDataset(
            root=root,
            split=split,
            img_size=image_size,
        ),
        lambda: ConnectorDataset(
            root=root,
            train=(
                split == "train"
            ),
            image_size=image_size,
        ),
        lambda: ConnectorDataset(
            root=root,
            train=(
                split == "train"
            ),
            img_size=image_size,
        ),
        lambda: ConnectorDataset(
            root=root,
            split=split,
        ),
    ]

    last_error = None

    for constructor in constructors:

        try:

            return constructor()

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"无法创建 {split} Dataset。"
        f"\n最后错误：{last_error}"
    )


# ============================================================
# DataLoader
# ============================================================

def build_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    """
    创建训练集和验证集 DataLoader。
    """

    train_dataset = build_dataset(
        root=data_root,
        split="train",
        image_size=image_size,
    )

    val_dataset = build_dataset(
        root=data_root,
        split="val",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        ),
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ============================================================
# History
# ============================================================

def create_history():
    """
    创建训练历史。
    """

    return {
        "epoch": [],
        "train_loss": [],
        "train_cls_loss": [],
        "train_reg_loss": [],
        "train_centerness_loss": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_reg_loss": [],
        "val_centerness_loss": [],
        "learning_rate": [],
        "train_time": [],
        "val_time": [],
    }


def append_history(
    history,
    epoch,
    train_metrics,
    val_metrics,
    learning_rate,
):
    """
    添加 epoch 记录。
    """

    history[
        "epoch"
    ].append(
        epoch
    )

    history[
        "train_loss"
    ].append(
        train_metrics[
            "loss"
        ]
    )

    history[
        "train_cls_loss"
    ].append(
        train_metrics[
            "cls_loss"
        ]
    )

    history[
        "train_reg_loss"
    ].append(
        train_metrics[
            "reg_loss"
        ]
    )

    history[
        "train_centerness_loss"
    ].append(
        train_metrics[
            "centerness_loss"
        ]
    )

    history[
        "val_loss"
    ].append(
        val_metrics[
            "loss"
        ]
    )

    history[
        "val_cls_loss"
    ].append(
        val_metrics[
            "cls_loss"
        ]
    )

    history[
        "val_reg_loss"
    ].append(
        val_metrics[
            "reg_loss"
        ]
    )

    history[
        "val_centerness_loss"
    ].append(
        val_metrics[
            "centerness_loss"
        ]
    )

    history[
        "learning_rate"
    ].append(
        learning_rate
    )

    history[
        "train_time"
    ].append(
        train_metrics[
            "time"
        ]
    )

    history[
        "val_time"
    ].append(
        val_metrics[
            "time"
        ]
    )


# ============================================================
# Save history
# ============================================================

def save_history(
    history,
    output_dir: str,
):
    """
    保存 JSON。
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_dir
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return history_path


# ============================================================
# Plot
# ============================================================

def save_training_curves(
    history,
    output_dir: str,
):
    """
    保存训练曲线。

    如果 matplotlib 不可用，
    不影响训练。
    """

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib 未安装，"
            "跳过训练曲线保存。"
        )

        return

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history[
        "epoch"
    ]

    if not epochs:

        return

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        epochs,
        history[
            "train_loss"
        ],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history[
            "val_loss"
        ],
        label="Validation Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "FCOS Training Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "loss_curve.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Component loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        epochs,
        history[
            "train_cls_loss"
        ],
        label="Classification",
    )

    plt.plot(
        epochs,
        history[
            "train_reg_loss"
        ],
        label="Regression",
    )

    plt.plot(
        epochs,
        history[
            "train_centerness_loss"
        ],
        label="Centerness",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "FCOS Loss Components"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "loss_components.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        epochs,
        history[
            "learning_rate"
        ],
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Learning Rate"
    )

    plt.title(
        "Learning Rate Schedule"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "learning_rate.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Main training function
# ============================================================

def train(
    data_root: str,
    output_dir: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_epochs: int = WARMUP_EPOCHS,
    num_workers: int = NUM_WORKERS,
    device_name: Optional[str] = None,
    resume: Optional[str] = None,
    use_amp: bool = USE_AMP,
):
    """
    完整训练流程。
    """

    # ========================================================
    # Seed
    # ========================================================

    seed_everything(
        SEED
    )

    # ========================================================
    # Device
    # ========================================================

    device = get_device(
        device_name
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Print configuration
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Process-Aware FCOS Training"
    )

    print(
        "=" * 80
    )

    print(
        f"Data root       : {data_root}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Image size      : "
        f"{image_size} × {image_size}"
    )

    print(
        f"Batch size      : {batch_size}"
    )

    print(
        f"Epochs          : {epochs}"
    )

    print(
        f"Learning rate   : {learning_rate}"
    )

    print(
        f"Weight decay    : {weight_decay}"
    )

    print(
        f"Warmup epochs   : {warmup_epochs}"
    )

    print(
        f"AMP             : {use_amp}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nBuilding datasets..."
    )

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(
        f"Training samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_dataset)}"
    )

    print(
        f"Training batches   : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(val_loader)}"
    )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nBuilding model..."
    )

    model = ProcessAwareFCOS(
        num_classes=NUM_CLASSES
    )

    model = model.to(
        device
    )

    print_model_summary(
        model
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = (
        build_loss_function()
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        min_lr=MIN_LR,
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = GradScaler(
        enabled=(
            use_amp
            and device.type
            == "cuda"
        )
    )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 0

    best_metric = float(
        "inf"
    )

    history = create_history()

    if resume is not None:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Resume training"
        )

        print(
            "=" * 70
        )

        (
            previous_epoch,
            best_metric,
            loaded_history,
        ) = load_checkpoint(
            path=resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        start_epoch = (
            previous_epoch + 1
        )

        if loaded_history:

            history = (
                loaded_history
            )

        print(
            f"Resume from epoch "
            f"{previous_epoch + 1}"
        )

    # ========================================================
    # Training
    # ========================================================

    epochs_without_improvement = 0

    for epoch in range(
        start_epoch,
        epochs,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            "=" * 80
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                total_epochs=epochs,
                base_lr=learning_rate,
                warmup_epochs=warmup_epochs,
                grad_clip_norm=GRAD_CLIP_NORM,
                use_amp=use_amp,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = (
            validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if (
            epoch + 1
            > warmup_epochs
        ):

            scheduler.step()

        # ----------------------------------------------------
        # LR
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        append_history(
            history=history,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=current_lr,
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Train loss       : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train cls loss   : "
            f"{train_metrics['cls_loss']:.6f}"
        )

        print(
            f"Train reg loss   : "
            f"{train_metrics['reg_loss']:.6f}"
        )

        print(
            f"Train center loss: "
            f"{train_metrics['centerness_loss']:.6f}"
        )

        print(
            f"Val loss         : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val cls loss     : "
            f"{val_metrics['cls_loss']:.6f}"
        )

        print(
            f"Val reg loss     : "
            f"{val_metrics['reg_loss']:.6f}"
        )

        print(
            f"Val center loss  : "
            f"{val_metrics['centerness_loss']:.6f}"
        )

        print(
            f"Learning rate    : "
            f"{current_lr:.3e}"
        )

        print(
            f"Train time       : "
            f"{train_metrics['time']:.2f} s"
        )

        print(
            f"Val time         : "
            f"{val_metrics['time']:.2f} s"
        )

        print(
            "-" * 80
        )

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        latest_path = (
            checkpoint_dir
            / "latest.pth"
        )

        save_checkpoint(
            path=str(
                latest_path
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            history=history,
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        current_metric = (
            val_metrics[
                "loss"
            ]
        )

        if current_metric < best_metric:

            best_metric = (
                current_metric
            )

            epochs_without_improvement = 0

            best_path = (
                checkpoint_dir
                / "best.pth"
            )

            save_checkpoint(
                path=str(
                    best_path
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                history=history,
            )

            print(
                "\n"
                "Best model updated."
            )

            print(
                f"Best validation loss: "
                f"{best_metric:.6f}"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            (epoch + 1)
            % SAVE_INTERVAL
            == 0
        ):

            epoch_path = (
                checkpoint_dir
                / (
                    f"epoch_"
                    f"{epoch + 1:03d}.pth"
                )
            )

            save_checkpoint(
                path=str(
                    epoch_path
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                history=history,
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        save_history(
            history,
            str(
                output_dir
            ),
        )

        save_training_curves(
            history,
            str(
                output_dir
            ),
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            EARLY_STOPPING_PATIENCE > 0
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\n"
                + "=" * 80
            )

            print(
                "Early stopping."
            )

            print(
                f"No improvement for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            print(
                "=" * 80
            )

            break

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Training finished."
    )

    print(
        f"Best validation loss: "
        f"{best_metric:.6f}"
    )

    print(
        f"Best checkpoint:"
        f"\n{checkpoint_dir / 'best.pth'}"
    )

    print(
        f"Latest checkpoint:"
        f"\n{checkpoint_dir / 'latest.pth'}"
    )

    print(
        "=" * 80
    )


# ============================================================
# Argument parser
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Train Process-Aware FCOS "
            "connector detector"
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "数据集根目录"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/fcos_improved",
        help=(
            "训练输出目录"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "训练 epoch 数"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "batch size"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help=(
            "输入图像尺寸"
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=(
            "初始学习率"
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=(
            "weight decay"
        ),
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help=(
            "warmup epoch"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=(
            "DataLoader workers"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "cuda / cpu"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "继续训练的 checkpoint"
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "关闭 AMP"
        ),
    )

    return parser


# ============================================================
# Entry
# ============================================================

def main():

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.workers,
        device_name=args.device,
        resume=args.resume,
        use_amp=(
            not args.no_amp
        ),
    )


if __name__ == "__main__":

    main()