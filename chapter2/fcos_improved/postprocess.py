# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Default configuration
# ============================================================

DEFAULT_STRIDES = (
    8,
    16,
    32,
    64,
)

DEFAULT_NUM_CLASSES = 12

DEFAULT_SCORE_THRESHOLD = 0.05

DEFAULT_NMS_THRESHOLD = 0.60

DEFAULT_TOP_K = 1000

DEFAULT_MAX_DETECTIONS = 100


DEFAULT_CLASS_NAMES = [
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


# ============================================================
# Basic box utilities
# ============================================================

def clip_boxes(
    boxes: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """
    将 bounding boxes 限制在图像范围内。

    输入：

        boxes:
            [N, 4]

    输出：

        boxes:
            [N, 4]
    """

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    boxes = boxes.clone()

    boxes[:, 0] = boxes[:, 0].clamp(
        min=0.0,
        max=float(image_width),
    )

    boxes[:, 1] = boxes[:, 1].clamp(
        min=0.0,
        max=float(image_height),
    )

    boxes[:, 2] = boxes[:, 2].clamp(
        min=0.0,
        max=float(image_width),
    )

    boxes[:, 3] = boxes[:, 3].clamp(
        min=0.0,
        max=float(image_height),
    )

    return boxes


def remove_invalid_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_box_size: float = 1.0,
):
    """
    删除宽度或高度过小的 box。
    """

    if boxes.numel() == 0:
        return (
            boxes.reshape(-1, 4),
            scores.reshape(-1),
            labels.reshape(-1),
        )

    widths = (
        boxes[:, 2]
        - boxes[:, 0]
    )

    heights = (
        boxes[:, 3]
        - boxes[:, 1]
    )

    keep = (
        (widths >= min_box_size)
        & (heights >= min_box_size)
    )

    return (
        boxes[keep],
        scores[keep],
        labels[keep],
    )


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    计算 IoU。

    boxes1:
        [N,4]

    boxes2:
        [M,4]

    return:
        [N,M]
    """

    if (
        boxes1.numel() == 0
        or boxes2.numel() == 0
    ):
        return torch.zeros(
            (
                boxes1.shape[0],
                boxes2.shape[0],
            ),
            dtype=boxes1.dtype,
            device=boxes1.device,
        )

    area1 = (
        (
            boxes1[:, 2]
            - boxes1[:, 0]
        ).clamp(min=0)
        *
        (
            boxes1[:, 3]
            - boxes1[:, 1]
        ).clamp(min=0)
    )

    area2 = (
        (
            boxes2[:, 2]
            - boxes2[:, 0]
        ).clamp(min=0)
        *
        (
            boxes2[:, 3]
            - boxes2[:, 1]
        ).clamp(min=0)
    )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (
        rb - lt
    ).clamp(
        min=0
    )

    intersection = (
        wh[..., 0]
        * wh[..., 1]
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / (
        union.clamp(min=1e-6)
    )


# ============================================================
# Pure PyTorch NMS
# ============================================================

def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.6,
) -> torch.Tensor:
    """
    Pure PyTorch NMS。

    不依赖 torchvision.ops.nms。

    输入：

        boxes:
            [N,4]

        scores:
            [N]

    输出：

        keep:
            [K]
    """

    if boxes.numel() == 0:
        return torch.empty(
            (
                0,
            ),
            dtype=torch.long,
            device=boxes.device,
        )

    order = torch.argsort(
        scores,
        descending=True,
    )

    keep = []

    while order.numel() > 0:

        current = order[0]

        keep.append(
            current
        )

        if order.numel() == 1:
            break

        current_box = boxes[
            current
        ].unsqueeze(0)

        remaining_boxes = boxes[
            order[1:]
        ]

        ious = box_iou(
            current_box,
            remaining_boxes,
        ).squeeze(0)

        remaining = order[
            1:
        ][
            ious <= iou_threshold
        ]

        order = remaining

    return torch.stack(
        keep
    )


# ============================================================
# Class-wise NMS
# ============================================================

def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.6,
) -> torch.Tensor:
    """
    按类别执行 NMS。

    不同类别之间不会互相抑制。

    输入：

        boxes:
            [N,4]

        scores:
            [N]

        labels:
            [N]

    输出：

        keep indices
    """

    if boxes.numel() == 0:
        return torch.empty(
            (
                0,
            ),
            dtype=torch.long,
            device=boxes.device,
        )

    keep_indices = []

    unique_labels = torch.unique(
        labels
    )

    for class_id in unique_labels:

        class_mask = (
            labels == class_id
        )

        class_indices = torch.nonzero(
            class_mask,
            as_tuple=False,
        ).squeeze(1)

        class_keep = nms(
            boxes[class_indices],
            scores[class_indices],
            iou_threshold,
        )

        keep_indices.append(
            class_indices[
                class_keep
            ]
        )

    if not keep_indices:

        return torch.empty(
            (
                0,
            ),
            dtype=torch.long,
            device=boxes.device,
        )

    keep_indices = torch.cat(
        keep_indices,
        dim=0,
    )

    # NMS 后重新按照 score 排序
    keep_indices = keep_indices[
        torch.argsort(
            scores[keep_indices],
            descending=True,
        )
    ]

    return keep_indices


# ============================================================
# FCOS Point Generator
# ============================================================

class FCOSPointGenerator:
    """
    FCOS 多尺度 point generator。

    P3:
        stride = 8

    P4:
        stride = 16

    P5:
        stride = 32

    P6:
        stride = 64

    对每一个 feature map location
    生成对应的原图坐标。

    默认采用：

        x = (j + 0.5) * stride
        y = (i + 0.5) * stride
    """

    def __init__(
        self,
        strides: Sequence[int] = DEFAULT_STRIDES,
    ):
        self.strides = tuple(
            int(s)
            for s in strides
        )

    @staticmethod
    def generate_points(
        feature_height: int,
        feature_width: int,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        生成一个 feature level 的 points。

        输出：

            [H*W, 2]

        每一行：

            [x, y]
        """

        shifts_x = (
            torch.arange(
                feature_width,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) * stride

        shifts_y = (
            torch.arange(
                feature_height,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) * stride

        yy, xx = torch.meshgrid(
            shifts_y,
            shifts_x,
            indexing="ij",
        )

        points = torch.stack(
            [
                xx.reshape(-1),
                yy.reshape(-1),
            ],
            dim=-1,
        )

        return points

    def __call__(
        self,
        features: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        features：

            [
                P3,
                P4,
                P5,
                P6
            ]

        每个 feature：

            [B,C,H,W]

        返回：

            list of points
        """

        if len(features) != len(
            self.strides
        ):

            raise ValueError(
                "feature levels 数量与 strides 不一致："
                f"{len(features)} vs "
                f"{len(self.strides)}"
            )

        points = []

        for feature, stride in zip(
            features,
            self.strides,
        ):

            if feature.ndim != 4:

                raise ValueError(
                    "feature 必须为 [B,C,H,W]，"
                    f"实际：{feature.shape}"
                )

            _, _, h, w = (
                feature.shape
            )

            points.append(
                self.generate_points(
                    h,
                    w,
                    stride,
                    feature.device,
                    feature.dtype,
                )
            )

        return points


# ============================================================
# FCOS Decoder
# ============================================================

class FCOSDecoder:
    """
    将 FCOS l/t/r/b 回归结果解码成 box。

    对于一个 point：

        left   = x - l
        top    = y - t
        right  = x + r
        bottom = y + b
    """

    def __init__(
        self,
        strides: Sequence[int] = DEFAULT_STRIDES,
        regression_in_feature_units: bool = False,
    ):
        self.strides = tuple(
            int(s)
            for s in strides
        )

        self.regression_in_feature_units = (
            bool(
                regression_in_feature_units
            )
        )

    def decode_level(
        self,
        points: torch.Tensor,
        regression: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """
        points：

            [N,2]

        regression：

            [N,4]

        输出：

            [N,4]
        """

        if regression.ndim != 2:
            raise ValueError(
                "regression 必须是 [N,4]"
            )

        if regression.shape[-1] != 4:
            raise ValueError(
                "regression 最后一维必须为 4"
            )

        if (
            points.shape[0]
            != regression.shape[0]
        ):
            raise ValueError(
                "points 与 regression 数量不一致"
            )

        # ----------------------------------------------------
        # 如果 regression 仍然是 feature map
        # 单位，则转换为 image pixel 单位。
        # ----------------------------------------------------

        if self.regression_in_feature_units:

            regression = (
                regression
                * float(stride)
            )

        left = regression[
            :, 0
        ]

        top = regression[
            :, 1
        ]

        right = regression[
            :, 2
        ]

        bottom = regression[
            :, 3
        ]

        x = points[
            :, 0
        ]

        y = points[
            :, 1
        ]

        x1 = x - left
        y1 = y - top
        x2 = x + right
        y2 = y + bottom

        boxes = torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

        return boxes


# ============================================================
# Score fusion
# ============================================================

def fuse_classification_centerness(
    classification_logits: torch.Tensor,
    centerness_logits: torch.Tensor,
) -> torch.Tensor:
    """
    FCOS score fusion。

    classification_logits：

        [N,C]

    centerness_logits：

        [N]

    返回：

        [N,C]

    使用：

        sigmoid(cls)
        *
        sigmoid(centerness)
    """

    cls_prob = torch.sigmoid(
        classification_logits
    )

    center_prob = torch.sigmoid(
        centerness_logits
    )

    if center_prob.ndim == 1:

        center_prob = center_prob.unsqueeze(
            -1
        )

    return (
        cls_prob
        * center_prob
    )


# ============================================================
# Single-level output normalization
# ============================================================

def _ensure_level_tensor(
    tensor: torch.Tensor,
    expected_channels: Optional[int] = None,
    name: str = "tensor",
) -> torch.Tensor:
    """
    将不同格式统一成：

        [B,C,H,W]

    支持：

        [B,C,H,W]

        [B,H,W,C]
    """

    if tensor.ndim != 4:

        raise ValueError(
            f"{name} 必须是 4D Tensor，"
            f"实际：{tensor.shape}"
        )

    if (
        expected_channels is not None
        and tensor.shape[1]
        != expected_channels
        and tensor.shape[-1]
        == expected_channels
    ):

        tensor = tensor.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    return tensor


# ============================================================
# Main PostProcessor
# ============================================================

class FCOSPostProcessor:
    """
    FCOS 最终后处理器。

    默认参数：

        num_classes = 12

        strides = [8,16,32,64]

        score_threshold = 0.05

        nms_threshold = 0.60

        top_k = 1000

        max_detections = 100
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        strides: Sequence[int] = DEFAULT_STRIDES,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        regression_in_feature_units: bool = False,
        min_box_size: float = 1.0,
        class_names: Optional[
            Sequence[str]
        ] = None,
    ):
        self.num_classes = int(
            num_classes
        )

        self.strides = tuple(
            int(s)
            for s in strides
        )

        self.score_threshold = float(
            score_threshold
        )

        self.nms_threshold = float(
            nms_threshold
        )

        self.top_k = int(
            top_k
        )

        self.max_detections = int(
            max_detections
        )

        self.regression_in_feature_units = (
            bool(
                regression_in_feature_units
            )
        )

        self.min_box_size = float(
            min_box_size
        )

        if class_names is None:

            self.class_names = list(
                DEFAULT_CLASS_NAMES
            )

        else:

            self.class_names = list(
                class_names
            )

        if len(
            self.class_names
        ) != self.num_classes:

            raise ValueError(
                "class_names 数量必须等于 "
                "num_classes"
            )

        self.point_generator = (
            FCOSPointGenerator(
                strides=self.strides
            )
        )

        self.decoder = FCOSDecoder(
            strides=self.strides,
            regression_in_feature_units=(
                regression_in_feature_units
            ),
        )

    # ========================================================
    # Level processing
    # ========================================================

    def process_level(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
        centerness_logits: torch.Tensor,
        stride: int,
        image_height: int,
        image_width: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        处理一个 FPN level。

        输入：

            cls_logits:
                [B,C,H,W]

            bbox_regression:
                [B,4,H,W]

            centerness_logits:
                [B,1,H,W]

        输出：

            boxes:
                [B,N,4]

            scores:
                [B,N,C]

            points:
                [N,2]
        """

        if cls_logits.ndim != 4:

            raise ValueError(
                "cls_logits 必须为 [B,C,H,W]"
            )

        if bbox_regression.ndim != 4:

            raise ValueError(
                "bbox_regression 必须为 [B,4,H,W]"
            )

        if centerness_logits.ndim != 4:

            raise ValueError(
                "centerness_logits 必须为 [B,1,H,W]"
            )

        batch_size = (
            cls_logits.shape[0]
        )

        channels = (
            cls_logits.shape[1]
        )

        if channels != self.num_classes:

            raise ValueError(
                "分类通道数错误："
                f"expected={self.num_classes}, "
                f"actual={channels}"
            )

        if bbox_regression.shape[1] != 4:

            raise ValueError(
                "bbox regression 必须有 4 个通道"
            )

        if centerness_logits.shape[1] != 1:

            raise ValueError(
                "centerness 必须有 1 个通道"
            )

        _, _, h, w = (
            cls_logits.shape
        )

        # ----------------------------------------------------
        # Generate points
        # ----------------------------------------------------

        points = (
            self.point_generator.generate_points(
                h,
                w,
                stride,
                cls_logits.device,
                cls_logits.dtype,
            )
        )

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        cls_logits = (
            cls_logits
            .permute(
                0,
                2,
                3,
                1,
            )
            .reshape(
                batch_size,
                -1,
                self.num_classes,
            )
        )

        bbox_regression = (
            bbox_regression
            .permute(
                0,
                2,
                3,
                1,
            )
            .reshape(
                batch_size,
                -1,
                4,
            )
        )

        centerness_logits = (
            centerness_logits
            .permute(
                0,
                2,
                3,
                1,
            )
            .reshape(
                batch_size,
                -1,
            )
        )

        # ----------------------------------------------------
        # Decode boxes
        # ----------------------------------------------------

        boxes = []

        for batch_index in range(
            batch_size
        ):

            decoded = (
                self.decoder.decode_level(
                    points,
                    bbox_regression[
                        batch_index
                    ],
                    stride,
                )
            )

            boxes.append(
                decoded
            )

        boxes = torch.stack(
            boxes,
            dim=0,
        )

        # ----------------------------------------------------
        # Score fusion
        # ----------------------------------------------------

        scores = (
            fuse_classification_centerness(
                cls_logits,
                centerness_logits,
            )
        )

        # ----------------------------------------------------
        # Clip
        # ----------------------------------------------------

        boxes = boxes.reshape(
            -1,
            4,
        )

        boxes = clip_boxes(
            boxes,
            image_height,
            image_width,
        )

        boxes = boxes.reshape(
            batch_size,
            -1,
            4,
        )

        return (
            boxes,
            scores,
            points,
        )

    # ========================================================
    # Prediction normalization
    # ========================================================

    def _normalize_predictions(
        self,
        predictions,
    ):
        """
        将 FCOS Head 的输出统一成：

            cls_outputs
            reg_outputs
            center_outputs

        每一个都是 list：

            [
                P3,
                P4,
                P5,
                P6
            ]

        支持两种常见格式：

        1.

            {
                "cls": [...],
                "bbox": [...],
                "centerness": [...]
            }

        2.

            (
                [...],
                [...],
                [...]
            )
        """

        if isinstance(
            predictions,
            dict,
        ):

            cls_outputs = (
                predictions.get(
                    "cls",
                    predictions.get(
                        "classification"
                    ),
                )
            )

            reg_outputs = (
                predictions.get(
                    "bbox",
                    predictions.get(
                        "regression"
                    ),
                )
            )

            center_outputs = (
                predictions.get(
                    "centerness",
                    predictions.get(
                        "center",
                    ),
                )
            )

            if (
                cls_outputs is None
                or reg_outputs is None
                or center_outputs is None
            ):

                raise KeyError(
                    "predictions 字典必须包含 "
                    "cls / bbox / centerness"
                )

        elif isinstance(
            predictions,
            (
                tuple,
                list,
            ),
        ):

            if len(
                predictions
            ) != 3:

                raise ValueError(
                    "tuple/list predictions 必须包含 "
                    "(cls, bbox, centerness)"
                )

            cls_outputs = (
                predictions[0]
            )

            reg_outputs = (
                predictions[1]
            )

            center_outputs = (
                predictions[2]
            )

        else:

            raise TypeError(
                "不支持的 predictions 类型："
                f"{type(predictions)}"
            )

        if not isinstance(
            cls_outputs,
            (
                tuple,
                list,
            ),
        ):

            cls_outputs = [
                cls_outputs
            ]

        if not isinstance(
            reg_outputs,
            (
                tuple,
                list,
            ),
        ):

            reg_outputs = [
                reg_outputs
            ]

        if not isinstance(
            center_outputs,
            (
                tuple,
                list,
            ),
        ):

            center_outputs = [
                center_outputs
            ]

        if not (
            len(cls_outputs)
            == len(reg_outputs)
            == len(center_outputs)
            == len(self.strides)
        ):

            raise ValueError(
                "预测层数量必须与 strides 一致："
                f"cls={len(cls_outputs)}, "
                f"bbox={len(reg_outputs)}, "
                f"centerness={len(center_outputs)}, "
                f"strides={len(self.strides)}"
            )

        return (
            list(cls_outputs),
            list(reg_outputs),
            list(center_outputs),
        )

    # ========================================================
    # Single image postprocess
    # ========================================================

    @torch.no_grad()
    def process_single(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        对单张图像进行：

            threshold
            top-k
            label extraction
            NMS
            max detections

        输入：

            boxes:
                [N,4]

            scores:
                [N,C]

        输出：

            {
                "boxes": [K,4],
                "scores": [K],
                "labels": [K]
            }
        """

        if boxes.ndim != 2:
            raise ValueError(
                "boxes 必须是 [N,4]"
            )

        if scores.ndim != 2:
            raise ValueError(
                "scores 必须是 [N,C]"
            )

        if boxes.shape[0] != scores.shape[0]:

            raise ValueError(
                "boxes 与 scores 数量不一致"
            )

        if boxes.numel() == 0:

            return {
                "boxes": torch.empty(
                    (
                        0,
                        4,
                    ),
                    dtype=boxes.dtype,
                    device=boxes.device,
                ),
                "scores": torch.empty(
                    (
                        0,
                    ),
                    dtype=scores.dtype,
                    device=scores.device,
                ),
                "labels": torch.empty(
                    (
                        0,
                    ),
                    dtype=torch.long,
                    device=boxes.device,
                ),
            }

        # ----------------------------------------------------
        # Extract best class
        # ----------------------------------------------------

        max_scores, labels = (
            scores.max(
                dim=1
            )
        )

        # ----------------------------------------------------
        # Score threshold
        # ----------------------------------------------------

        keep = (
            max_scores
            >= self.score_threshold
        )

        boxes = boxes[
            keep
        ]

        max_scores = max_scores[
            keep
        ]

        labels = labels[
            keep
        ]

        if boxes.numel() == 0:

            return {
                "boxes": torch.empty(
                    (
                        0,
                        4,
                    ),
                    dtype=boxes.dtype,
                    device=boxes.device,
                ),
                "scores": torch.empty(
                    (
                        0,
                    ),
                    dtype=max_scores.dtype,
                    device=max_scores.device,
                ),
                "labels": torch.empty(
                    (
                        0,
                    ),
                    dtype=torch.long,
                    device=labels.device,
                ),
            }

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        if (
            self.top_k > 0
            and boxes.shape[0]
            > self.top_k
        ):

            top_scores, top_indices = (
                torch.topk(
                    max_scores,
                    k=self.top_k,
                    largest=True,
                    sorted=True,
                )
            )

            boxes = boxes[
                top_indices
            ]

            labels = labels[
                top_indices
            ]

            max_scores = top_scores

        # ----------------------------------------------------
        # Remove invalid boxes
        # ----------------------------------------------------

        (
            boxes,
            max_scores,
            labels,
        ) = remove_invalid_boxes(
            boxes,
            max_scores,
            labels,
            min_box_size=self.min_box_size,
        )

        if boxes.numel() == 0:

            return {
                "boxes": boxes.reshape(
                    -1,
                    4,
                ),
                "scores": max_scores.reshape(
                    -1
                ),
                "labels": labels.reshape(
                    -1
                ),
            }

        # ----------------------------------------------------
        # Class-wise NMS
        # ----------------------------------------------------

        keep = batched_nms(
            boxes,
            max_scores,
            labels,
            self.nms_threshold,
        )

        boxes = boxes[
            keep
        ]

        max_scores = max_scores[
            keep
        ]

        labels = labels[
            keep
        ]

        # ----------------------------------------------------
        # Final max detections
        # ----------------------------------------------------

        if (
            self.max_detections > 0
            and boxes.shape[0]
            > self.max_detections
        ):

            max_scores, top_indices = (
                torch.topk(
                    max_scores,
                    k=self.max_detections,
                    largest=True,
                    sorted=True,
                )
            )

            boxes = boxes[
                top_indices
            ]

            labels = labels[
                top_indices
            ]

        else:

            order = torch.argsort(
                max_scores,
                descending=True,
            )

            boxes = boxes[
                order
            ]

            max_scores = max_scores[
                order
            ]

            labels = labels[
                order
            ]

        return {
            "boxes": boxes,
            "scores": max_scores,
            "labels": labels,
        }

    # ========================================================
    # Main forward
    # ========================================================

    @torch.no_grad()
    def __call__(
        self,
        predictions,
        image_size: Optional[
            Tuple[int, int]
        ] = None,
    ) -> List[
        Dict[str, torch.Tensor]
    ]:
        """
        对整个 batch 进行后处理。

        参数：

            predictions:
                FCOS Head 输出

            image_size:
                (H,W)

        返回：

            List[Dict]

        每个 batch item：

            {
                "boxes": [N,4],
                "scores": [N],
                "labels": [N]
            }
        """

        (
            cls_outputs,
            reg_outputs,
            center_outputs,
        ) = self._normalize_predictions(
            predictions
        )

        # ----------------------------------------------------
        # Infer image size
        # ----------------------------------------------------

        if image_size is None:

            first_feature = (
                cls_outputs[0]
            )

            if first_feature.ndim != 4:

                raise ValueError(
                    "无法从预测结果推断 image size"
                )

            _, _, h, w = (
                first_feature.shape
            )

            image_height = (
                h
                * self.strides[0]
            )

            image_width = (
                w
                * self.strides[0]
            )

        else:

            image_height = int(
                image_size[0]
            )

            image_width = int(
                image_size[1]
            )

        # ----------------------------------------------------
        # Batch size
        # ----------------------------------------------------

        batch_size = (
            cls_outputs[0].shape[0]
        )

        # ----------------------------------------------------
        # Collect levels
        # ----------------------------------------------------

        all_boxes = []
        all_scores = []

        for (
            cls_level,
            reg_level,
            center_level,
            stride,
        ) in zip(
            cls_outputs,
            reg_outputs,
            center_outputs,
            self.strides,
        ):

            cls_level = (
                _ensure_level_tensor(
                    cls_level,
                    expected_channels=self.num_classes,
                    name="classification",
                )
            )

            reg_level = (
                _ensure_level_tensor(
                    reg_level,
                    expected_channels=4,
                    name="regression",
                )
            )

            center_level = (
                _ensure_level_tensor(
                    center_level,
                    expected_channels=1,
                    name="centerness",
                )
            )

            boxes, scores, _ = (
                self.process_level(
                    cls_level,
                    reg_level,
                    center_level,
                    stride,
                    image_height,
                    image_width,
                )
            )

            all_boxes.append(
                boxes
            )

            all_scores.append(
                scores
            )

        # ----------------------------------------------------
        # Concatenate levels
        # ----------------------------------------------------

        all_boxes = torch.cat(
            all_boxes,
            dim=1,
        )

        all_scores = torch.cat(
            all_scores,
            dim=1,
        )

        # ----------------------------------------------------
        # Per image
        # ----------------------------------------------------

        results = []

        for batch_index in range(
            batch_size
        ):

            result = self.process_single(
                all_boxes[
                    batch_index
                ],
                all_scores[
                    batch_index
                ],
            )

            results.append(
                result
            )

        return results


# ============================================================
# Convert detections to original image coordinates
# ============================================================

def scale_boxes_to_original_image(
    boxes: torch.Tensor,
    original_size: Tuple[int, int],
    current_size: Tuple[int, int],
) -> torch.Tensor:
    """
    将网络输入尺寸下的 boxes
    映射回原始图像。

    参数：

        boxes:
            [N,4]

        original_size:
            (orig_h, orig_w)

        current_size:
            (input_h, input_w)
    """

    if boxes.numel() == 0:

        return boxes.reshape(
            -1,
            4,
        )

    orig_h, orig_w = (
        int(original_size[0]),
        int(original_size[1]),
    )

    current_h, current_w = (
        int(current_size[0]),
        int(current_size[1]),
    )

    scale_x = (
        float(orig_w)
        / float(current_w)
    )

    scale_y = (
        float(orig_h)
        / float(current_h)
    )

    scaled = boxes.clone()

    scaled[:, [0, 2]] *= (
        scale_x
    )

    scaled[:, [1, 3]] *= (
        scale_y
    )

    scaled = clip_boxes(
        scaled,
        orig_h,
        orig_w,
    )

    return scaled


# ============================================================
# Convert result to CPU / numpy
# ============================================================

def detections_to_cpu(
    detections: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    将检测结果转移到 CPU。
    """

    return {
        key: value.detach().cpu()
        for key, value
        in detections.items()
    }


def detections_to_numpy(
    detections: Dict[str, torch.Tensor],
) -> Dict[str, "object"]:
    """
    将检测结果转换成 numpy。

    返回：

        boxes:
            np.ndarray

        scores:
            np.ndarray

        labels:
            np.ndarray
    """

    return {
        key: value.detach().cpu().numpy()
        for key, value
        in detections.items()
    }


# ============================================================
# Detection formatting
# ============================================================

def format_detections(
    detections: Dict[str, torch.Tensor],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> List[Dict[str, object]]:
    """
    将 Tensor 检测结果转换为易读的 Python list。

    每个 detection：

        {
            "box": [x1,y1,x2,y2],
            "score": float,
            "label": int,
            "class_name": str
        }
    """

    boxes = (
        detections["boxes"]
        .detach()
        .cpu()
    )

    scores = (
        detections["scores"]
        .detach()
        .cpu()
    )

    labels = (
        detections["labels"]
        .detach()
        .cpu()
    )

    result = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):

        label_id = int(
            label.item()
        )

        if (
            0 <= label_id
            < len(class_names)
        ):

            class_name = (
                class_names[
                    label_id
                ]
            )

        else:

            class_name = (
                f"class_{label_id}"
            )

        result.append(
            {
                "box": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ],
                "score": float(
                    score
                ),
                "label": label_id,
                "class_name": class_name,
            }
        )

    return result


# ============================================================
# Convenience function
# ============================================================

def postprocess_predictions(
    predictions,
    image_size: Tuple[int, int] = (
        640,
        640,
    ),
    num_classes: int = 12,
    strides: Sequence[int] = DEFAULT_STRIDES,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.60,
    top_k: int = 1000,
    max_detections: int = 100,
    regression_in_feature_units: bool = False,
):
    """
    函数式接口。

    用于 inference.py。

    示例：

        processor = FCOSPostProcessor(...)

    或：

        results = postprocess_predictions(
            predictions,
            image_size=(640,640),
        )
    """

    processor = FCOSPostProcessor(
        num_classes=num_classes,
        strides=strides,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        max_detections=max_detections,
        regression_in_feature_units=(
            regression_in_feature_units
        ),
    )

    return processor(
        predictions,
        image_size=image_size,
    )


# ============================================================
# Synthetic prediction generator
# ============================================================

def create_dummy_predictions(
    batch_size: int = 2,
    num_classes: int = 12,
    image_size: int = 640,
    strides: Sequence[int] = DEFAULT_STRIDES,
):
    """
    创建假的 FCOS 输出，
    用于测试 postprocess.py。

    P3:
        80 × 80

    P4:
        40 × 40

    P5:
        20 × 20

    P6:
        10 × 10
    """

    cls_outputs = []
    reg_outputs = []
    center_outputs = []

    for stride in strides:

        h = image_size // stride
        w = image_size // stride

        cls_outputs.append(
            torch.randn(
                batch_size,
                num_classes,
                h,
                w,
            )
        )

        # ----------------------------------------------------
        # FCOS regression should be positive.
        # ----------------------------------------------------

        reg_outputs.append(
            torch.rand(
                batch_size,
                4,
                h,
                w,
            )
            * 8.0
        )

        center_outputs.append(
            torch.randn(
                batch_size,
                1,
                h,
                w,
            )
        )

    return {
        "cls": cls_outputs,
        "bbox": reg_outputs,
        "centerness": center_outputs,
    }


# ============================================================
# Test score fusion
# ============================================================

def test_score_fusion():
    """
    测试分类分数和 centerness 融合。
    """

    print("\n")
    print("=" * 70)
    print("Testing Score Fusion")
    print("=" * 70)

    cls_logits = torch.tensor(
        [
            [
                2.0,
                -1.0,
                0.0,
            ],
            [
                -1.0,
                3.0,
                0.0,
            ],
        ]
    )

    center_logits = torch.tensor(
        [
            2.0,
            -1.0,
        ]
    )

    scores = (
        fuse_classification_centerness(
            cls_logits,
            center_logits,
        )
    )

    print(
        "Scores:"
    )

    print(
        scores
    )

    assert scores.shape == (
        2,
        3,
    )

    assert torch.all(
        scores >= 0
    )

    assert torch.all(
        scores <= 1
    )

    print(
        "\nScore fusion test passed."
    )

    print("=" * 70)


# ============================================================
# Test NMS
# ============================================================

def test_nms():
    """
    测试 NMS。
    """

    print("\n")
    print("=" * 70)
    print("Testing NMS")
    print("=" * 70)

    boxes = torch.tensor(
        [
            [
                10.0,
                10.0,
                100.0,
                100.0,
            ],
            [
                12.0,
                12.0,
                98.0,
                98.0,
            ],
            [
                200.0,
                200.0,
                300.0,
                300.0,
            ],
        ]
    )

    scores = torch.tensor(
        [
            0.95,
            0.80,
            0.90,
        ]
    )

    keep = nms(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    print(
        "Keep indices:"
    )

    print(
        keep
    )

    assert keep.numel() == 2

    print(
        "\nNMS test passed."
    )

    print("=" * 70)


# ============================================================
# Test postprocessor
# ============================================================

def test_postprocessor():
    """
    完整测试：

        dummy predictions
            ↓
        FCOSPostProcessor
            ↓
        detections
    """

    print("\n")
    print("=" * 70)
    print("Testing FCOS PostProcessor")
    print("=" * 70)

    predictions = (
        create_dummy_predictions(
            batch_size=2,
            num_classes=12,
            image_size=640,
            strides=(
                8,
                16,
                32,
                64,
            ),
        )
    )

    processor = (
        FCOSPostProcessor(
            num_classes=12,
            strides=(
                8,
                16,
                32,
                64,
            ),
            score_threshold=0.50,
            nms_threshold=0.60,
            top_k=300,
            max_detections=100,
            regression_in_feature_units=False,
        )
    )

    results = processor(
        predictions,
        image_size=(
            640,
            640,
        ),
    )

    assert len(
        results
    ) == 2

    for index, result in enumerate(
        results
    ):

        print(
            f"\nImage {index}:"
        )

        print(
            f"  boxes : "
            f"{tuple(result['boxes'].shape)}"
        )

        print(
            f"  scores: "
            f"{tuple(result['scores'].shape)}"
        )

        print(
            f"  labels: "
            f"{tuple(result['labels'].shape)}"
        )

        assert (
            result["boxes"].ndim
            == 2
        )

        assert (
            result["boxes"].shape[1]
            == 4
        )

        assert (
            result["scores"].ndim
            == 1
        )

        assert (
            result["labels"].ndim
            == 1
        )

        assert (
            result["boxes"].shape[0]
            == result["scores"].shape[0]
            == result["labels"].shape[0]
        )

        if result["boxes"].numel() > 0:

            assert torch.all(
                result["boxes"][:, 0]
                >= 0
            )

            assert torch.all(
                result["boxes"][:, 1]
                >= 0
            )

            assert torch.all(
                result["boxes"][:, 2]
                <= 640
            )

            assert torch.all(
                result["boxes"][:, 3]
                <= 640
            )

            assert torch.all(
                result["scores"]
                >= 0
            )

            assert torch.all(
                result["scores"]
                <= 1
            )

            assert torch.all(
                result["labels"]
                >= 0
            )

            assert torch.all(
                result["labels"]
                < 12
            )

    print(
        "\nPostprocessor test passed."
    )

    print("=" * 70)


# ============================================================
# Test coordinate conversion
# ============================================================

def test_scale_boxes():
    """
    测试从 640×640 映射回原图。
    """

    print("\n")
    print("=" * 70)
    print("Testing Box Coordinate Scaling")
    print("=" * 70)

    boxes = torch.tensor(
        [
            [
                100.0,
                100.0,
                300.0,
                300.0,
            ],
        ]
    )

    scaled = (
        scale_boxes_to_original_image(
            boxes,
            original_size=(
                1280,
                1920,
            ),
            current_size=(
                640,
                640,
            ),
        )
    )

    print(
        "Original input boxes:"
    )

    print(
        boxes
    )

    print(
        "\nScaled boxes:"
    )

    print(
        scaled
    )

    expected = torch.tensor(
        [
            [
                300.0,
                200.0,
                900.0,
                600.0,
            ],
        ]
    )

    assert torch.allclose(
        scaled,
        expected,
        atol=1e-5,
    )

    print(
        "\nCoordinate scaling test passed."
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FCOS postprocess.py"
    )

    print(
        "=" * 70
    )

    test_score_fusion()

    test_nms()

    test_scale_boxes()

    test_postprocessor()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "All postprocess.py tests passed."
    )

    print(
        "=" * 70
    )