# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic utilities
# ============================================================

EPS = 1e-6


def _empty_tensor(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    创建空 Tensor。
    """
    return torch.empty(
        0,
        device=device,
        dtype=dtype,
    )


# ============================================================
# Sigmoid Focal Loss
# ============================================================

class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss。

    FCOS 分类分支采用：

        FL(p_t)
        =
        -alpha_t (1-p_t)^gamma log(p_t)

    输入：

        logits:
            [N, C]

        targets:
            [N, C]

    targets 为：

        0 -> negative
        1 -> positive

    参数：

        alpha = 0.25
        gamma = 2.0
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "sum",
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha 必须位于 [0,1]。"
            )

        if gamma < 0:
            raise ValueError(
                "gamma 必须 >= 0。"
            )

        if reduction not in (
            "none",
            "sum",
            "mean",
        ):
            raise ValueError(
                "reduction 必须为 none、sum 或 mean。"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.shape != targets.shape:
            raise ValueError(
                "logits 与 targets 的 shape 必须一致。"
            )

        targets = targets.to(
            dtype=logits.dtype
        )

        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # p_t
        prob = torch.sigmoid(logits)

        p_t = (
            prob * targets
            + (1.0 - prob) * (1.0 - targets)
        )

        # alpha_t
        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha)
            * (1.0 - targets)
        )

        # focal modulation
        focal_weight = (
            alpha_t
            * (1.0 - p_t).pow(self.gamma)
        )

        loss = (
            focal_weight
            * bce_loss
        )

        if self.reduction == "none":
            return loss

        if self.reduction == "mean":
            return loss.mean()

        return loss.sum()


# ============================================================
# IoU / GIoU
# ============================================================

def box_area(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    boxes:

        [..., 4]

    format：

        x1, y1, x2, y2
    """

    width = (
        boxes[..., 2]
        - boxes[..., 0]
    ).clamp(min=0)

    height = (
        boxes[..., 3]
        - boxes[..., 1]
    ).clamp(min=0)

    return width * height


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    计算逐元素 IoU。

    boxes1:
        [N,4]

    boxes2:
        [N,4]

    返回：

        [N]
    """

    if boxes1.shape != boxes2.shape:
        raise ValueError(
            "box_iou 要求 boxes1 与 boxes2 shape 一致。"
        )

    if boxes1.ndim != 2:
        raise ValueError(
            "boxes 必须为 [N,4]。"
        )

    if boxes1.shape[-1] != 4:
        raise ValueError(
            "boxes 最后一维必须为 4。"
        )

    inter_x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0],
    )

    inter_y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1],
    )

    inter_x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2],
    )

    inter_y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3],
    )

    inter_w = (
        inter_x2 - inter_x1
    ).clamp(min=0)

    inter_h = (
        inter_y2 - inter_y1
    ).clamp(min=0)

    inter_area = (
        inter_w * inter_h
    )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    union = (
        area1
        + area2
        - inter_area
    ).clamp(min=EPS)

    return inter_area / union


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    计算逐元素 GIoU。

    GIoU：

        IoU
        -
        (C - Union) / C

    其中 C 为两个 box 的最小外接区域。

    输入：

        boxes1: [N,4]
        boxes2: [N,4]

    输出：

        [N]
    """

    if boxes1.shape != boxes2.shape:
        raise ValueError(
            "GIoU 要求 boxes1 与 boxes2 shape 一致。"
        )

    if boxes1.ndim != 2:
        raise ValueError(
            "boxes 必须为 [N,4]。"
        )

    if boxes1.shape[-1] != 4:
        raise ValueError(
            "boxes 最后一维必须为 4。"
        )

    iou = box_iou(
        boxes1,
        boxes2,
    )

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    inter_x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0],
    )

    inter_y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1],
    )

    inter_x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2],
    )

    inter_y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3],
    )

    inter_w = (
        inter_x2 - inter_x1
    ).clamp(min=0)

    inter_h = (
        inter_y2 - inter_y1
    ).clamp(min=0)

    inter_area = (
        inter_w * inter_h
    )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    union = (
        area1
        + area2
        - inter_area
    ).clamp(min=EPS)

    # --------------------------------------------------------
    # Smallest enclosing box
    # --------------------------------------------------------

    enc_x1 = torch.minimum(
        boxes1[:, 0],
        boxes2[:, 0],
    )

    enc_y1 = torch.minimum(
        boxes1[:, 1],
        boxes2[:, 1],
    )

    enc_x2 = torch.maximum(
        boxes1[:, 2],
        boxes2[:, 2],
    )

    enc_y2 = torch.maximum(
        boxes1[:, 3],
        boxes2[:, 3],
    )

    enc_w = (
        enc_x2 - enc_x1
    ).clamp(min=0)

    enc_h = (
        enc_y2 - enc_y1
    ).clamp(min=0)

    enc_area = (
        enc_w * enc_h
    ).clamp(min=EPS)

    giou = (
        iou
        - (enc_area - union)
        / enc_area
    )

    return giou


class GIoULoss(nn.Module):
    """
    GIoU Loss：

        L_GIoU = 1 - GIoU
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:

        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        giou = generalized_box_iou(
            pred_boxes,
            target_boxes,
        )

        loss = 1.0 - giou

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        return loss.mean()


# ============================================================
# FCOS Centerness
# ============================================================

def compute_centerness(
    ltrb: torch.Tensor,
) -> torch.Tensor:
    """
    根据 FCOS l/t/r/b 计算 centerness。

    输入：

        [N,4]

    顺序：

        l, t, r, b

    公式：

        sqrt(
            min(l,r)/max(l,r)
            *
            min(t,b)/max(t,b)
        )

    返回：

        [N]
    """

    if ltrb.ndim != 2:
        raise ValueError(
            "ltrb 必须为 [N,4]。"
        )

    if ltrb.shape[-1] != 4:
        raise ValueError(
            "ltrb 最后一维必须为 4。"
        )

    left_right = ltrb[
        :, [0, 2]
    ]

    top_bottom = ltrb[
        :, [1, 3]
    ]

    lr_min = left_right.min(
        dim=1
    ).values

    lr_max = left_right.max(
        dim=1
    ).values.clamp(min=EPS)

    tb_min = top_bottom.min(
        dim=1
    ).values

    tb_max = top_bottom.max(
        dim=1
    ).values.clamp(min=EPS)

    centerness = torch.sqrt(
        (
            lr_min / lr_max
        )
        *
        (
            tb_min / tb_max
        )
    )

    return centerness.clamp(
        min=0.0,
        max=1.0,
    )


# ============================================================
# FCOS Point Generator
# ============================================================

class FCOSPointGenerator:
    """
    FCOS 多尺度 point generator。

    P3-P6：

        P3 stride 8
        P4 stride 16
        P5 stride 32
        P6 stride 64
    """

    def __init__(
        self,
        strides: Sequence[int] = (
            8,
            16,
            32,
            64,
        ),
        center_sampling: bool = False,
        center_sampling_radius: float = 1.5,
    ):
        self.strides = tuple(
            int(x)
            for x in strides
        )

        self.center_sampling = (
            center_sampling
        )

        self.center_sampling_radius = (
            center_sampling_radius
        )

    def generate_single(
        self,
        height: int,
        width: int,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        生成一个 level 的 points。

        返回：

            [H*W, 2]

        坐标：

            x, y
        """

        shifts_x = (
            torch.arange(
                width,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) * stride

        shifts_y = (
            torch.arange(
                height,
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

    def generate(
        self,
        feature_shapes: Sequence[
            Tuple[int, int]
        ],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        输入：

            [(H3,W3), ...]

        返回：

            points_per_level
            strides_per_level
        """

        if len(feature_shapes) != len(
            self.strides
        ):
            raise ValueError(
                "feature_shapes 与 strides 数量不一致。"
            )

        points = []
        strides = []

        for (
            (height, width),
            stride,
        ) in zip(
            feature_shapes,
            self.strides,
        ):

            level_points = self.generate_single(
                height=height,
                width=width,
                stride=stride,
                device=device,
                dtype=dtype,
            )

            points.append(
                level_points
            )

            strides.append(
                torch.full(
                    (
                        height * width,
                    ),
                    float(stride),
                    device=device,
                    dtype=dtype,
                )
            )

        return points, strides


# ============================================================
# FCOS Target Assigner
# ============================================================

class FCOSTargetAssigner:
    """
    FCOS target assignment。

    每一个 feature point：

        1. 判断是否位于 GT box 内
        2. 判断该 box 是否属于当前 FPN level
        3. 多个 GT 同时匹配时选择面积最小者

    size ranges：

        P3: [-1, 64]
        P4: [64, 128]
        P5: [128, 256]
        P6: [256, INF]

    对于一个 point：

        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y1

    正样本要求：

        min(l,t,r,b) > 0

    如果一个 point 同时落入多个 GT，
    选择面积最小的 GT。
    """

    def __init__(
        self,
        strides: Sequence[int] = (
            8,
            16,
            32,
            64,
        ),
        regress_ranges: Optional[
            Sequence[Tuple[float, float]]
        ] = None,
        center_sampling: bool = False,
        center_sampling_radius: float = 1.5,
    ):
        self.strides = tuple(
            int(x)
            for x in strides
        )

        if regress_ranges is None:

            self.regress_ranges = (
                (-1.0, 64.0),
                (64.0, 128.0),
                (128.0, 256.0),
                (256.0, float("inf")),
            )

        else:

            self.regress_ranges = tuple(
                (
                    float(low),
                    float(high),
                )
                for low, high
                in regress_ranges
            )

        if len(self.regress_ranges) != len(
            self.strides
        ):
            raise ValueError(
                "regress_ranges 与 strides 数量必须一致。"
            )

        self.center_sampling = (
            center_sampling
        )

        self.center_sampling_radius = (
            center_sampling_radius
        )

    def _center_sampling_mask(
        self,
        points: torch.Tensor,
        boxes: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """
        FCOS center sampling。

        返回：

            [N_points, N_gt]

        默认不开启，因此默认训练配置与
        config.py 保持简单。
        """

        if boxes.numel() == 0:
            return torch.zeros(
                (
                    points.shape[0],
                    0,
                ),
                dtype=torch.bool,
                device=points.device,
            )

        centers_x = (
            boxes[:, 0]
            + boxes[:, 2]
        ) * 0.5

        centers_y = (
            boxes[:, 1]
            + boxes[:, 3]
        ) * 0.5

        radius = (
            self.center_sampling_radius
            * stride
        )

        center_x_min = torch.maximum(
            boxes[:, 0],
            centers_x - radius,
        )

        center_y_min = torch.maximum(
            boxes[:, 1],
            centers_y - radius,
        )

        center_x_max = torch.minimum(
            boxes[:, 2],
            centers_x + radius,
        )

        center_y_max = torch.minimum(
            boxes[:, 3],
            centers_y + radius,
        )

        px = points[:, 0].unsqueeze(1)
        py = points[:, 1].unsqueeze(1)

        return (
            (px >= center_x_min.unsqueeze(0))
            & (px <= center_x_max.unsqueeze(0))
            & (py >= center_y_min.unsqueeze(0))
            & (py <= center_y_max.unsqueeze(0))
        )

    def assign_single_level(
        self,
        points: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        regress_range: Tuple[float, float],
        stride: int,
    ) -> Dict[str, torch.Tensor]:
        """
        单个 FPN level 进行 target assignment。

        返回：

            labels:
                [N]

                -1 -> ignore
                 0...C-1 -> positive

            bbox_targets:
                [N,4]

            centerness_targets:
                [N]

            positive_mask:
                [N]
        """

        num_points = points.shape[0]
        device = points.device

        # ----------------------------------------------------
        # No GT
        # ----------------------------------------------------

        if boxes.numel() == 0:

            return {
                "labels": torch.full(
                    (num_points,),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
                "bbox_targets": torch.zeros(
                    (
                        num_points,
                        4,
                    ),
                    dtype=points.dtype,
                    device=device,
                ),
                "centerness_targets": torch.zeros(
                    num_points,
                    dtype=points.dtype,
                    device=device,
                ),
                "positive_mask": torch.zeros(
                    num_points,
                    dtype=torch.bool,
                    device=device,
                ),
            }

        # ----------------------------------------------------
        # l/t/r/b
        # ----------------------------------------------------

        x = points[:, 0].unsqueeze(1)
        y = points[:, 1].unsqueeze(1)

        left = (
            x
            - boxes[:, 0].unsqueeze(0)
        )

        top = (
            y
            - boxes[:, 1].unsqueeze(0)
        )

        right = (
            boxes[:, 2].unsqueeze(0)
            - x
        )

        bottom = (
            boxes[:, 3].unsqueeze(0)
            - y
        )

        bbox_targets = torch.stack(
            [
                left,
                top,
                right,
                bottom,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Inside GT box
        # ----------------------------------------------------

        inside_box = (
            bbox_targets.min(
                dim=-1
            ).values > 0
        )

        # ----------------------------------------------------
        # Regression range
        # ----------------------------------------------------

        max_regression = (
            bbox_targets.max(
                dim=-1
            ).values
        )

        range_low, range_high = (
            regress_range
        )

        in_range = (
            max_regression >= range_low
        ) & (
            max_regression <= range_high
        )

        # ----------------------------------------------------
        # Center sampling
        # ----------------------------------------------------

        if self.center_sampling:

            center_mask = (
                self._center_sampling_mask(
                    points,
                    boxes,
                    stride,
                )
            )

        else:

            center_mask = torch.ones_like(
                inside_box
            )

        # ----------------------------------------------------
        # Final candidate mask
        # ----------------------------------------------------

        candidate_mask = (
            inside_box
            & in_range
            & center_mask
        )

        # ----------------------------------------------------
        # Choose smallest GT
        # ----------------------------------------------------

        areas = (
            (
                boxes[:, 2]
                - boxes[:, 0]
            ).clamp(min=0)
            *
            (
                boxes[:, 3]
                - boxes[:, 1]
            ).clamp(min=0)
        )

        areas = areas.unsqueeze(0).expand(
            num_points,
            -1,
        )

        # 非候选 GT 设置为 inf。
        areas = torch.where(
            candidate_mask,
            areas,
            torch.full_like(
                areas,
                float("inf"),
            ),
        )

        min_area, min_index = (
            areas.min(dim=1)
        )

        positive_mask = torch.isfinite(
            min_area
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        assigned_labels = torch.full(
            (
                num_points,
            ),
            -1,
            dtype=torch.long,
            device=device,
        )

        assigned_labels[
            positive_mask
        ] = labels[
            min_index[
                positive_mask
            ]
        ].long()

        # ----------------------------------------------------
        # Bbox target
        # ----------------------------------------------------

        assigned_bbox_targets = torch.zeros(
            (
                num_points,
                4,
            ),
            dtype=points.dtype,
            device=device,
        )

        assigned_bbox_targets[
            positive_mask
        ] = bbox_targets[
            positive_mask,
            min_index[
                positive_mask
            ],
        ]

        # ----------------------------------------------------
        # Centerness
        # ----------------------------------------------------

        assigned_centerness = torch.zeros(
            num_points,
            dtype=points.dtype,
            device=device,
        )

        if positive_mask.any():

            assigned_centerness[
                positive_mask
            ] = compute_centerness(
                assigned_bbox_targets[
                    positive_mask
                ]
            )

        return {
            "labels": assigned_labels,
            "bbox_targets": assigned_bbox_targets,
            "centerness_targets": assigned_centerness,
            "positive_mask": positive_mask,
        }

    def assign_single_image(
        self,
        points_per_level: Sequence[torch.Tensor],
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        对一张图像的 P3-P6 进行 target assignment。
        """

        all_labels = []
        all_bbox_targets = []
        all_centerness = []
        all_positive = []

        for level_index, (
            points,
            stride,
            regress_range,
        ) in enumerate(
            zip(
                points_per_level,
                self.strides,
                self.regress_ranges,
            )
        ):

            result = self.assign_single_level(
                points=points,
                boxes=boxes,
                labels=labels,
                regress_range=regress_range,
                stride=stride,
            )

            all_labels.append(
                result["labels"]
            )

            all_bbox_targets.append(
                result["bbox_targets"]
            )

            all_centerness.append(
                result["centerness_targets"]
            )

            all_positive.append(
                result["positive_mask"]
            )

        return {
            "labels": all_labels,
            "bbox_targets": all_bbox_targets,
            "centerness_targets": all_centerness,
            "positive_mask": all_positive,
        }


# ============================================================
# FCOS Loss
# ============================================================

class FCOSLoss(nn.Module):
    """
    FCOS 总损失。

    总损失：

        L =
            L_cls
            + lambda_bbox * L_bbox
            + lambda_ctr * L_ctr

    默认：

        lambda_cls  = 1
        lambda_bbox = 1
        lambda_ctr  = 1

    与 config.py：

        focal_alpha = 0.25
        focal_gamma = 2.0
        bbox_loss_weight = 1.0
        centerness_loss_weight = 1.0
        classification_loss_weight = 1.0
        normalize_by_positive = True

    对应。
    """

    def __init__(
        self,
        num_classes: int = 12,
        strides: Sequence[int] = (
            8,
            16,
            32,
            64,
        ),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bbox_loss_weight: float = 1.0,
        centerness_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        normalize_by_positive: bool = True,
        center_sampling: bool = False,
        center_sampling_radius: float = 1.5,
        regress_ranges: Optional[
            Sequence[Tuple[float, float]]
        ] = None,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.strides = tuple(
            int(x)
            for x in strides
        )

        self.bbox_loss_weight = (
            bbox_loss_weight
        )

        self.centerness_loss_weight = (
            centerness_loss_weight
        )

        self.classification_loss_weight = (
            classification_loss_weight
        )

        self.normalize_by_positive = (
            normalize_by_positive
        )

        self.focal_loss = SigmoidFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="sum",
        )

        self.bbox_loss = GIoULoss(
            reduction="sum"
        )

        self.target_assigner = (
            FCOSTargetAssigner(
                strides=self.strides,
                regress_ranges=regress_ranges,
                center_sampling=center_sampling,
                center_sampling_radius=center_sampling_radius,
            )
        )

    # ========================================================
    # Prepare predictions
    # ========================================================

    def _flatten_level_prediction(
        self,
        prediction: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        """
        [B,C,H,W]
        ->
        [B,H*W,C]
        """

        if prediction.ndim != 4:
            raise ValueError(
                "prediction 必须为 [B,C,H,W]。"
            )

        if prediction.shape[1] != channels:
            raise ValueError(
                f"prediction 通道数错误，"
                f"期望 {channels}，"
                f"实际 {prediction.shape[1]}。"
            )

        return prediction.permute(
            0,
            2,
            3,
            1,
        ).contiguous().view(
            prediction.shape[0],
            -1,
            channels,
        )

    # ========================================================
    # Build targets
    # ========================================================

    def build_targets(
        self,
        cls_outputs: Sequence[torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, List]:
        """
        为整个 batch 构造 P3-P6 targets。

        targets 中每张图片：

            {
                "boxes": [N,4],
                "labels": [N]
            }

        返回：

            labels
            bbox_targets
            centerness_targets
            positive_masks

        每个字段都是：

            List[level][batch]
        """

        if len(cls_outputs) != len(
            self.strides
        ):
            raise ValueError(
                "预测 level 数量与 strides 不一致。"
            )

        batch_size = (
            cls_outputs[0].shape[0]
        )

        if len(targets) != batch_size:
            raise ValueError(
                "targets 数量必须等于 batch size。"
            )

        # ----------------------------------------------------
        # Feature shapes
        # ----------------------------------------------------

        feature_shapes = []

        for prediction in cls_outputs:

            feature_shapes.append(
                (
                    prediction.shape[2],
                    prediction.shape[3],
                )
            )

        device = cls_outputs[0].device
        dtype = cls_outputs[0].dtype

        # ----------------------------------------------------
        # Generate points
        # ----------------------------------------------------

        point_generator = FCOSPointGenerator(
            strides=self.strides,
        )

        points_per_level, _ = (
            point_generator.generate(
                feature_shapes=feature_shapes,
                device=device,
                dtype=dtype,
            )
        )

        # ----------------------------------------------------
        # Batch targets
        # ----------------------------------------------------

        labels_all = [
            []
            for _ in self.strides
        ]

        bbox_all = [
            []
            for _ in self.strides
        ]

        centerness_all = [
            []
            for _ in self.strides
        ]

        positive_all = [
            []
            for _ in self.strides
        ]

        for image_index in range(
            batch_size
        ):

            image_target = targets[
                image_index
            ]

            boxes = image_target[
                "boxes"
            ]

            labels = image_target[
                "labels"
            ]

            if boxes is None:

                boxes = torch.empty(
                    (
                        0,
                        4,
                    ),
                    device=device,
                    dtype=dtype,
                )

            if labels is None:

                labels = torch.empty(
                    (
                        0,
                    ),
                    device=device,
                    dtype=torch.long,
                )

            boxes = boxes.to(
                device=device,
                dtype=dtype,
            )

            labels = labels.to(
                device=device,
                dtype=torch.long,
            )

            result = (
                self.target_assigner.assign_single_image(
                    points_per_level=points_per_level,
                    boxes=boxes,
                    labels=labels,
                )
            )

            for level_index in range(
                len(self.strides)
            ):

                labels_all[
                    level_index
                ].append(
                    result["labels"][
                        level_index
                    ]
                )

                bbox_all[
                    level_index
                ].append(
                    result["bbox_targets"][
                        level_index
                    ]
                )

                centerness_all[
                    level_index
                ].append(
                    result[
                        "centerness_targets"
                    ][level_index]
                )

                positive_all[
                    level_index
                ].append(
                    result[
                        "positive_mask"
                    ][level_index]
                )

        return {
            "labels": labels_all,
            "bbox_targets": bbox_all,
            "centerness_targets": centerness_all,
            "positive_masks": positive_all,
            "points_per_level": points_per_level,
        }

    # ========================================================
    # Classification target
    # ========================================================

    def _build_class_targets(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        将：

            [N]

        转为：

            [N,C]

        其中：

            -1 -> background / ignore
            0...C-1 -> positive class

        对于 FCOS：

            background = 全 0

        ignore：

            全 0，并通过 positive mask /
            valid mask 控制损失。

        这里返回 one-hot target。
        """

        num_points = labels.shape[0]

        class_targets = torch.zeros(
            (
                num_points,
                self.num_classes,
            ),
            dtype=torch.float32,
            device=labels.device,
        )

        positive = (
            labels >= 0
        )

        if positive.any():

            positive_labels = labels[
                positive
            ]

            if (
                positive_labels.min() < 0
                or positive_labels.max()
                >= self.num_classes
            ):
                raise ValueError(
                    "GT label 超出 num_classes 范围。"
                )

            class_targets[
                positive,
                positive_labels,
            ] = 1.0

        return class_targets

    # ========================================================
    # Single level loss
    # ========================================================

    def _loss_single_level(
        self,
        cls_logits: torch.Tensor,
        bbox_reg: torch.Tensor,
        centerness: torch.Tensor,
        level_labels: List[torch.Tensor],
        level_bbox_targets: List[torch.Tensor],
        level_centerness_targets: List[torch.Tensor],
        level_positive_masks: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算单个 FPN level 的 loss。
        """

        batch_size = (
            cls_logits.shape[0]
        )

        num_points = (
            cls_logits.shape[2]
            * cls_logits.shape[3]
        )

        # ----------------------------------------------------
        # Flatten prediction
        # ----------------------------------------------------

        cls_pred = (
            self._flatten_level_prediction(
                cls_logits,
                self.num_classes,
            )
        )

        bbox_pred = (
            self._flatten_level_prediction(
                bbox_reg,
                4,
            )
        )

        center_pred = (
            self._flatten_level_prediction(
                centerness,
                1,
            ).squeeze(-1)
        )

        # ----------------------------------------------------
        # Flatten targets
        # ----------------------------------------------------

        labels = torch.stack(
            level_labels,
            dim=0,
        )

        bbox_targets = torch.stack(
            level_bbox_targets,
            dim=0,
        )

        center_targets = torch.stack(
            level_centerness_targets,
            dim=0,
        )

        positive_mask = torch.stack(
            level_positive_masks,
            dim=0,
        )

        if labels.shape != (
            batch_size,
            num_points,
        ):
            raise RuntimeError(
                "FCOS target shape 与 prediction 不一致。"
            )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        cls_target = []

        for batch_index in range(
            batch_size
        ):

            cls_target.append(
                self._build_class_targets(
                    labels[batch_index]
                )
            )

        cls_target = torch.stack(
            cls_target,
            dim=0,
        )

        cls_pred_flat = cls_pred.reshape(
            -1,
            self.num_classes,
        )

        cls_target_flat = cls_target.reshape(
            -1,
            self.num_classes,
        )

        cls_loss = self.focal_loss(
            cls_pred_flat,
            cls_target_flat,
        )

        # ----------------------------------------------------
        # Positive samples
        # ----------------------------------------------------

        positive = (
            positive_mask
            .reshape(-1)
        )

        num_positive = positive.sum()

        # ----------------------------------------------------
        # Bbox loss
        # ----------------------------------------------------

        if num_positive > 0:

            bbox_pred_positive = (
                bbox_pred.reshape(
                    -1,
                    4,
                )[positive]
            )

            bbox_target_positive = (
                bbox_targets.reshape(
                    -1,
                    4,
                )[positive]
            )

            center_target_positive = (
                center_targets.reshape(
                    -1
                )[positive]
            )

            # ------------------------------------------------
            # Need points
            # ------------------------------------------------
            #
            # bbox prediction 是 l/t/r/b，
            # 这里先根据 level points 恢复为
            # x1/y1/x2/y2。
            #
            # points 在外部计算并在 forward 中传入。
            #
            # 因此这里暂时返回 ltrb，
            # 实际 GIoU 在 forward 中完成。
            # ------------------------------------------------

            bbox_loss = bbox_pred_positive.sum() * 0.0

            center_loss = F.binary_cross_entropy_with_logits(
                center_pred.reshape(-1)[positive],
                center_target_positive,
                reduction="sum",
            )

        else:

            bbox_loss = (
                bbox_pred.sum() * 0.0
            )

            center_loss = (
                center_pred.sum() * 0.0
            )

        return {
            "cls_loss": cls_loss,
            "bbox_loss_placeholder": bbox_loss,
            "centerness_loss": center_loss,
            "num_positive": num_positive.float(),
            "bbox_pred_positive": (
                bbox_pred.reshape(-1, 4)[positive]
            ),
            "bbox_target_positive": (
                bbox_targets.reshape(-1, 4)[positive]
            ),
            "positive_mask": positive,
            "center_target_positive": (
                center_targets.reshape(-1)[positive]
            ),
        }

    # ========================================================
    # Decode ltrb
    # ========================================================

    @staticmethod
    def decode_ltrb(
        points: torch.Tensor,
        ltrb: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 l/t/r/b 转换成 xyxy。

        points:

            [N,2]

        ltrb:

            [N,4]
        """

        if points.ndim != 2:
            raise ValueError(
                "points 必须为 [N,2]。"
            )

        if ltrb.ndim != 2:
            raise ValueError(
                "ltrb 必须为 [N,4]。"
            )

        x = points[:, 0]
        y = points[:, 1]

        x1 = (
            x - ltrb[:, 0]
        )

        y1 = (
            y - ltrb[:, 1]
        )

        x2 = (
            x + ltrb[:, 2]
        )

        y2 = (
            y + ltrb[:, 3]
        )

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

    # ========================================================
    # Main forward
    # ========================================================

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Sequence[
            Dict[str, torch.Tensor]
        ],
    ) -> Dict[str, torch.Tensor]:
        """
        计算 FCOS 总损失。

        predictions：

            {
                "cls_logits": [
                    P3,
                    P4,
                    P5,
                    P6,
                ],

                "bbox_reg": [
                    P3,
                    P4,
                    P5,
                    P6,
                ],

                "centerness": [
                    P3,
                    P4,
                    P5,
                    P6,
                ],
            }

        targets：

            [
                {
                    "boxes": Tensor[N,4],
                    "labels": Tensor[N],
                },
                ...
            ]

        返回：

            {
                "loss":
                    total loss,

                "cls_loss":
                    classification loss,

                "bbox_loss":
                    GIoU loss,

                "centerness_loss":
                    centerness loss,

                "num_positive":
                    positive sample count
            }
        """

        cls_outputs = predictions[
            "cls_logits"
        ]

        bbox_outputs = predictions[
            "bbox_reg"
        ]

        center_outputs = predictions[
            "centerness"
        ]

        if not (
            len(cls_outputs)
            == len(bbox_outputs)
            == len(center_outputs)
            == len(self.strides)
        ):
            raise ValueError(
                "FCOS predictions level 数量错误。"
            )

        # ----------------------------------------------------
        # Build targets
        # ----------------------------------------------------

        target_dict = self.build_targets(
            cls_outputs=cls_outputs,
            targets=targets,
        )

        points_per_level = target_dict[
            "points_per_level"
        ]

        total_cls_loss = (
            cls_outputs[0].sum() * 0.0
        )

        total_bbox_loss = (
            bbox_outputs[0].sum() * 0.0
        )

        total_center_loss = (
            center_outputs[0].sum() * 0.0
        )

        total_positive = (
            cls_outputs[0].new_tensor(
                0.0
            )
        )

        # ----------------------------------------------------
        # Process each level
        # ----------------------------------------------------

        for level_index in range(
            len(self.strides)
        ):

            cls_logits = (
                cls_outputs[level_index]
            )

            bbox_reg = (
                bbox_outputs[level_index]
            )

            center_logits = (
                center_outputs[level_index]
            )

            batch_size = (
                cls_logits.shape[0]
            )

            height = (
                cls_logits.shape[2]
            )

            width = (
                cls_logits.shape[3]
            )

            num_points = (
                height * width
            )

            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            cls_pred = (
                cls_logits.permute(
                    0,
                    2,
                    3,
                    1,
                )
                .contiguous()
                .view(
                    batch_size,
                    num_points,
                    self.num_classes,
                )
            )

            bbox_pred = (
                bbox_reg.permute(
                    0,
                    2,
                    3,
                    1,
                )
                .contiguous()
                .view(
                    batch_size,
                    num_points,
                    4,
                )
            )

            center_pred = (
                center_logits.permute(
                    0,
                    2,
                    3,
                    1,
                )
                .contiguous()
                .view(
                    batch_size,
                    num_points,
                )
            )

            # ------------------------------------------------
            # Targets
            # ------------------------------------------------

            labels = torch.stack(
                target_dict[
                    "labels"
                ][level_index],
                dim=0,
            )

            bbox_targets = torch.stack(
                target_dict[
                    "bbox_targets"
                ][level_index],
                dim=0,
            )

            center_targets = torch.stack(
                target_dict[
                    "centerness_targets"
                ][level_index],
                dim=0,
            )

            positive_mask = torch.stack(
                target_dict[
                    "positive_masks"
                ][level_index],
                dim=0,
            )

            # ------------------------------------------------
            # Classification target
            # ------------------------------------------------

            cls_targets = torch.zeros(
                (
                    batch_size,
                    num_points,
                    self.num_classes,
                ),
                dtype=cls_pred.dtype,
                device=cls_pred.device,
            )

            positive_indices = (
                positive_mask
            )

            if positive_indices.any():

                positive_labels = labels[
                    positive_indices
                ]

                if (
                    positive_labels.min() < 0
                    or positive_labels.max()
                    >= self.num_classes
                ):
                    raise ValueError(
                        "GT label 超出类别范围。"
                    )

                cls_targets[
                    positive_indices,
                    positive_labels,
                ] = 1.0

            # ------------------------------------------------
            # Classification loss
            # ------------------------------------------------

            cls_loss = self.focal_loss(
                cls_pred.reshape(
                    -1,
                    self.num_classes,
                ),
                cls_targets.reshape(
                    -1,
                    self.num_classes,
                ),
            )

            # ------------------------------------------------
            # Positive samples
            # ------------------------------------------------

            positive_flat = (
                positive_mask.reshape(-1)
            )

            num_positive = (
                positive_flat.sum()
            )

            total_positive = (
                total_positive
                + num_positive.float()
            )

            # ------------------------------------------------
            # Bbox + centerness
            # ------------------------------------------------

            if num_positive > 0:

                bbox_pred_positive = (
                    bbox_pred.reshape(
                        -1,
                        4,
                    )[positive_flat]
                )

                bbox_target_positive = (
                    bbox_targets.reshape(
                        -1,
                        4,
                    )[positive_flat]
                )

                center_target_positive = (
                    center_targets.reshape(
                        -1
                    )[positive_flat]
                )

                # ------------------------------------------------
                # Positive points
                # ------------------------------------------------

                points = points_per_level[
                    level_index
                ]

                # Batch 中每张图片使用相同的
                # feature point 坐标。
                points_batch = (
                    points.unsqueeze(0)
                    .expand(
                        batch_size,
                        -1,
                        -1,
                    )
                    .reshape(
                        -1,
                        2,
                    )
                )

                points_positive = (
                    points_batch[
                        positive_flat
                    ]
                )

                # ------------------------------------------------
                # Decode prediction
                # ------------------------------------------------

                pred_boxes = (
                    self.decode_ltrb(
                        points_positive,
                        bbox_pred_positive,
                    )
                )

                # ------------------------------------------------
                # Decode target
                # ------------------------------------------------

                target_boxes = (
                    self.decode_ltrb(
                        points_positive,
                        bbox_target_positive,
                    )
                )

                # ------------------------------------------------
                # GIoU
                # ------------------------------------------------

                giou_loss = self.bbox_loss(
                    pred_boxes,
                    target_boxes,
                )

                # ------------------------------------------------
                # Centerness
                # ------------------------------------------------

                center_pred_positive = (
                    center_pred.reshape(
                        -1
                    )[positive_flat]
                )

                center_loss = (
                    F.binary_cross_entropy_with_logits(
                        center_pred_positive,
                        center_target_positive,
                        reduction="sum",
                    )
                )

                total_bbox_loss = (
                    total_bbox_loss
                    + giou_loss
                )

                total_center_loss = (
                    total_center_loss
                    + center_loss
                )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            total_cls_loss = (
                total_cls_loss
                + cls_loss
            )

        # ====================================================
        # Normalization
        # ====================================================

        if self.normalize_by_positive:

            normalizer = (
                total_positive.clamp(
                    min=1.0
                )
            )

        else:

            normalizer = cls_outputs[
                0
            ].shape[0]

            normalizer = cls_outputs[
                0
            ].new_tensor(
                float(normalizer)
            )

        total_cls_loss = (
            total_cls_loss
            / normalizer
        )

        total_bbox_loss = (
            total_bbox_loss
            / normalizer
        )

        total_center_loss = (
            total_center_loss
            / normalizer
        )

        # ====================================================
        # Weighted total loss
        # ====================================================

        total_loss = (
            self.classification_loss_weight
            * total_cls_loss
            +
            self.bbox_loss_weight
            * total_bbox_loss
            +
            self.centerness_loss_weight
            * total_center_loss
        )

        return {
            "loss": total_loss,
            "cls_loss": total_cls_loss,
            "bbox_loss": total_bbox_loss,
            "centerness_loss": total_center_loss,
            "num_positive": total_positive,
        }


# ============================================================
# Factory
# ============================================================

def build_fcos_loss(
    num_classes: int = 12,
    strides: Sequence[int] = (
        8,
        16,
        32,
        64,
    ),
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    bbox_loss_weight: float = 1.0,
    centerness_loss_weight: float = 1.0,
    classification_loss_weight: float = 1.0,
    normalize_by_positive: bool = True,
) -> FCOSLoss:
    """
    构建 FCOS Loss。

    与 config.py 默认设置对应。
    """

    return FCOSLoss(
        num_classes=num_classes,
        strides=strides,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        bbox_loss_weight=bbox_loss_weight,
        centerness_loss_weight=centerness_loss_weight,
        classification_loss_weight=classification_loss_weight,
        normalize_by_positive=normalize_by_positive,
    )


# ============================================================
# Test utilities
# ============================================================

def create_dummy_predictions(
    batch_size: int = 2,
    num_classes: int = 12,
    device: str = "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """
    创建模拟 FCOS 输出。
    """

    shapes = [
        (80, 80),
        (40, 40),
        (20, 20),
        (10, 10),
    ]

    cls_outputs = []
    bbox_outputs = []
    center_outputs = []

    for height, width in shapes:

        cls_outputs.append(
            torch.randn(
                batch_size,
                num_classes,
                height,
                width,
                device=device,
                requires_grad=True,
            )
        )

        bbox_outputs.append(
            F.relu(
                torch.randn(
                    batch_size,
                    4,
                    height,
                    width,
                    device=device,
                    requires_grad=True,
                )
            )
        )

        center_outputs.append(
            torch.randn(
                batch_size,
                1,
                height,
                width,
                device=device,
                requires_grad=True,
            )
        )

    return {
        "cls_logits": cls_outputs,
        "bbox_reg": bbox_outputs,
        "centerness": center_outputs,
    }


def create_dummy_targets(
    batch_size: int = 2,
    num_classes: int = 12,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """
    创建模拟 GT。

    每张图片创建两个 connector。
    """

    targets = []

    for batch_index in range(
        batch_size
    ):

        if batch_index == 0:

            boxes = torch.tensor(
                [
                    [
                        120.0,
                        150.0,
                        240.0,
                        300.0,
                    ],
                    [
                        350.0,
                        350.0,
                        500.0,
                        500.0,
                    ],
                ],
                dtype=torch.float32,
                device=device,
            )

            labels = torch.tensor(
                [
                    0,
                    5,
                ],
                dtype=torch.long,
                device=device,
            )

        else:

            boxes = torch.tensor(
                [
                    [
                        100.0,
                        100.0,
                        180.0,
                        200.0,
                    ],
                    [
                        400.0,
                        250.0,
                        560.0,
                        420.0,
                    ],
                ],
                dtype=torch.float32,
                device=device,
            )

            labels = torch.tensor(
                [
                    2,
                    9,
                ],
                dtype=torch.long,
                device=device,
            )

        targets.append(
            {
                "boxes": boxes,
                "labels": labels,
            }
        )

    return targets


# ============================================================
# Unit test
# ============================================================

def test_fcos_loss():
    """
    FCOS Loss 完整测试。

    测试：

        1. target assignment
        2. classification loss
        3. GIoU loss
        4. centerness loss
        5. total loss
        6. backward
        7. 无目标图像
        8. 数值稳定性
    """

    print("\n")
    print("=" * 70)
    print("Testing FCOS Loss")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Build loss
    # --------------------------------------------------------

    criterion = build_fcos_loss(
        num_classes=12,
        strides=(
            8,
            16,
            32,
            64,
        ),
        focal_alpha=0.25,
        focal_gamma=2.0,
        bbox_loss_weight=1.0,
        centerness_loss_weight=1.0,
        classification_loss_weight=1.0,
        normalize_by_positive=True,
    ).to(device)

    # --------------------------------------------------------
    # Dummy prediction
    # --------------------------------------------------------

    predictions = create_dummy_predictions(
        batch_size=2,
        num_classes=12,
        device=device,
    )

    # --------------------------------------------------------
    # Dummy target
    # --------------------------------------------------------

    targets = create_dummy_targets(
        batch_size=2,
        num_classes=12,
        device=device,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    loss_dict = criterion(
        predictions,
        targets,
    )

    print("\nLoss values:")

    for key, value in loss_dict.items():

        print(
            f"  {key}: "
            f"{value.detach().item():.6f}"
        )

    # --------------------------------------------------------
    # Numerical check
    # --------------------------------------------------------

    for key, value in loss_dict.items():

        if not torch.isfinite(
            value
        ).all():

            raise RuntimeError(
                f"{key} 存在 NaN 或 Inf。"
            )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss_dict[
        "loss"
    ].backward()

    print(
        "\nBackward test: OK"
    )

    # --------------------------------------------------------
    # Gradient test
    # --------------------------------------------------------

    for level_index in range(4):

        cls_grad = predictions[
            "cls_logits"
        ][level_index].grad

        center_grad = predictions[
            "centerness"
        ][level_index].grad

        if cls_grad is None:

            raise RuntimeError(
                f"P{level_index + 3} "
                "classification 没有梯度。"
            )

        if center_grad is None:

            raise RuntimeError(
                f"P{level_index + 3} "
                "centerness 没有梯度。"
            )

    print(
        "Gradient test: OK"
    )

    # --------------------------------------------------------
    # Empty GT test
    # --------------------------------------------------------

    empty_targets = [
        {
            "boxes": torch.empty(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
                device=device,
            ),
            "labels": torch.empty(
                (
                    0,
                ),
                dtype=torch.long,
                device=device,
            ),
        },
        {
            "boxes": torch.empty(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
                device=device,
            ),
            "labels": torch.empty(
                (
                    0,
                ),
                dtype=torch.long,
                device=device,
            ),
        },
    ]

    predictions_empty = (
        create_dummy_predictions(
            batch_size=2,
            num_classes=12,
            device=device,
        )
    )

    empty_loss = criterion(
        predictions_empty,
        empty_targets,
    )

    print(
        "\nEmpty-GT test:"
    )

    print(
        f"  loss = "
        f"{empty_loss['loss'].item():.6f}"
    )

    print(
        f"  positive = "
        f"{empty_loss['num_positive'].item():.0f}"
    )

    if not torch.isfinite(
        empty_loss["loss"]
    ):

        raise RuntimeError(
            "无 GT 情况下 loss 出现 NaN/Inf。"
        )

    assert (
        empty_loss["num_positive"].item()
        == 0
    )

    # --------------------------------------------------------
    # Centerness test
    # --------------------------------------------------------

    test_ltrb = torch.tensor(
        [
            [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            [
                5.0,
                10.0,
                5.0,
                10.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    center = compute_centerness(
        test_ltrb
    )

    print(
        "\nCenterness test:"
    )

    print(
        center
    )

    assert torch.all(
        center <= 1.0
    )

    assert torch.all(
        center >= 0.0
    )

    # --------------------------------------------------------
    # GIoU test
    # --------------------------------------------------------

    boxes_a = torch.tensor(
        [
            [
                10.0,
                10.0,
                50.0,
                50.0,
            ],
            [
                20.0,
                20.0,
                60.0,
                60.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    boxes_b = torch.tensor(
        [
            [
                10.0,
                10.0,
                50.0,
                50.0,
            ],
            [
                30.0,
                30.0,
                70.0,
                70.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    giou = generalized_box_iou(
        boxes_a,
        boxes_b,
    )

    print(
        "\nGIoU test:"
    )

    print(
        giou
    )

    assert torch.all(
        giou <= 1.0 + EPS
    )

    print("\n")
    print("=" * 70)
    print("FCOS Loss test passed.")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_fcos_loss()