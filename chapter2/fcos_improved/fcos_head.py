# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict, List, Tuple, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Type definitions
# ============================================================

Tensor = torch.Tensor
FeatureInput = Union[
    List[Tensor],
    Tuple[Tensor, ...],
    Dict[str, Tensor]
]


# ============================================================
# Basic Conv Block
# ============================================================

class ConvBlock(nn.Module):
    """
    FCOS Head 基础卷积模块。

    结构：

        Conv2d
          ↓
        Normalization
          ↓
        ReLU

    默认使用 GroupNorm，而不是 BatchNorm。

    对于目标检测中的小 batch 训练，
    GroupNorm 通常比 BatchNorm 更不依赖 batch size。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=not norm,
        )

        self.norm = None

        if norm:
            # 保证 GroupNorm 的 group 数可以整除通道数。
            groups = min(num_groups, out_channels)

            while out_channels % groups != 0 and groups > 1:
                groups -= 1

            self.norm = nn.GroupNorm(
                num_groups=groups,
                num_channels=out_channels,
            )

        self.activation = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(
            self.conv.weight,
            mean=0.0,
            std=0.01,
        )

        if self.conv.bias is not None:
            nn.init.constant_(
                self.conv.bias,
                0.0,
            )

        if self.norm is not None:
            nn.init.constant_(
                self.norm.weight,
                1.0,
            )
            nn.init.constant_(
                self.norm.bias,
                0.0,
            )

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        x = self.activation(x)

        return x


# ============================================================
# Scale
# ============================================================

class Scale(nn.Module):
    """
    FCOS 每个特征层独立的可学习尺度参数。

    FCOS 中不同 FPN level 对应不同目标尺寸范围，
    使用独立 Scale 对 bbox regression 进行尺度调整。

    输出：

        y = x * exp(scale)

    初始时 scale = 0，
    因此：

        exp(0) = 1
    """

    def __init__(self, init_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(
                float(init_value),
                dtype=torch.float32,
            )
        )

    def forward(self, x: Tensor) -> Tensor:

        return x * self.scale.exp()


# ============================================================
# FCOS Head
# ============================================================

class FCOSHead(nn.Module):
    """
    FCOS 检测头。

    参数：

        num_classes:
            类别数量。

        in_channels:
            输入 FPN 特征通道数。

        feat_channels:
            FCOS head 内部特征通道数。

        num_convs:
            分类塔和回归塔中的卷积层数量。

        num_levels:
            FPN level 数量。

        strides:
            P3-P6 对应的 stride。

        norm:
            是否使用 GroupNorm。

        centerness:
            是否输出 centerness。

        use_scale:
            是否使用每层独立 Scale。

        prior_prob:
            分类 bias 初始化对应的先验概率。
    """

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_convs: int = 4,
        num_levels: int = 4,
        strides: Tuple[int, ...] = (8, 16, 32, 64),
        norm: bool = True,
        centerness: bool = True,
        use_scale: bool = True,
        prior_prob: float = 0.01,
        num_groups: int = 32,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes 必须大于 0。"
            )

        if in_channels <= 0:
            raise ValueError(
                "in_channels 必须大于 0。"
            )

        if feat_channels <= 0:
            raise ValueError(
                "feat_channels 必须大于 0。"
            )

        if num_convs <= 0:
            raise ValueError(
                "num_convs 必须大于 0。"
            )

        if num_levels <= 0:
            raise ValueError(
                "num_levels 必须大于 0。"
            )

        if len(strides) != num_levels:
            raise ValueError(
                "strides 数量必须与 num_levels 一致。"
            )

        if not 0.0 < prior_prob < 1.0:
            raise ValueError(
                "prior_prob 必须位于 (0, 1) 范围内。"
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.num_levels = num_levels
        self.strides = tuple(strides)
        self.use_centerness = centerness
        self.use_scale = use_scale
        self.prior_prob = prior_prob

        # ----------------------------------------------------
        # Classification tower
        # ----------------------------------------------------

        cls_tower = []

        for i in range(num_convs):

            cls_tower.append(
                ConvBlock(
                    in_channels=(
                        in_channels
                        if i == 0
                        else feat_channels
                    ),
                    out_channels=feat_channels,
                    kernel_size=3,
                    padding=1,
                    norm=norm,
                    num_groups=num_groups,
                )
            )

        self.cls_tower = nn.Sequential(
            *cls_tower
        )

        # ----------------------------------------------------
        # Regression tower
        # ----------------------------------------------------

        bbox_tower = []

        for i in range(num_convs):

            bbox_tower.append(
                ConvBlock(
                    in_channels=(
                        in_channels
                        if i == 0
                        else feat_channels
                    ),
                    out_channels=feat_channels,
                    kernel_size=3,
                    padding=1,
                    norm=norm,
                    num_groups=num_groups,
                )
            )

        self.bbox_tower = nn.Sequential(
            *bbox_tower
        )

        # ----------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------

        self.cls_logits = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bbox_pred = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if centerness:
            self.centerness = nn.Conv2d(
                in_channels=feat_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.centerness = None

        # ----------------------------------------------------
        # Per-level Scale
        # ----------------------------------------------------

        if use_scale:
            self.scales = nn.ModuleList(
                [
                    Scale(init_value=0.0)
                    for _ in range(num_levels)
                ]
            )
        else:
            self.scales = None

        # ----------------------------------------------------
        # Output activation
        # ----------------------------------------------------

        # FCOS bbox regression要求非负，
        # 因此在最终输出阶段使用 ReLU。
        self.bbox_activation = nn.ReLU(inplace=True)

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        self._init_prediction_layers()

    # ========================================================
    # Initialization
    # ========================================================

    def _init_prediction_layers(self):
        """
        初始化最终预测层。

        分类 bias：

            b = -log((1-p)/p)

        当 p = 0.01 时：

            b ≈ -4.595

        可以降低训练初期大量背景位置产生的
        正类别响应。
        """

        prior_bias = -math.log(
            (1.0 - self.prior_prob)
            / self.prior_prob
        )

        nn.init.normal_(
            self.cls_logits.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.cls_logits.bias,
            prior_bias,
        )

        nn.init.normal_(
            self.bbox_pred.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.bbox_pred.bias,
            0.0,
        )

        if self.centerness is not None:

            nn.init.normal_(
                self.centerness.weight,
                mean=0.0,
                std=0.01,
            )

            nn.init.constant_(
                self.centerness.bias,
                0.0,
            )

    # ========================================================
    # Input processing
    # ========================================================

    def _normalize_features(
        self,
        features: FeatureInput,
    ) -> List[Tensor]:
        """
        将不同形式的 FPN 输出统一转换为：

            [P3, P4, P5, P6]

        支持：

            list
            tuple
            dict
        """

        if isinstance(features, dict):

            required_names = [
                "P3",
                "P4",
                "P5",
                "P6",
            ]

            missing = [
                name
                for name in required_names
                if name not in features
            ]

            if missing:
                raise KeyError(
                    f"FPN 输出缺少特征层: {missing}"
                )

            features = [
                features[name]
                for name in required_names
            ]

        elif isinstance(features, (list, tuple)):

            features = list(features)

        else:

            raise TypeError(
                "features 必须是 list、tuple 或 dict。"
            )

        if len(features) != self.num_levels:

            raise ValueError(
                f"期望 {self.num_levels} 个 FPN 特征层，"
                f"实际得到 {len(features)} 个。"
            )

        for i, feature in enumerate(features):

            if not isinstance(feature, torch.Tensor):

                raise TypeError(
                    f"第 {i} 个特征不是 torch.Tensor。"
                )

            if feature.ndim != 4:

                raise ValueError(
                    f"第 {i} 个特征应为四维张量 "
                    f"[B,C,H,W]，"
                    f"实际为 {feature.shape}。"
                )

            if feature.shape[1] != self.in_channels:

                raise ValueError(
                    f"第 {i} 个特征通道数错误："
                    f"期望 {self.in_channels}，"
                    f"实际 {feature.shape[1]}。"
                )

        return features

    # ========================================================
    # Single level forward
    # ========================================================

    def _forward_single(
        self,
        feature: Tensor,
        level_index: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        对单个 FPN level 进行预测。

        返回：

            cls_logits
            bbox_reg
            centerness
        """

        # ----------------------------------------------------
        # Classification branch
        # ----------------------------------------------------

        cls_feature = self.cls_tower(feature)

        cls_logits = self.cls_logits(
            cls_feature
        )

        # ----------------------------------------------------
        # Regression branch
        # ----------------------------------------------------

        bbox_feature = self.bbox_tower(feature)

        bbox_reg = self.bbox_pred(
            bbox_feature
        )

        # FCOS l/t/r/b 必须为非负值。
        bbox_reg = self.bbox_activation(
            bbox_reg
        )

        # 每一个 FPN level 使用独立 Scale。
        if self.scales is not None:

            bbox_reg = self.scales[level_index](
                bbox_reg
            )

        # ----------------------------------------------------
        # Centerness branch
        # ----------------------------------------------------

        if self.centerness is not None:

            centerness = self.centerness(
                bbox_feature
            )

        else:

            centerness = None

        return (
            cls_logits,
            bbox_reg,
            centerness,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        features: FeatureInput,
    ) -> Dict[str, List[Tensor]]:
        """
        前向传播。

        输入：

            features:
                [P3, P4, P5, P6]

        输出：

            {
                "cls_logits": [...],
                "bbox_reg": [...],
                "centerness": [...]
            }

        每个 list 包含四个 FPN level。
        """

        features = self._normalize_features(
            features
        )

        cls_outputs = []
        bbox_outputs = []
        centerness_outputs = []

        for level_index, feature in enumerate(features):

            (
                cls_logits,
                bbox_reg,
                centerness,
            ) = self._forward_single(
                feature=feature,
                level_index=level_index,
            )

            cls_outputs.append(
                cls_logits
            )

            bbox_outputs.append(
                bbox_reg
            )

            centerness_outputs.append(
                centerness
            )

        return {
            "cls_logits": cls_outputs,
            "bbox_reg": bbox_outputs,
            "centerness": centerness_outputs,
        }

    # ========================================================
    # Flatten predictions
    # ========================================================

    def flatten_predictions(
        self,
        outputs: Dict[str, List[Tensor]],
    ) -> Dict[str, Tensor]:
        """
        将不同 FPN level 的输出展平。

        例如：

            P3:
                [B, C, 80, 80]

            P4:
                [B, C, 40, 40]

        转换为：

            [B, 6400, C]
            [B, 1600, C]

        最终拼接：

            [B, 8500, C]

        对于 640×640 输入：

            80×80
          + 40×40
          + 20×20
          + 10×10
          = 8500 points
        """

        cls_outputs = outputs["cls_logits"]
        bbox_outputs = outputs["bbox_reg"]
        centerness_outputs = outputs["centerness"]

        flat_cls = []
        flat_bbox = []
        flat_centerness = []

        for cls_pred, bbox_pred, center_pred in zip(
            cls_outputs,
            bbox_outputs,
            centerness_outputs,
        ):

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            cls_pred = cls_pred.permute(
                0, 2, 3, 1
            ).contiguous()

            cls_pred = cls_pred.view(
                cls_pred.shape[0],
                -1,
                self.num_classes,
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox_pred = bbox_pred.permute(
                0, 2, 3, 1
            ).contiguous()

            bbox_pred = bbox_pred.view(
                bbox_pred.shape[0],
                -1,
                4,
            )

            # ------------------------------------------------
            # Centerness
            # ------------------------------------------------

            if center_pred is not None:

                center_pred = center_pred.permute(
                    0, 2, 3, 1
                ).contiguous()

                center_pred = center_pred.view(
                    center_pred.shape[0],
                    -1,
                    1,
                )

            flat_cls.append(
                cls_pred
            )

            flat_bbox.append(
                bbox_pred
            )

            if center_pred is not None:

                flat_centerness.append(
                    center_pred
                )

        flat_cls = torch.cat(
            flat_cls,
            dim=1,
        )

        flat_bbox = torch.cat(
            flat_bbox,
            dim=1,
        )

        if len(flat_centerness) > 0:

            flat_centerness = torch.cat(
                flat_centerness,
                dim=1,
            )

        else:

            flat_centerness = None

        return {
            "cls_logits": flat_cls,
            "bbox_reg": flat_bbox,
            "centerness": flat_centerness,
        }

    # ========================================================
    # Generate FCOS points
    # ========================================================

    def generate_points(
        self,
        features: FeatureInput,
    ) -> List[Tensor]:
        """
        为 P3-P6 生成 FCOS reference points。

        每个点对应 feature map 中一个像素位置。

        坐标使用输入图像尺度：

            x = (j + 0.5) * stride
            y = (i + 0.5) * stride

        返回：

            [
                [H3*W3, 2],
                [H4*W4, 2],
                [H5*W5, 2],
                [H6*W6, 2]
            ]

        坐标顺序：

            [x, y]
        """

        features = self._normalize_features(
            features
        )

        points = []

        for feature, stride in zip(
            features,
            self.strides,
        ):

            _, _, height, width = (
                feature.shape
            )

            device = feature.device
            dtype = feature.dtype

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

            level_points = torch.stack(
                [
                    xx.reshape(-1),
                    yy.reshape(-1),
                ],
                dim=-1,
            )

            points.append(
                level_points
            )

        return points

    # ========================================================
    # Get all points
    # ========================================================

    def generate_points_with_stride(
        self,
        features: FeatureInput,
    ) -> Tuple[Tensor, Tensor]:
        """
        生成所有 FPN level 的 points 和 stride。

        返回：

            points:
                [N, 2]

            strides:
                [N]

        对 640×640 输入：

            N = 8500
        """

        features = self._normalize_features(
            features
        )

        points_list = self.generate_points(
            features
        )

        stride_list = []

        for feature, stride in zip(
            features,
            self.strides,
        ):

            _, _, height, width = (
                feature.shape
            )

            num_points = height * width

            stride_tensor = torch.full(
                (num_points,),
                float(stride),
                dtype=feature.dtype,
                device=feature.device,
            )

            stride_list.append(
                stride_tensor
            )

        points = torch.cat(
            points_list,
            dim=0,
        )

        strides = torch.cat(
            stride_list,
            dim=0,
        )

        return points, strides

    # ========================================================
    # Decode ltrb
    # ========================================================

    @staticmethod
    def decode_boxes(
        points: Tensor,
        bbox_reg: Tensor,
    ) -> Tensor:
        """
        将 FCOS l/t/r/b 转换为：

            x1, y1, x2, y2

        输入：

            points:
                [N, 2]

            bbox_reg:
                [B, N, 4]

        输出：

            boxes:
                [B, N, 4]
        """

        if points.ndim != 2:
            raise ValueError(
                "points 应为 [N, 2]。"
            )

        if bbox_reg.ndim != 3:
            raise ValueError(
                "bbox_reg 应为 [B, N, 4]。"
            )

        if bbox_reg.shape[-1] != 4:
            raise ValueError(
                "bbox_reg 最后一维必须为 4。"
            )

        if points.shape[0] != bbox_reg.shape[1]:
            raise ValueError(
                "points 数量与 bbox_reg 数量不一致。"
            )

        points = points.to(
            device=bbox_reg.device,
            dtype=bbox_reg.dtype,
        )

        x = points[:, 0]
        y = points[:, 1]

        left = bbox_reg[:, :, 0]
        top = bbox_reg[:, :, 1]
        right = bbox_reg[:, :, 2]
        bottom = bbox_reg[:, :, 3]

        x1 = x.unsqueeze(0) - left
        y1 = y.unsqueeze(0) - top

        x2 = x.unsqueeze(0) + right
        y2 = y.unsqueeze(0) + bottom

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

    # ========================================================
    # Export helper
    # ========================================================

    def forward_flatten(
        self,
        features: FeatureInput,
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        一次完成：

            FCOS Head
                ↓
            flatten
                ↓
            generate points

        返回：

            cls_logits
            bbox_reg
            centerness
            points
            strides
        """

        outputs = self.forward(
            features
        )

        flat_outputs = self.flatten_predictions(
            outputs
        )

        points, strides = (
            self.generate_points_with_stride(
                features
            )
        )

        flat_outputs["points"] = points
        flat_outputs["strides"] = strides

        return flat_outputs


# ============================================================
# Factory function
# ============================================================

def build_fcos_head(
    num_classes: int = 12,
    in_channels: int = 256,
    feat_channels: int = 256,
    num_convs: int = 4,
    num_levels: int = 4,
    strides: Tuple[int, ...] = (8, 16, 32, 64),
    norm: bool = True,
    centerness: bool = True,
    use_scale: bool = True,
    prior_prob: float = 0.01,
) -> FCOSHead:
    """
    构建 FCOS Head。

    默认配置与 config.py 对应：

        num_classes = 12
        in_channels = 256
        feat_channels = 256
        num_convs = 4
        strides = [8,16,32,64]
        centerness = True
        scale = True
    """

    return FCOSHead(
        num_classes=num_classes,
        in_channels=in_channels,
        feat_channels=feat_channels,
        num_convs=num_convs,
        num_levels=num_levels,
        strides=strides,
        norm=norm,
        centerness=centerness,
        use_scale=use_scale,
        prior_prob=prior_prob,
    )


# ============================================================
# Parameter count
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:
    """
    统计可训练参数数量。
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Feature shape printer
# ============================================================

def print_output_shapes(
    outputs: Dict[str, List[Tensor]],
):
    """
    打印 FCOS Head 各层输出尺寸。
    """

    print("\n" + "=" * 70)
    print("FCOS Head 输出尺寸")
    print("=" * 70)

    cls_outputs = outputs["cls_logits"]
    bbox_outputs = outputs["bbox_reg"]
    center_outputs = outputs["centerness"]

    for i in range(len(cls_outputs)):

        print(
            f"P{i + 3}:"
        )

        print(
            f"  cls_logits   : "
            f"{tuple(cls_outputs[i].shape)}"
        )

        print(
            f"  bbox_reg     : "
            f"{tuple(bbox_outputs[i].shape)}"
        )

        if center_outputs[i] is not None:

            print(
                f"  centerness   : "
                f"{tuple(center_outputs[i].shape)}"
            )

    print("=" * 70)


# ============================================================
# Unit test
# ============================================================

def test_fcos_head():
    """
    FCOS Head 独立测试。

    输入：

        640×640

    预期：

        P3 = [2,256,80,80]
        P4 = [2,256,40,40]
        P5 = [2,256,20,20]
        P6 = [2,256,10,10]

    输出：

        cls:
            [2,12,H,W]

        bbox:
            [2,4,H,W]

        centerness:
            [2,1,H,W]

    flatten：

        [2,8500,12]
        [2,8500,4]
        [2,8500,1]
    """

    print("\n")
    print("=" * 70)
    print("Testing FCOS Head")
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
    # Model
    # --------------------------------------------------------

    model = build_fcos_head(
        num_classes=12,
        in_channels=256,
        feat_channels=256,
        num_convs=4,
        num_levels=4,
        strides=(8, 16, 32, 64),
        norm=True,
        centerness=True,
        use_scale=True,
        prior_prob=0.01,
    ).to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Fake FPN features
    # --------------------------------------------------------

    features = [
        torch.randn(
            2,
            256,
            80,
            80,
            device=device,
        ),

        torch.randn(
            2,
            256,
            40,
            40,
            device=device,
        ),

        torch.randn(
            2,
            256,
            20,
            20,
            device=device,
        ),

        torch.randn(
            2,
            256,
            10,
            10,
            device=device,
        ),
    ]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(
            features
        )

    print_output_shapes(
        outputs
    )

    # --------------------------------------------------------
    # Shape assertions
    # --------------------------------------------------------

    expected_cls_shapes = [
        (2, 12, 80, 80),
        (2, 12, 40, 40),
        (2, 12, 20, 20),
        (2, 12, 10, 10),
    ]

    expected_bbox_shapes = [
        (2, 4, 80, 80),
        (2, 4, 40, 40),
        (2, 4, 20, 20),
        (2, 4, 10, 10),
    ]

    expected_center_shapes = [
        (2, 1, 80, 80),
        (2, 1, 40, 40),
        (2, 1, 20, 20),
        (2, 1, 10, 10),
    ]

    for i in range(4):

        assert (
            outputs["cls_logits"][i].shape
            == expected_cls_shapes[i]
        )

        assert (
            outputs["bbox_reg"][i].shape
            == expected_bbox_shapes[i]
        )

        assert (
            outputs["centerness"][i].shape
            == expected_center_shapes[i]
        )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flat = model.flatten_predictions(
        outputs
    )

    print("\nFlattened outputs:")

    print(
        "  cls_logits:",
        tuple(flat["cls_logits"].shape)
    )

    print(
        "  bbox_reg:",
        tuple(flat["bbox_reg"].shape)
    )

    print(
        "  centerness:",
        tuple(flat["centerness"].shape)
    )

    assert (
        flat["cls_logits"].shape
        == (2, 8500, 12)
    )

    assert (
        flat["bbox_reg"].shape
        == (2, 8500, 4)
    )

    assert (
        flat["centerness"].shape
        == (2, 8500, 1)
    )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    points, strides = (
        model.generate_points_with_stride(
            features
        )
    )

    print("\nPoint information:")

    print(
        "  points:",
        tuple(points.shape)
    )

    print(
        "  strides:",
        tuple(strides.shape)
    )

    assert points.shape == (
        8500,
        2,
    )

    assert strides.shape == (
        8500,
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    boxes = FCOSHead.decode_boxes(
        points=points,
        bbox_reg=flat["bbox_reg"],
    )

    print(
        "  decoded boxes:",
        tuple(boxes.shape)
    )

    assert boxes.shape == (
        2,
        8500,
        4,
    )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    for name, tensor in flat.items():

        if tensor is None:
            continue

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} 存在 NaN 或 Inf。"
            )

    # bbox regression must be non-negative
    assert (
        flat["bbox_reg"] >= 0
    ).all()

    # --------------------------------------------------------
    # Gradient test
    # --------------------------------------------------------

    model.train()

    train_features = [
        feature.clone().requires_grad_(True)
        for feature in features
    ]

    train_outputs = model(
        train_features
    )

    loss = 0.0

    for tensor in train_outputs["cls_logits"]:
        loss = loss + tensor.mean()

    for tensor in train_outputs["bbox_reg"]:
        loss = loss + tensor.mean()

    for tensor in train_outputs["centerness"]:
        loss = loss + tensor.mean()

    loss.backward()

    print(
        "\nGradient test:"
    )

    for i, feature in enumerate(
        train_features
    ):

        assert feature.grad is not None

        print(
            f"  P{i + 3}: "
            f"gradient OK"
        )

    print("\n")
    print("=" * 70)
    print("FCOS Head test passed.")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_fcos_head()