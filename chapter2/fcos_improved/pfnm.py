# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 工艺语义编码器
# ============================================================

class ProcessSemanticEncoder(nn.Module):
    """
    工艺语义编码器。

    将 16 维工艺信息映射到高维工艺语义空间。

    输入：
        process_info:
            [B, 16]

    输出：
        process_embedding:
            [B, hidden_dim]

    默认：

        16
         ↓
        128
         ↓
        256
    """

    def __init__(
        self,
        process_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        if process_dim <= 0:
            raise ValueError(
                "process_dim 必须大于 0。"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim 必须大于 0。"
            )

        intermediate_dim = max(
            hidden_dim // 2,
            32,
        )

        layers = [
            nn.Linear(
                process_dim,
                intermediate_dim,
            ),

            nn.LayerNorm(
                intermediate_dim
            ),

            nn.GELU(),
        ]

        if dropout > 0.0:
            layers.append(
                nn.Dropout(dropout)
            )

        layers.extend(
            [
                nn.Linear(
                    intermediate_dim,
                    hidden_dim,
                ),

                nn.LayerNorm(
                    hidden_dim
                ),

                nn.GELU(),
            ]
        )

        if dropout > 0.0:
            layers.append(
                nn.Dropout(dropout)
            )

        self.encoder = nn.Sequential(
            *layers
        )

    def forward(
        self,
        process_info: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            process_info:
                [B, D]

        返回：
            [B, hidden_dim]
        """

        if not isinstance(
            process_info,
            torch.Tensor,
        ):
            raise TypeError(
                "process_info 必须为 torch.Tensor。"
            )

        if process_info.ndim == 1:
            process_info = process_info.unsqueeze(0)

        if process_info.ndim != 2:
            raise ValueError(
                "process_info 必须为 [B, D]。"
            )

        return self.encoder(
            process_info
        )


# ============================================================
# 工艺条件通道调制
# ============================================================

class ProcessChannelModulation(nn.Module):
    """
    工艺条件驱动的通道调制。

    根据工艺语义生成每个视觉特征通道的：

        gamma
        beta

    调制形式：

        F_c = F * (1 + gamma) + beta

    输入：

        feature:
            [B, C, H, W]

        process_embedding:
            [B, D]

    输出：

        [B, C, H, W]
    """

    def __init__(
        self,
        feature_channels: int = 256,
        process_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.feature_channels = (
            feature_channels
        )

        # ----------------------------------------------------
        # Gamma
        # ----------------------------------------------------

        self.gamma_generator = nn.Sequential(
            nn.Linear(
                process_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                feature_channels,
            ),
        )

        # ----------------------------------------------------
        # Beta
        # ----------------------------------------------------

        self.beta_generator = nn.Sequential(
            nn.Linear(
                process_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                feature_channels,
            ),
        )

        # ----------------------------------------------------
        # 初始化
        #
        # 让训练初始阶段尽可能接近原始 FPN。
        # ----------------------------------------------------

        nn.init.zeros_(
            self.gamma_generator[-1].weight
        )

        nn.init.zeros_(
            self.gamma_generator[-1].bias
        )

        nn.init.zeros_(
            self.beta_generator[-1].weight
        )

        nn.init.zeros_(
            self.beta_generator[-1].bias
        )

    def forward(
        self,
        feature: torch.Tensor,
        process_embedding: torch.Tensor,
    ) -> torch.Tensor:

        self._check_input(
            feature,
            process_embedding,
        )

        gamma = self.gamma_generator(
            process_embedding
        )

        beta = self.beta_generator(
            process_embedding
        )

        # ----------------------------------------------------
        # [B,C]
        #
        # →
        #
        # [B,C,1,1]
        # ----------------------------------------------------

        gamma = gamma.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        beta = beta.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        output = (
            feature * (1.0 + gamma)
            + beta
        )

        return output

    def _check_input(
        self,
        feature: torch.Tensor,
        process_embedding: torch.Tensor,
    ):

        if feature.ndim != 4:
            raise ValueError(
                "feature 必须为 [B,C,H,W]。"
            )

        if process_embedding.ndim != 2:
            raise ValueError(
                "process_embedding 必须为 [B,D]。"
            )

        if feature.shape[0] != (
            process_embedding.shape[0]
        ):
            raise ValueError(
                "feature 和 process_embedding "
                "batch size 不一致。"
            )

        if feature.shape[1] != (
            self.feature_channels
        ):
            raise ValueError(
                "feature 通道数错误："
                f"expected={self.feature_channels}, "
                f"actual={feature.shape[1]}"
            )


# ============================================================
# 工艺条件空间调制
# ============================================================

class ProcessSpatialModulation(nn.Module):
    """
    工艺条件驱动的空间调制。

    将工艺语义与视觉特征结合，
    生成空间调制图。

    输入：

        feature:
            [B,C,H,W]

        process_embedding:
            [B,D]

    输出：

        [B,C,H,W]
    """

    def __init__(
        self,
        feature_channels: int = 256,
        process_dim: int = 256,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.feature_channels = (
            feature_channels
        )

        self.hidden_channels = (
            hidden_channels
        )

        # ----------------------------------------------------
        # 工艺语义投影
        # ----------------------------------------------------

        self.process_projection = nn.Sequential(
            nn.Linear(
                process_dim,
                hidden_channels,
            ),

            nn.GELU(),
        )

        # ----------------------------------------------------
        # 视觉特征投影
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.GELU(),
        )

        # ----------------------------------------------------
        # 空间调制
        # ----------------------------------------------------

        self.spatial_generator = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(
        self,
        feature: torch.Tensor,
        process_embedding: torch.Tensor,
    ) -> torch.Tensor:

        if feature.ndim != 4:
            raise ValueError(
                "feature 必须为 [B,C,H,W]。"
            )

        if process_embedding.ndim != 2:
            raise ValueError(
                "process_embedding 必须为 [B,D]。"
            )

        batch_size = feature.shape[0]
        channels = feature.shape[1]
        height = feature.shape[2]
        width = feature.shape[3]

        if channels != self.feature_channels:
            raise ValueError(
                "feature 通道数错误："
                f"expected={self.feature_channels}, "
                f"actual={channels}"
            )

        if process_embedding.shape[0] != (
            batch_size
        ):
            raise ValueError(
                "feature 和 process_embedding "
                "batch size 不一致。"
            )

        # ----------------------------------------------------
        # 视觉特征
        # ----------------------------------------------------

        visual_feature = (
            self.feature_projection(
                feature
            )
        )

        # ----------------------------------------------------
        # 工艺语义
        #
        # [B,D]
        #
        # →
        #
        # [B,C]
        #
        # →
        #
        # [B,C,H,W]
        # ----------------------------------------------------

        process_feature = (
            self.process_projection(
                process_embedding
            )
        )

        process_feature = (
            process_feature
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        process_feature = (
            process_feature.expand(
                -1,
                -1,
                height,
                width,
            )
        )

        # ----------------------------------------------------
        # 工艺语义与视觉特征融合
        # ----------------------------------------------------

        conditioned_feature = (
            visual_feature
            + process_feature
        )

        # ----------------------------------------------------
        # 生成空间调制图
        # ----------------------------------------------------

        spatial_map = (
            self.spatial_generator(
                conditioned_feature
            )
        )

        spatial_attention = torch.sigmoid(
            spatial_map
        )

        # ----------------------------------------------------
        # 空间调制
        #
        # 使用：
        #
        #     F' = F * (1 + M)
        #
        # 而不是：
        #
        #     F' = F * M
        #
        # 以避免完全抑制原始视觉特征。
        # ----------------------------------------------------

        output = (
            feature
            * (1.0 + spatial_attention)
        )

        return output


# ============================================================
# 单尺度 PFNM
# ============================================================

class PFNMLevel(nn.Module):
    """
    单个尺度的 PFNM。

    结构：

        Input Feature
              │
              ▼
        Channel Modulation
              │
              ▼
        Spatial Modulation
              │
              ▼
        3×3 Feature Fusion
              │
              ▼
          Residual
              │
              ▼
        Output Feature
    """

    def __init__(
        self,
        feature_channels: int = 256,
        process_dim: int = 256,
        spatial_hidden_dim: int = 64,
        use_channel_modulation: bool = True,
        use_spatial_modulation: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.feature_channels = (
            feature_channels
        )

        self.use_channel_modulation = (
            use_channel_modulation
        )

        self.use_spatial_modulation = (
            use_spatial_modulation
        )

        self.use_residual = (
            use_residual
        )

        # ====================================================
        # Channel modulation
        # ====================================================

        if use_channel_modulation:

            self.channel_modulation = (
                ProcessChannelModulation(
                    feature_channels=
                        feature_channels,

                    process_dim=
                        process_dim,

                    hidden_dim=
                        feature_channels,
                )
            )

        else:

            self.channel_modulation = (
                nn.Identity()
            )

        # ====================================================
        # Spatial modulation
        # ====================================================

        if use_spatial_modulation:

            self.spatial_modulation = (
                ProcessSpatialModulation(
                    feature_channels=
                        feature_channels,

                    process_dim=
                        process_dim,

                    hidden_channels=
                        spatial_hidden_dim,
                )
            )

        else:

            self.spatial_modulation = (
                nn.Identity()
            )

        # ====================================================
        # Feature fusion
        # ====================================================

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                feature_channels
            ),

            nn.GELU(),
        )

        # ====================================================
        # Dropout
        # ====================================================

        if dropout > 0.0:

            self.dropout = nn.Dropout2d(
                dropout
            )

        else:

            self.dropout = nn.Identity()

        # ====================================================
        # Residual scaling
        # ====================================================

        if use_residual:

            self.residual_scale = nn.Parameter(
                torch.tensor(0.1)
            )

        else:

            self.register_parameter(
                "residual_scale",
                None,
            )

    def forward(
        self,
        feature: torch.Tensor,
        process_embedding: torch.Tensor,
    ) -> torch.Tensor:

        identity = feature

        # ====================================================
        # Channel modulation
        # ====================================================

        if self.use_channel_modulation:

            x = self.channel_modulation(
                feature,
                process_embedding,
            )

        else:

            x = feature

        # ====================================================
        # Spatial modulation
        # ====================================================

        if self.use_spatial_modulation:

            x = self.spatial_modulation(
                x,
                process_embedding,
            )

        # ====================================================
        # Feature fusion
        # ====================================================

        x = self.feature_fusion(x)

        x = self.dropout(x)

        # ====================================================
        # Residual fusion
        # ====================================================

        if self.use_residual:

            x = (
                identity
                + self.residual_scale * x
            )

        return x


# ============================================================
# 多尺度 PFNM
# ============================================================

class PFNM(nn.Module):
    """
    多尺度 Process Feature-aware Network Modulation。

    对 P3-P6 分别进行工艺语义驱动的特征调制。

    输入：

        features:
        {
            "P3": [B,256,80,80],
            "P4": [B,256,40,40],
            "P5": [B,256,20,20],
            "P6": [B,256,10,10]
        }

        process_info:
            [B,16]

    输出：

        {
            "P3": [B,256,80,80],
            "P4": [B,256,40,40],
            "P5": [B,256,20,20],
            "P6": [B,256,10,10]
        }
    """

    def __init__(
        self,
        in_channels: int = 256,
        process_dim: int = 16,
        hidden_dim: int = 256,
        modulation_dim: int = 256,
        num_levels: int = 4,
        level_names: Optional[
            Sequence[str]
        ] = None,
        use_channel_modulation: bool = True,
        use_spatial_modulation: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.process_dim = process_dim

        self.hidden_dim = hidden_dim

        self.modulation_dim = modulation_dim

        self.num_levels = num_levels

        # ====================================================
        # Level names
        # ====================================================

        if level_names is None:

            level_names = [
                "P3",
                "P4",
                "P5",
                "P6",
            ]

        self.level_names = list(
            level_names
        )

        if len(self.level_names) != (
            self.num_levels
        ):
            raise ValueError(
                "level_names 数量必须与 "
                "num_levels 一致。"
            )

        # ====================================================
        # Process semantic encoder
        # ====================================================

        self.process_encoder = (
            ProcessSemanticEncoder(
                process_dim=process_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        )

        # ====================================================
        # Modulation embedding
        # ====================================================

        if hidden_dim != modulation_dim:

            self.modulation_projection = (
                nn.Sequential(
                    nn.Linear(
                        hidden_dim,
                        modulation_dim,
                    ),

                    nn.LayerNorm(
                        modulation_dim
                    ),

                    nn.GELU(),
                )
            )

        else:

            self.modulation_projection = (
                nn.Identity()
            )

        # ====================================================
        # Multi-scale PFNM levels
        # ====================================================

        self.level_modules = nn.ModuleDict()

        for level in self.level_names:

            self.level_modules[level] = (
                PFNMLevel(
                    feature_channels=
                        in_channels,

                    process_dim=
                        modulation_dim,

                    spatial_hidden_dim=
                        max(
                            32,
                            in_channels // 4,
                        ),

                    use_channel_modulation=
                        use_channel_modulation,

                    use_spatial_modulation=
                        use_spatial_modulation,

                    use_residual=
                        use_residual,

                    dropout=
                        dropout,
                )
            )

    # ========================================================
    # 输入检查
    # ========================================================

    def _validate_features(
        self,
        features: Dict[str, torch.Tensor],
    ):
        """
        检查 P3-P6。
        """

        if not isinstance(
            features,
            dict,
        ):
            raise TypeError(
                "features 必须为 dict。"
            )

        missing_levels = [
            level
            for level in self.level_names
            if level not in features
        ]

        if missing_levels:

            raise KeyError(
                "缺少 FPN 特征层："
                + ", ".join(
                    missing_levels
                )
            )

        batch_size = None

        for level in self.level_names:

            feature = features[level]

            if not isinstance(
                feature,
                torch.Tensor,
            ):
                raise TypeError(
                    f"{level} 必须为 torch.Tensor。"
                )

            if feature.ndim != 4:

                raise ValueError(
                    f"{level} 必须为 "
                    "[B,C,H,W]。"
                )

            if feature.shape[1] != (
                self.in_channels
            ):

                raise ValueError(
                    f"{level} 通道数错误："
                    f"expected={self.in_channels}, "
                    f"actual={feature.shape[1]}"
                )

            if batch_size is None:

                batch_size = (
                    feature.shape[0]
                )

            elif batch_size != (
                feature.shape[0]
            ):

                raise ValueError(
                    "P3-P6 的 batch size 不一致。"
                )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        process_info: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        参数：

            features:
                FPN 输出的 P3-P6。

            process_info:
                工艺信息 [B,16]。

        返回：

            PFNM 调制后的 P3-P6。
        """

        # ====================================================
        # 检查视觉特征
        # ====================================================

        self._validate_features(
            features
        )

        # ====================================================
        # 检查工艺信息
        # ====================================================

        if not isinstance(
            process_info,
            torch.Tensor,
        ):
            raise TypeError(
                "process_info 必须为 torch.Tensor。"
            )

        if process_info.ndim == 1:

            process_info = (
                process_info.unsqueeze(0)
            )

        if process_info.ndim != 2:

            raise ValueError(
                "process_info 必须为 [B,D]。"
            )

        if process_info.shape[-1] != (
            self.process_dim
        ):

            raise ValueError(
                "process_info 维度错误："
                f"expected={self.process_dim}, "
                f"actual={process_info.shape[-1]}"
            )

        # ====================================================
        # 获取设备
        # ====================================================

        reference_feature = features[
            self.level_names[0]
        ]

        if process_info.device != (
            reference_feature.device
        ):

            process_info = process_info.to(
                reference_feature.device
            )

        # ====================================================
        # Batch size
        # ====================================================

        if process_info.shape[0] != (
            reference_feature.shape[0]
        ):

            raise ValueError(
                "process_info 与 FPN 特征 "
                "batch size 不一致。"
            )

        # ====================================================
        # Process semantic encoding
        # ====================================================

        process_embedding = (
            self.process_encoder(
                process_info
            )
        )

        # ====================================================
        # Modulation embedding
        # ====================================================

        process_embedding = (
            self.modulation_projection(
                process_embedding
            )
        )

        # ====================================================
        # Multi-scale modulation
        # ====================================================

        outputs = {}

        for level in self.level_names:

            outputs[level] = (
                self.level_modules[level](
                    features[level],
                    process_embedding,
                )
            )

        return outputs


# ============================================================
# PFNM 消融版本
# ============================================================

class IdentityPFNM(nn.Module):
    """
    无 PFNM 基线。

    直接输出原始 FPN 特征。

    用于：

        w/o PFNM

    消融实验。
    """

    def __init__(
        self,
        level_names: Optional[
            Sequence[str]
        ] = None,
    ):
        super().__init__()

        if level_names is None:

            level_names = [
                "P3",
                "P4",
                "P5",
                "P6",
            ]

        self.level_names = list(
            level_names
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        process_info: Optional[
            torch.Tensor
        ] = None,
    ) -> Dict[str, torch.Tensor]:

        return {
            level: features[level]
            for level in self.level_names
        }


# ============================================================
# PFNM 工厂函数
# ============================================================

def build_pfnm(
    in_channels: int = 256,
    process_dim: int = 16,
    hidden_dim: int = 256,
    modulation_dim: int = 256,
    num_levels: int = 4,
    level_names: Optional[
        Sequence[str]
    ] = None,
    use_channel_modulation: bool = True,
    use_spatial_modulation: bool = True,
    use_residual: bool = True,
    dropout: float = 0.0,
) -> PFNM:
    """
    构建 PFNM。
    """

    return PFNM(
        in_channels=in_channels,
        process_dim=process_dim,
        hidden_dim=hidden_dim,
        modulation_dim=modulation_dim,
        num_levels=num_levels,
        level_names=level_names,
        use_channel_modulation=
            use_channel_modulation,
        use_spatial_modulation=
            use_spatial_modulation,
        use_residual=
            use_residual,
        dropout=dropout,
    )


# ============================================================
# 参数统计
# ============================================================

def count_parameters(
    model: nn.Module,
) -> Tuple[int, int]:
    """
    统计模型参数量。

    返回：

        total_params
        trainable_params
    """

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_params,
        trainable_params,
    )


# ============================================================
# 特征尺寸打印
# ============================================================

def print_feature_shapes(
    features: Dict[str, torch.Tensor],
):
    """
    打印 P3-P6 特征尺寸。
    """

    print("\n" + "=" * 70)
    print("PFNM Feature Shapes")
    print("=" * 70)

    for level in [
        "P3",
        "P4",
        "P5",
        "P6",
    ]:

        if level not in features:
            continue

        feature = features[level]

        print(
            f"{level}: "
            f"shape={tuple(feature.shape)}, "
            f"dtype={feature.dtype}, "
            f"device={feature.device}"
        )

    print("=" * 70)


# ============================================================
# 梯度检查
# ============================================================

def check_gradients(
    model: nn.Module,
):
    """
    检查 PFNM 是否存在可训练参数。
    """

    has_gradient_parameter = False

    for parameter in model.parameters():

        if parameter.requires_grad:

            has_gradient_parameter = True

            break

    if not has_gradient_parameter:

        raise RuntimeError(
            "PFNM 中没有可训练参数。"
        )

    return True


# ============================================================
# 单元测试
# ============================================================

def test_pfnm():
    """
    PFNM 完整测试。
    """

    print("\n" + "=" * 70)
    print("Testing pfnm.py")
    print("=" * 70)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    batch_size = 2

    # ========================================================
    # 构造 FPN 输出
    # ========================================================

    features = {

        "P3": torch.randn(
            batch_size,
            256,
            80,
            80,
            device=device,
        ),

        "P4": torch.randn(
            batch_size,
            256,
            40,
            40,
            device=device,
        ),

        "P5": torch.randn(
            batch_size,
            256,
            20,
            20,
            device=device,
        ),

        "P6": torch.randn(
            batch_size,
            256,
            10,
            10,
            device=device,
        ),
    }

    print("\nFPN input:")
    print_feature_shapes(
        features
    )

    # ========================================================
    # 构造工艺信息
    # ========================================================

    process_info = torch.tensor(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,

                10.0,
                0.05,
                10.0,
                5.0,

                15.0,
                10.0,
                20.0,
                5.0,

                0.5,
                0.5,
                1.0,
                0.5,
            ],

            [
                0.0,
                1.0,
                0.0,
                0.0,

                10.0,
                0.05,
                10.0,
                5.0,

                15.0,
                10.0,
                20.0,
                5.0,

                0.5,
                0.5,
                1.0,
                0.5,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "\nProcess information:"
    )

    print(
        f"shape = "
        f"{tuple(process_info.shape)}"
    )

    assert process_info.shape == (
        batch_size,
        16,
    )

    # ========================================================
    # 构建 PFNM
    # ========================================================

    model = build_pfnm(
        in_channels=256,
        process_dim=16,
        hidden_dim=256,
        modulation_dim=256,
        num_levels=4,
        level_names=[
            "P3",
            "P4",
            "P5",
            "P6",
        ],
        use_channel_modulation=True,
        use_spatial_modulation=True,
        use_residual=True,
        dropout=0.0,
    ).to(device)

    # ========================================================
    # 参数统计
    # ========================================================

    total_params, trainable_params = (
        count_parameters(model)
    )

    print("\nPFNM parameters:")
    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    assert (
        total_params > 0
    )

    assert (
        trainable_params > 0
    )

    check_gradients(
        model
    )

    # ========================================================
    # Evaluation forward
    # ========================================================

    model.eval()

    with torch.no_grad():

        outputs = model(
            features,
            process_info,
        )

    # ========================================================
    # 输出尺寸
    # ========================================================

    print(
        "\nPFNM output:"
    )

    print_feature_shapes(
        outputs
    )

    expected_shapes = {

        "P3": (
            batch_size,
            256,
            80,
            80,
        ),

        "P4": (
            batch_size,
            256,
            40,
            40,
        ),

        "P5": (
            batch_size,
            256,
            20,
            20,
        ),

        "P6": (
            batch_size,
            256,
            10,
            10,
        ),
    }

    for level, expected_shape in (
        expected_shapes.items()
    ):

        actual_shape = tuple(
            outputs[level].shape
        )

        assert (
            actual_shape
            == expected_shape
        ), (
            f"{level} shape 错误："
            f"expected={expected_shape}, "
            f"actual={actual_shape}"
        )

    print(
        "\n输出尺寸检查通过。"
    )

    # ========================================================
    # 数值检查
    # ========================================================

    for level, feature in (
        outputs.items()
    ):

        assert torch.isfinite(
            feature
        ).all(), (
            f"{level} 中存在 NaN 或 Inf。"
        )

    print(
        "输出数值检查通过。"
    )

    # ========================================================
    # 梯度测试
    # ========================================================

    model.train()

    train_features = {

        "P3": torch.randn(
            1,
            256,
            80,
            80,
            device=device,
            requires_grad=True,
        ),

        "P4": torch.randn(
            1,
            256,
            40,
            40,
            device=device,
            requires_grad=True,
        ),

        "P5": torch.randn(
            1,
            256,
            20,
            20,
            device=device,
            requires_grad=True,
        ),

        "P6": torch.randn(
            1,
            256,
            10,
            10,
            device=device,
            requires_grad=True,
        ),
    }

    train_process_info = torch.randn(
        1,
        16,
        device=device,
        requires_grad=True,
    )

    train_outputs = model(
        train_features,
        train_process_info,
    )

    loss = sum(
        feature.mean()
        for feature
        in train_outputs.values()
    )

    loss.backward()

    # --------------------------------------------------------
    # 检查 FPN 特征梯度
    # --------------------------------------------------------

    for level, feature in (
        train_features.items()
    ):

        assert feature.grad is not None, (
            f"{level} 没有梯度。"
        )

    # --------------------------------------------------------
    # 检查工艺信息梯度
    # --------------------------------------------------------

    assert (
        train_process_info.grad
        is not None
    ), (
        "process_info 没有梯度。"
    )

    print(
        "反向传播检查通过。"
    )

    # ========================================================
    # 工艺条件变化测试
    # ========================================================

    model.eval()

    process_a = torch.zeros(
        batch_size,
        16,
        device=device,
    )

    process_b = torch.ones(
        batch_size,
        16,
        device=device,
    )

    with torch.no_grad():

        output_a = model(
            features,
            process_a,
        )

        output_b = model(
            features,
            process_b,
        )

    total_difference = 0.0

    for level in [
        "P3",
        "P4",
        "P5",
        "P6",
    ]:

        difference = torch.mean(
            torch.abs(
                output_a[level]
                - output_b[level]
            )
        ).item()

        total_difference += (
            difference
        )

        print(
            f"{level} mean difference: "
            f"{difference:.8f}"
        )

    print(
        "Total mean difference: "
        f"{total_difference:.8f}"
    )

    # ========================================================
    # Identity PFNM 测试
    # ========================================================

    identity_model = IdentityPFNM()

    identity_outputs = identity_model(
        features,
        process_info,
    )

    for level in [
        "P3",
        "P4",
        "P5",
        "P6",
    ]:

        assert (
            identity_outputs[level]
            is features[level]
        )

    print(
        "IdentityPFNM 消融接口检查通过。"
    )

    # ========================================================
    # 单独关闭通道调制测试
    # ========================================================

    model_without_channel = build_pfnm(
        in_channels=256,
        process_dim=16,
        hidden_dim=256,
        modulation_dim=256,
        num_levels=4,
        use_channel_modulation=False,
        use_spatial_modulation=True,
        use_residual=True,
    ).to(device)

    model_without_channel.eval()

    with torch.no_grad():

        output_without_channel = (
            model_without_channel(
                features,
                process_info,
            )
        )

    assert (
        output_without_channel["P3"].shape
        == features["P3"].shape
    )

    print(
        "关闭通道调制检查通过。"
    )

    # ========================================================
    # 单独关闭空间调制测试
    # ========================================================

    model_without_spatial = build_pfnm(
        in_channels=256,
        process_dim=16,
        hidden_dim=256,
        modulation_dim=256,
        num_levels=4,
        use_channel_modulation=True,
        use_spatial_modulation=False,
        use_residual=True,
    ).to(device)

    model_without_spatial.eval()

    with torch.no_grad():

        output_without_spatial = (
            model_without_spatial(
                features,
                process_info,
            )
        )

    assert (
        output_without_spatial["P3"].shape
        == features["P3"].shape
    )

    print(
        "关闭空间调制检查通过。"
    )

    # ========================================================
    # 完成
    # ========================================================

    print("\n" + "=" * 70)
    print("pfnm.py Test Passed")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_pfnm()