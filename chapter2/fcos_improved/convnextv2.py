# coding=utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional


# ============================================================
# LayerNorm
# ============================================================

class LayerNorm(nn.Module):
    """
    LayerNorm

    支持两种数据格式：

        channels_last:
            [B, H, W, C]

        channels_first:
            [B, C, H, W]
    """

    def __init__(
            self,
            normalized_shape,
            eps: float = 1e-6,
            data_format: str = "channels_last"
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(normalized_shape)
        )

        self.bias = nn.Parameter(
            torch.zeros(normalized_shape)
        )

        self.eps = eps

        if data_format not in [
            "channels_last",
            "channels_first"
        ]:
            raise ValueError(
                "data_format must be 'channels_last' "
                "or 'channels_first'"
            )

        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.data_format == "channels_last":

            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps
            )

        elif self.data_format == "channels_first":

            mean = x.mean(
                dim=1,
                keepdim=True
            )

            var = (
                x - mean
            ).pow(2).mean(
                dim=1,
                keepdim=True
            )

            x = (
                x - mean
            ) / torch.sqrt(
                var + self.eps
            )

            x = (
                self.weight[:, None, None] * x
                + self.bias[:, None, None]
            )

            return x


# ============================================================
# GRN
# ============================================================

class GRN(nn.Module):
    """
    Global Response Normalization

    ConvNeXt V2 的核心组件之一。

    输入：
        [B, H, W, C]

    输出：
        [B, H, W, C]
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6
    ):
        super().__init__()

        self.gamma = nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )

        self.eps = eps

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # 计算空间维度上的 L2 norm
        # ----------------------------------------------------

        gx = torch.norm(
            x,
            p=2,
            dim=(1, 2),
            keepdim=True
        )

        # ----------------------------------------------------
        # 通道间归一化
        # ----------------------------------------------------

        nx = gx / (
            gx.mean(
                dim=-1,
                keepdim=True
            ) + self.eps
        )

        # ----------------------------------------------------
        # GRN
        # ----------------------------------------------------

        return (
            self.gamma * (
                x * nx
            )
            + self.beta
            + x
        )


# ============================================================
# DropPath
# ============================================================

class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Path
    """

    def __init__(
            self,
            drop_prob: float = 0.0
    ):
        super().__init__()

        self.drop_prob = float(
            drop_prob
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        if self.drop_prob == 0.0:
            return x

        if not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (
            x.shape[0],
        ) + (
            1,
        ) * (
            x.ndim - 1
        )

        random_tensor = (
            keep_prob
            + torch.rand(
                shape,
                dtype=x.dtype,
                device=x.device
            )
        )

        random_tensor.floor_()

        return (
            x
            / keep_prob
            * random_tensor
        )


# ============================================================
# ConvNeXt V2 Block
# ============================================================

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 Block

    结构：

        Input
          │
          ├── Depthwise Conv 7×7
          │
          ├── LayerNorm
          │
          ├── Linear Expansion
          │
          ├── GELU
          │
          ├── GRN
          │
          ├── Linear Projection
          │
          ├── DropPath
          │
          └── Residual

    输入：
        [B, C, H, W]

    输出：
        [B, C, H, W]
    """

    def __init__(
            self,
            dim: int,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            kernel_size: int = 7,
            expansion_ratio: int = 4
    ):
        super().__init__()

        self.dim = dim

        # ----------------------------------------------------
        # Depthwise convolution
        # ----------------------------------------------------

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        self.norm = LayerNorm(
            dim,
            eps=1e-6,
            data_format="channels_last"
        )

        # ----------------------------------------------------
        # Pointwise expansion
        # ----------------------------------------------------

        hidden_dim = (
            dim * expansion_ratio
        )

        self.pwconv1 = nn.Linear(
            dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------

        self.act = nn.GELU()

        # ----------------------------------------------------
        # GRN
        # ----------------------------------------------------

        self.grn = GRN(
            hidden_dim
        )

        # ----------------------------------------------------
        # Pointwise projection
        # ----------------------------------------------------

        self.pwconv2 = nn.Linear(
            hidden_dim,
            dim
        )

        # ----------------------------------------------------
        # Layer Scale
        # ----------------------------------------------------

        if layer_scale_init_value > 0:

            self.gamma = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim)
            )

        else:

            self.gamma = None

        # ----------------------------------------------------
        # DropPath
        # ----------------------------------------------------

        self.drop_path = DropPath(
            drop_path
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        shortcut = x

        # ----------------------------------------------------
        # Depthwise convolution
        # ----------------------------------------------------

        x = self.dwconv(x)

        # ----------------------------------------------------
        # [B, C, H, W]
        # →
        # [B, H, W, C]
        # ----------------------------------------------------

        x = x.permute(
            0,
            2,
            3,
            1
        )

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        x = self.norm(x)

        # ----------------------------------------------------
        # Pointwise MLP
        # ----------------------------------------------------

        x = self.pwconv1(x)

        x = self.act(x)

        # ----------------------------------------------------
        # GRN
        # ----------------------------------------------------

        x = self.grn(x)

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        x = self.pwconv2(x)

        # ----------------------------------------------------
        # Layer Scale
        # ----------------------------------------------------

        if self.gamma is not None:

            x = (
                self.gamma * x
            )

        # ----------------------------------------------------
        # [B, H, W, C]
        # →
        # [B, C, H, W]
        # ----------------------------------------------------

        x = x.permute(
            0,
            3,
            1,
            2
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = shortcut + self.drop_path(x)

        return x


# ============================================================
# Downsampling Layer
# ============================================================

class DownsampleLayer(nn.Module):
    """
    ConvNeXt V2 Downsampling Layer

    使用：

        LayerNorm
        +
        2×2 Conv, stride=2

    用于相邻 Stage 之间进行空间下采样。
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int
    ):
        super().__init__()

        self.norm = LayerNorm(
            in_dim,
            eps=1e-6,
            data_format="channels_first"
        )

        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=2,
            stride=2
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.norm(x)

        x = self.conv(x)

        return x


# ============================================================
# ConvNeXt V2 Stage
# ============================================================

class ConvNeXtV2Stage(nn.Module):
    """
    ConvNeXt V2 Stage

    一个 Stage 由多个 ConvNeXt V2 Block 构成。
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            drop_path_rates: List[float],
            layer_scale_init_value: float = 1e-6,
            expansion_ratio: int = 4
    ):
        super().__init__()

        blocks = []

        for i in range(depth):

            blocks.append(
                ConvNeXtV2Block(
                    dim=dim,
                    drop_path=drop_path_rates[i],
                    layer_scale_init_value=
                    layer_scale_init_value,
                    expansion_ratio=
                    expansion_ratio
                )
            )

        self.blocks = nn.Sequential(
            *blocks
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.blocks(x)


# ============================================================
# ConvNeXt V2 Backbone
# ============================================================

class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2 Backbone

    默认配置：

        depths:
            [3, 3, 9, 3]

        dims:
            [96, 192, 384, 768]

    对应：

        C2:
            96 channels

        C3:
            192 channels

        C4:
            384 channels

        C5:
            768 channels

    参数：

        in_chans:
            输入图像通道数

        depths:
            四个 Stage 的 Block 数量

        dims:
            四个 Stage 的通道数

        drop_path_rate:
            stochastic depth 最大概率

        layer_scale_init_value:
            Layer Scale 初始值

        expansion_ratio:
            MLP expansion ratio

        out_indices:
            返回哪些 Stage

    输出：

        List[Tensor]

        默认：
            [C2, C3, C4, C5]
    """

    def __init__(
            self,
            in_chans: int = 3,
            depths: Tuple[int, int, int, int] = (
                3,
                3,
                9,
                3
            ),
            dims: Tuple[int, int, int, int] = (
                96,
                192,
                384,
                768
            ),
            drop_path_rate: float = 0.1,
            layer_scale_init_value: float = 1e-6,
            expansion_ratio: int = 4,
            out_indices: Tuple[int, ...] = (
                0,
                1,
                2,
                3
            )
    ):
        super().__init__()

        # ----------------------------------------------------
        # 参数检查
        # ----------------------------------------------------

        if len(depths) != 4:
            raise ValueError(
                "depths must contain 4 values."
            )

        if len(dims) != 4:
            raise ValueError(
                "dims must contain 4 values."
            )

        self.depths = depths

        self.dims = dims

        self.out_indices = out_indices

        # ----------------------------------------------------
        # Stem
        #
        # 3×3 input
        # →
        # 4×4 / stride 4
        # ----------------------------------------------------

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=dims[0],
                kernel_size=4,
                stride=4
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first"
            )
        )

        # ----------------------------------------------------
        # Stochastic Depth rates
        # ----------------------------------------------------

        total_blocks = sum(
            depths
        )

        drop_path_rates = torch.linspace(
            0,
            drop_path_rate,
            total_blocks
        ).tolist()

        # ----------------------------------------------------
        # Stages
        # ----------------------------------------------------

        self.stages = nn.ModuleList()

        current_block = 0

        for stage_idx in range(4):

            stage_depth = depths[
                stage_idx
            ]

            stage_drop_rates = (
                drop_path_rates[
                    current_block:
                    current_block + stage_depth
                ]
            )

            stage = ConvNeXtV2Stage(
                dim=dims[stage_idx],
                depth=stage_depth,
                drop_path_rates=stage_drop_rates,
                layer_scale_init_value=
                layer_scale_init_value,
                expansion_ratio=
                expansion_ratio
            )

            self.stages.append(
                stage
            )

            current_block += stage_depth

        # ----------------------------------------------------
        # Downsampling
        #
        # Stage 1 → Stage 2
        # Stage 2 → Stage 3
        # Stage 3 → Stage 4
        # ----------------------------------------------------

        self.downsample_layers = nn.ModuleList()

        for i in range(3):

            self.downsample_layers.append(
                DownsampleLayer(
                    in_dim=dims[i],
                    out_dim=dims[i + 1]
                )
            )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )

    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(
            self,
            module: nn.Module
    ):

        if isinstance(
                module,
                nn.Conv2d
        ):

            nn.init.trunc_normal_(
                module.weight,
                std=0.02
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )

        elif isinstance(
                module,
                nn.Linear
        ):

            nn.init.trunc_normal_(
                module.weight,
                std=0.02
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
            self,
            x: torch.Tensor
    ) -> List[torch.Tensor]:

        outputs = []

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Four stages
        # ----------------------------------------------------

        for stage_idx in range(4):

            # Stage
            x = self.stages[
                stage_idx
            ](x)

            # 保存当前尺度
            if stage_idx in self.out_indices:

                outputs.append(x)

            # Downsample
            if stage_idx < 3:

                x = self.downsample_layers[
                    stage_idx
                ](x)

        return outputs


# ============================================================
# ConvNeXt V2 Tiny
# ============================================================

class ConvNeXtV2Tiny(ConvNeXtV2):
    """
    ConvNeXt V2 Tiny

    对应配置：

        depths = [3, 3, 9, 3]
        dims   = [96, 192, 384, 768]
    """

    def __init__(
            self,
            in_chans: int = 3,
            drop_path_rate: float = 0.1,
            out_indices: Tuple[int, ...] = (
                0,
                1,
                2,
                3
            )
    ):

        super().__init__(
            in_chans=in_chans,
            depths=(
                3,
                3,
                9,
                3
            ),
            dims=(
                96,
                192,
                384,
                768
            ),
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )


# ============================================================
# ConvNeXt V2 Small
# ============================================================

class ConvNeXtV2Small(ConvNeXtV2):
    """
    ConvNeXt V2 Small

    配置：

        depths = [3, 3, 27, 3]
        dims   = [96, 192, 384, 768]
    """

    def __init__(
            self,
            in_chans: int = 3,
            drop_path_rate: float = 0.1,
            out_indices: Tuple[int, ...] = (
                0,
                1,
                2,
                3
            )
    ):

        super().__init__(
            in_chans=in_chans,
            depths=(
                3,
                3,
                27,
                3
            ),
            dims=(
                96,
                192,
                384,
                768
            ),
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )


# ============================================================
# ConvNeXt V2 Base
# ============================================================

class ConvNeXtV2Base(ConvNeXtV2):
    """
    ConvNeXt V2 Base

    配置：

        depths = [3, 3, 27, 3]
        dims   = [128, 256, 512, 1024]
    """

    def __init__(
            self,
            in_chans: int = 3,
            drop_path_rate: float = 0.1,
            out_indices: Tuple[int, ...] = (
                0,
                1,
                2,
                3
            )
    ):

        super().__init__(
            in_chans=in_chans,
            depths=(
                3,
                3,
                27,
                3
            ),
            dims=(
                128,
                256,
                512,
                1024
            ),
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )


# ============================================================
# Factory Function
# ============================================================

def build_convnextv2(
        variant: str = "tiny",
        in_chans: int = 3,
        drop_path_rate: float = 0.1,
        out_indices: Tuple[int, ...] = (
            0,
            1,
            2,
            3
        )
) -> ConvNeXtV2:

    """
    构建 ConvNeXt V2 Backbone。

    参数：
        variant:
            tiny / small / base

        in_chans:
            输入通道

        drop_path_rate:
            DropPath 最大概率

        out_indices:
            输出 Stage

    返回：
        ConvNeXtV2
    """

    variant = variant.lower()

    if variant == "tiny":

        return ConvNeXtV2Tiny(
            in_chans=in_chans,
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )

    elif variant == "small":

        return ConvNeXtV2Small(
            in_chans=in_chans,
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )

    elif variant == "base":

        return ConvNeXtV2Base(
            in_chans=in_chans,
            drop_path_rate=
            drop_path_rate,
            out_indices=
            out_indices
        )

    else:

        raise ValueError(
            f"Unsupported ConvNeXt V2 variant: {variant}. "
            f"Available variants: tiny, small, base."
        )


# ============================================================
# Parameter Information
# ============================================================

def count_parameters(
        model: nn.Module
) -> int:

    """
    计算模型可训练参数量。
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Test
# ============================================================

def _test():

    print("=" * 70)
    print("ConvNeXt V2 Backbone Test")
    print("=" * 70)

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = build_convnextv2(
        variant="tiny",
        in_chans=3,
        drop_path_rate=0.1
    )

    model.eval()

    # --------------------------------------------------------
    # 输入
    # --------------------------------------------------------

    x = torch.randn(
        2,
        3,
        640,
        640
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        features = model(x)

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print(
        f"Input shape: {tuple(x.shape)}"
    )

    print()

    for i, feature in enumerate(
            features
    ):

        print(
            f"C{i + 2}: "
            f"{tuple(feature.shape)}"
        )

    # --------------------------------------------------------
    # 参数量
    # --------------------------------------------------------

    params = count_parameters(
        model
    )

    print()

    print(
        f"Trainable parameters: "
        f"{params / 1e6:.2f} M"
    )

    # --------------------------------------------------------
    # 检查输出
    # --------------------------------------------------------

    expected_shapes = [
        (2, 96, 160, 160),
        (2, 192, 80, 80),
        (2, 384, 40, 40),
        (2, 768, 20, 20)
    ]

    for feature, expected in zip(
            features,
            expected_shapes
    ):

        assert tuple(
            feature.shape
        ) == expected, (
            f"Shape mismatch: "
            f"{tuple(feature.shape)} "
            f"!= {expected}"
        )

    print()

    print(
        "All shape tests passed."
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    _test()