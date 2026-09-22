# coding=utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional, Union


# ============================================================
# Basic Conv Block
# ============================================================

class ConvBNAct(nn.Module):
    """
    基础卷积模块：

        Conv2d
          ↓
        BatchNorm2d
          ↓
        Activation

    用于 EMA 内部的局部特征提取。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            activation: bool = True
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.activation = (
            nn.SiLU(inplace=True)
            if activation
            else nn.Identity()
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.bn(x)

        x = self.activation(x)

        return x


# ============================================================
# Efficient Multi-Scale Attention
# ============================================================

class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention

    输入：
        x: [B, C, H, W]

    输出：
        y: [B, C, H, W]

    核心思想：

        1. 将通道划分为多个 group。
        2. 在每个 group 内进行空间方向信息建模。
        3. 使用横向和纵向全局池化提取空间上下文。
        4. 使用 1×1 卷积完成跨空间特征交互。
        5. 使用 3×3 卷积提取局部多尺度信息。
        6. 通过两个注意力分支计算空间权重。
        7. 使用 Softmax 对响应进行归一化。
        8. 通过残差连接增强稳定性。

    适用于：
        ConvNeXt V2 C4
        ConvNeXt V2 C5
    """

    def __init__(
            self,
            channels: int,
            groups: int = 32,
            kernel_size: int = 3,
            use_residual: bool = True,
            reduction: int = 1
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be greater than 0."
            )

        if groups <= 0:
            raise ValueError(
                "groups must be greater than 0."
            )

        # ----------------------------------------------------
        # 如果通道数不能被 groups 整除，
        # 自动寻找最大的合法 group 数。
        # ----------------------------------------------------

        if channels % groups != 0:

            valid_groups = []

            for g in range(
                    1,
                    groups + 1
            ):

                if channels % g == 0:
                    valid_groups.append(g)

            groups = max(
                valid_groups
            )

        self.channels = channels

        self.groups = groups

        self.group_channels = (
            channels // groups
        )

        self.kernel_size = kernel_size

        self.use_residual = use_residual

        # ----------------------------------------------------
        # Group Normalization
        #
        # EMA 内部按照 group 处理特征。
        # ----------------------------------------------------

        self.group_norm = nn.GroupNorm(
            num_groups=groups,
            num_channels=channels
        )

        # ----------------------------------------------------
        # 1×1 convolution
        #
        # 用于横向/纵向空间信息融合。
        # ----------------------------------------------------

        self.conv1x1 = nn.Conv2d(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # ----------------------------------------------------
        # 3×3 convolution
        #
        # 提取局部空间上下文。
        # ----------------------------------------------------

        self.conv3x3 = nn.Conv2d(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )

        # ----------------------------------------------------
        # Sigmoid
        # ----------------------------------------------------

        self.sigmoid = nn.Sigmoid()

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        self.softmax = nn.Softmax(
            dim=-1
        )

        # ----------------------------------------------------
        # 最终融合卷积
        # ----------------------------------------------------

        self.fusion = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.fusion_bn = nn.BatchNorm2d(
            channels
        )

        # ----------------------------------------------------
        # 可学习缩放系数
        #
        # 从较小值开始，使训练初期不会过度改变
        # ConvNeXt V2 原始特征。
        # ----------------------------------------------------

        self.gamma = nn.Parameter(
            torch.ones(1) * 0.1
        )

        # ----------------------------------------------------
        # 权重初始化
        # ----------------------------------------------------

        self._init_weights()

    # ========================================================
    # Weight Initialization
    # ========================================================

    def _init_weights(self):

        for module in self.modules():

            if isinstance(
                    module,
                    nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0
                    )

            elif isinstance(
                    module,
                    nn.BatchNorm2d
            ):

                nn.init.constant_(
                    module.weight,
                    1
                )

                nn.init.constant_(
                    module.bias,
                    0
                )

            elif isinstance(
                    module,
                    nn.GroupNorm
            ):

                nn.init.constant_(
                    module.weight,
                    1
                )

                nn.init.constant_(
                    module.bias,
                    0
                )

    # ========================================================
    # Group Features
    # ========================================================

    def _reshape_groups(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        将：

            [B, C, H, W]

        转换为：

            [B*G, C/G, H, W]
        """

        B, C, H, W = x.shape

        x = x.reshape(
            B * self.groups,
            self.group_channels,
            H,
            W
        )

        return x

    # ========================================================
    # Restore Groups
    # ========================================================

    def _restore_groups(
            self,
            x: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        将：

            [B*G, C/G, H, W]

        恢复为：

            [B, C, H, W]
        """

        _, _, H, W = x.shape

        x = x.reshape(
            batch_size,
            self.channels,
            H,
            W
        )

        return x

    # ========================================================
    # Forward
    # ========================================================

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 4:

            raise ValueError(
                "EMA expects a 4D tensor "
                "[B, C, H, W]. "
                f"Got shape: {tuple(x.shape)}"
            )

        B, C, H, W = x.shape

        if C != self.channels:

            raise ValueError(
                f"EMA expected {self.channels} "
                f"channels, but got {C}."
            )

        identity = x

        # ----------------------------------------------------
        # Group Normalization
        # ----------------------------------------------------

        x_norm = self.group_norm(x)

        # ----------------------------------------------------
        # Group reshape
        # ----------------------------------------------------

        group_x = self._reshape_groups(
            x_norm
        )

        # ====================================================
        # Branch 1:
        # Horizontal and vertical global pooling
        # ====================================================

        # ----------------------------------------------------
        # Height pooling
        #
        # [B*G, Cg, H, W]
        # →
        # [B*G, Cg, H, 1]
        # ----------------------------------------------------

        x_h = group_x.mean(
            dim=3,
            keepdim=True
        )

        # ----------------------------------------------------
        # Width pooling
        #
        # [B*G, Cg, H, W]
        # →
        # [B*G, Cg, 1, W]
        # ----------------------------------------------------

        x_w = group_x.mean(
            dim=2,
            keepdim=True
        )

        # ----------------------------------------------------
        # 转换 width branch
        # ----------------------------------------------------

        x_w = x_w.permute(
            0,
            1,
            3,
            2
        )

        # ----------------------------------------------------
        # 拼接
        #
        # [B*G, Cg, H+W, 1]
        # ----------------------------------------------------

        hw = torch.cat(
            [
                x_h,
                x_w
            ],
            dim=2
        )

        # ----------------------------------------------------
        # 1×1 convolution
        # ----------------------------------------------------

        hw = self.conv1x1(
            hw
        )

        # ----------------------------------------------------
        # 拆分
        # ----------------------------------------------------

        h_attn = hw[
            :,
            :,
            :H,
            :
        ]

        w_attn = hw[
            :,
            :,
            H:H + W,
            :
        ]

        # ----------------------------------------------------
        # 恢复 width 方向
        # ----------------------------------------------------

        w_attn = w_attn.permute(
            0,
            1,
            3,
            2
        )

        # ----------------------------------------------------
        # Sigmoid spatial attention
        # ----------------------------------------------------

        h_attn = self.sigmoid(
            h_attn
        )

        w_attn = self.sigmoid(
            w_attn
        )

        # ----------------------------------------------------
        # 空间方向注意力
        # ----------------------------------------------------

        branch1 = (
            group_x
            * h_attn
            * w_attn
        )

        # ====================================================
        # Branch 2:
        # Local multi-scale feature
        # ====================================================

        branch2 = self.conv3x3(
            group_x
        )

        # ====================================================
        # Feature interaction
        # ====================================================

        # ----------------------------------------------------
        # Global spatial pooling
        # ----------------------------------------------------

        b1_pool = branch1.mean(
            dim=(2, 3)
        )

        b2_pool = branch2.mean(
            dim=(2, 3)
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        b1_weight = self.softmax(
            b1_pool
        )

        b2_weight = self.softmax(
            b2_pool
        )

        # ----------------------------------------------------
        # 将两个分支响应融合
        # ----------------------------------------------------

        attention = (
            b1_weight
            + b2_weight
        )

        # ----------------------------------------------------
        # [B*G, Cg]
        # →
        # [B*G, Cg, 1, 1]
        # ----------------------------------------------------

        attention = attention[
            :,
            :,
            None,
            None
        ]

        # ----------------------------------------------------
        # Sigmoid
        # ----------------------------------------------------

        attention = self.sigmoid(
            attention
        )

        # ----------------------------------------------------
        # 对 group feature 进行调制
        # ----------------------------------------------------

        group_out = (
            group_x
            * attention
        )

        # ====================================================
        # Restore
        # ====================================================

        out = self._restore_groups(
            group_out,
            B
        )

        # ====================================================
        # Fusion
        # ====================================================

        out = self.fusion(
            out
        )

        out = self.fusion_bn(
            out
        )

        # ====================================================
        # Residual
        # ====================================================

        if self.use_residual:

            out = (
                identity
                + self.gamma * out
            )

        else:

            out = out * self.gamma

        return out


# ============================================================
# EMA for C4
# ============================================================

class EMA_C4(nn.Module):
    """
    C4 EMA

    ConvNeXt V2 Tiny：
        C4 = 384 channels
    """

    def __init__(
            self,
            groups: int = 32,
            kernel_size: int = 3,
            use_residual: bool = True
    ):
        super().__init__()

        self.ema = EMA(
            channels=384,
            groups=groups,
            kernel_size=kernel_size,
            use_residual=use_residual
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.ema(x)


# ============================================================
# EMA for C5
# ============================================================

class EMA_C5(nn.Module):
    """
    C5 EMA

    ConvNeXt V2 Tiny：
        C5 = 768 channels
    """

    def __init__(
            self,
            groups: int = 32,
            kernel_size: int = 3,
            use_residual: bool = True
    ):
        super().__init__()

        self.ema = EMA(
            channels=768,
            groups=groups,
            kernel_size=kernel_size,
            use_residual=use_residual
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.ema(x)


# ============================================================
# Multi-scale EMA
# ============================================================

class MultiScaleEMA(nn.Module):
    """
    多尺度 EMA 封装模块。

    输入：

        features = [
            C2,
            C3,
            C4,
            C5
        ]

    输出：

        outputs = [
            C2,
            C3,
            C4_EMA,
            C5_EMA
        ]

    当前 Chapter 2 中：

        C2 → 不处理
        C3 → 不处理
        C4 → EMA
        C5 → EMA
    """

    def __init__(
            self,
            channels: Tuple[int, int, int, int] = (
                96,
                192,
                384,
                768
            ),
            groups: int = 32,
            kernel_size: int = 3,
            apply_to: Tuple[str, ...] = (
                "C4",
                "C5"
            ),
            use_residual: bool = True
    ):
        super().__init__()

        self.channels = channels

        self.apply_to = apply_to

        self.groups = groups

        self.kernel_size = kernel_size

        self.use_residual = use_residual

        # ----------------------------------------------------
        # C2
        # ----------------------------------------------------

        if "C2" in apply_to:

            self.ema_c2 = EMA(
                channels=channels[0],
                groups=groups,
                kernel_size=kernel_size,
                use_residual=use_residual
            )

        else:

            self.ema_c2 = nn.Identity()

        # ----------------------------------------------------
        # C3
        # ----------------------------------------------------

        if "C3" in apply_to:

            self.ema_c3 = EMA(
                channels=channels[1],
                groups=groups,
                kernel_size=kernel_size,
                use_residual=use_residual
            )

        else:

            self.ema_c3 = nn.Identity()

        # ----------------------------------------------------
        # C4
        # ----------------------------------------------------

        if "C4" in apply_to:

            self.ema_c4 = EMA(
                channels=channels[2],
                groups=groups,
                kernel_size=kernel_size,
                use_residual=use_residual
            )

        else:

            self.ema_c4 = nn.Identity()

        # ----------------------------------------------------
        # C5
        # ----------------------------------------------------

        if "C5" in apply_to:

            self.ema_c5 = EMA(
                channels=channels[3],
                groups=groups,
                kernel_size=kernel_size,
                use_residual=use_residual
            )

        else:

            self.ema_c5 = nn.Identity()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
            self,
            features: List[torch.Tensor]
    ) -> List[torch.Tensor]:

        if not isinstance(
                features,
                (list, tuple)
        ):

            raise TypeError(
                "MultiScaleEMA expects "
                "a list or tuple of feature tensors."
            )

        if len(features) != 4:

            raise ValueError(
                "MultiScaleEMA expects four "
                "feature levels: C2, C3, C4, C5."
            )

        c2, c3, c4, c5 = features

        # ----------------------------------------------------
        # C2
        # ----------------------------------------------------

        c2 = self.ema_c2(
            c2
        )

        # ----------------------------------------------------
        # C3
        # ----------------------------------------------------

        c3 = self.ema_c3(
            c3
        )

        # ----------------------------------------------------
        # C4
        # ----------------------------------------------------

        c4 = self.ema_c4(
            c4
        )

        # ----------------------------------------------------
        # C5
        # ----------------------------------------------------

        c5 = self.ema_c5(
            c5
        )

        return [
            c2,
            c3,
            c4,
            c5
        ]


# ============================================================
# Factory Function
# ============================================================

def build_ema(
        channels: int,
        groups: int = 32,
        kernel_size: int = 3,
        use_residual: bool = True
) -> EMA:
    """
    创建单尺度 EMA。
    """

    return EMA(
        channels=channels,
        groups=groups,
        kernel_size=kernel_size,
        use_residual=use_residual
    )


# ============================================================
# Build Multi-scale EMA
# ============================================================

def build_multiscale_ema(
        channels: Tuple[int, int, int, int] = (
            96,
            192,
            384,
            768
        ),
        groups: int = 32,
        kernel_size: int = 3,
        apply_to: Tuple[str, ...] = (
            "C4",
            "C5"
        ),
        use_residual: bool = True
) -> MultiScaleEMA:
    """
    创建多尺度 EMA。

    默认：

        C2 → Identity
        C3 → Identity
        C4 → EMA
        C5 → EMA
    """

    return MultiScaleEMA(
        channels=channels,
        groups=groups,
        kernel_size=kernel_size,
        apply_to=apply_to,
        use_residual=use_residual
    )


# ============================================================
# Parameter Counter
# ============================================================

def count_parameters(
        model: nn.Module
) -> int:
    """
    计算可训练参数量。
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Single EMA Test
# ============================================================

def test_single_ema():

    print("=" * 70)

    print(
        "Testing Single EMA"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # C4
    # --------------------------------------------------------

    model = EMA(
        channels=384,
        groups=32,
        kernel_size=3,
        use_residual=True
    )

    model.eval()

    x = torch.randn(
        2,
        384,
        40,
        40
    )

    with torch.no_grad():

        y = model(x)

    print(
        f"Input : {tuple(x.shape)}"
    )

    print(
        f"Output: {tuple(y.shape)}"
    )

    params = count_parameters(
        model
    )

    print(
        f"Parameters: "
        f"{params / 1e6:.4f} M"
    )

    assert y.shape == x.shape

    print(
        "Single EMA test passed."
    )

    print("=" * 70)


# ============================================================
# Multi-scale EMA Test
# ============================================================

def test_multiscale_ema():

    print("=" * 70)

    print(
        "Testing Multi-scale EMA"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = build_multiscale_ema(
        channels=(
            96,
            192,
            384,
            768
        ),
        groups=32,
        kernel_size=3,
        apply_to=(
            "C4",
            "C5"
        ),
        use_residual=True
    )

    model.eval()

    # --------------------------------------------------------
    # 模拟 ConvNeXt V2 输出
    # --------------------------------------------------------

    c2 = torch.randn(
        2,
        96,
        160,
        160
    )

    c3 = torch.randn(
        2,
        192,
        80,
        80
    )

    c4 = torch.randn(
        2,
        384,
        40,
        40
    )

    c5 = torch.randn(
        2,
        768,
        20,
        20
    )

    features = [
        c2,
        c3,
        c4,
        c5
    ]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            features
        )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    for i, (
            feature,
            output
    ) in enumerate(
        zip(
            features,
            outputs
        )
    ):

        print(
            f"C{i + 2}: "
            f"{tuple(feature.shape)} "
            f"→ "
            f"{tuple(output.shape)}"
        )

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    for feature, output in zip(
            features,
            outputs
    ):

        assert (
            feature.shape
            == output.shape
        )

    # --------------------------------------------------------
    # 检查 C2/C3 未经过 EMA
    # --------------------------------------------------------

    assert torch.equal(
        outputs[0],
        c2
    )

    assert torch.equal(
        outputs[1],
        c3
    )

    print()

    print(
        "C2: unchanged"
    )

    print(
        "C3: unchanged"
    )

    print(
        "C4: EMA applied"
    )

    print(
        "C5: EMA applied"
    )

    print()

    print(
        "Multi-scale EMA test passed."
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()

    test_single_ema()

    print()

    test_multiscale_ema()