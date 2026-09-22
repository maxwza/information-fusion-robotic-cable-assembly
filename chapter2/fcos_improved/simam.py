# coding=utf-8



import torch
import torch.nn as nn

from typing import List, Tuple, Union


# ============================================================
# SimAM
# ============================================================

class SimAM(nn.Module):
    """
    Simple Attention Module

    SimAM 根据单个神经元相对于同一通道其它神经元的
    响应差异计算能量函数，并将其转换为注意力权重。

    输入：
        [B, C, H, W]

    输出：
        [B, C, H, W]

    参数：
        e_lambda:
            能量函数中的正则化系数。

            常用值：
                1e-4
    """

    def __init__(
            self,
            e_lambda: float = 1e-4
    ):
        super().__init__()

        if e_lambda <= 0:

            raise ValueError(
                "e_lambda must be greater than 0."
            )

        self.e_lambda = e_lambda

        # ----------------------------------------------------
        # SimAM 本身没有需要训练的参数。
        # ----------------------------------------------------

        self.activation = nn.Sigmoid()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # 输入检查
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "SimAM expects a 4D tensor "
                "[B, C, H, W]. "
                f"Got shape: {tuple(x.shape)}"
            )

        # ----------------------------------------------------
        # 空间维度
        # ----------------------------------------------------

        H = x.size(2)

        W = x.size(3)

        # ----------------------------------------------------
        # 空间位置数量
        # ----------------------------------------------------

        n = H * W - 1

        # ----------------------------------------------------
        # 防止 H=W=1 时除零
        # ----------------------------------------------------

        if n <= 0:

            return x

        # ----------------------------------------------------
        # 每个通道的均值
        #
        # [B, C, H, W]
        # →
        # [B, C, 1, 1]
        # ----------------------------------------------------

        x_mean = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # 每个神经元与通道均值之间的平方差
        # ----------------------------------------------------

        x_diff = (
            x - x_mean
        ).pow(2)

        # ----------------------------------------------------
        # 计算空间位置上的方差
        #
        # 使用：
        #
        # mean((x - mean)^2)
        #
        # 再根据 SimAM 的形式进行归一化。
        # ----------------------------------------------------

        variance = (
            x_diff.sum(
                dim=(2, 3),
                keepdim=True
            )
            / n
        )

        # ----------------------------------------------------
        # Energy function
        #
        # E_t =
        #   (x_t - mean)^2
        #   ----------------
        #   4(lambda + variance)
        #   + 0.5
        #
        # 能量越低表示该神经元越具有显著性。
        # ----------------------------------------------------

        energy = (
            x_diff
            / (
                4.0
                * (
                    variance
                    + self.e_lambda
                )
            )
            + 0.5
        )

        # ----------------------------------------------------
        # Energy → Attention
        #
        # 能量低：
        #     sigmoid(1 / E) 较大
        #
        # 能量高：
        #     sigmoid(1 / E) 较小
        # ----------------------------------------------------

        attention = self.activation(
            1.0 / energy
        )

        # ----------------------------------------------------
        # Feature refinement
        # ----------------------------------------------------

        out = (
            x
            * attention
        )

        return out


# ============================================================
# SimAM with Residual
# ============================================================

class SimAMResidual(nn.Module):
    """
    带残差连接的 SimAM。

    结构：

        x
        │
        ├──────────────┐
        │              │
        ↓              │
      SimAM            │
        │              │
        └────── + ─────┘
               │
              out

    注意：
        该模块用于需要保持原始特征稳定性的场景。

    参数：
        e_lambda:
            SimAM 正则化参数。

        residual_weight:
            注意力分支的缩放系数。
    """

    def __init__(
            self,
            e_lambda: float = 1e-4,
            residual_weight: float = 1.0
    ):
        super().__init__()

        self.simam = SimAM(
            e_lambda=e_lambda
        )

        self.residual_weight = (
            residual_weight
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        refined = self.simam(x)

        return (
            x
            + self.residual_weight
            * refined
        )


# ============================================================
# SimAM Feature Refinement
# ============================================================

class SimAMFeatureRefinement(nn.Module):
    """
    SimAM 特征细化模块。

    用于 FPN 中：

        lateral feature
              ↓
            SimAM
              ↓
        refined feature
    """

    def __init__(
            self,
            e_lambda: float = 1e-4,
            use_residual: bool = False
    ):
        super().__init__()

        self.use_residual = (
            use_residual
        )

        if use_residual:

            self.simam = SimAMResidual(
                e_lambda=e_lambda
            )

        else:

            self.simam = SimAM(
                e_lambda=e_lambda
            )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.simam(x)


# ============================================================
# Multi-scale SimAM
# ============================================================

class MultiScaleSimAM(nn.Module):
    """
    多尺度 SimAM。

    输入：

        [
            P3,
            P4,
            P5,
            P6
        ]

    或：

        [
            C2,
            C3,
            C4,
            C5
        ]

    输出：

        与输入数量和尺寸完全一致。

    每个尺度独立计算 SimAM。
    """

    def __init__(
            self,
            num_levels: int = 4,
            e_lambda: float = 1e-4,
            use_residual: bool = False
    ):
        super().__init__()

        if num_levels <= 0:

            raise ValueError(
                "num_levels must be greater than 0."
            )

        self.num_levels = num_levels

        self.e_lambda = e_lambda

        self.use_residual = (
            use_residual
        )

        # ----------------------------------------------------
        # 为每个尺度建立独立 SimAM。
        #
        # SimAM 没有可学习参数，因此各层实际上可以共享，
        # 但这里保持独立实例，使结构更加清晰。
        # ----------------------------------------------------

        self.modules_list = nn.ModuleList()

        for _ in range(
                num_levels
        ):

            self.modules_list.append(
                SimAMFeatureRefinement(
                    e_lambda=e_lambda,
                    use_residual=use_residual
                )
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
            self,
            features: Union[
                List[torch.Tensor],
                Tuple[torch.Tensor, ...]
            ]
    ) -> List[torch.Tensor]:

        if not isinstance(
                features,
                (list, tuple)
        ):

            raise TypeError(
                "MultiScaleSimAM expects "
                "a list or tuple of tensors."
            )

        if len(features) != self.num_levels:

            raise ValueError(
                f"Expected {self.num_levels} "
                f"feature levels, but got "
                f"{len(features)}."
            )

        outputs = []

        for i in range(
                self.num_levels
        ):

            output = self.modules_list[i](
                features[i]
            )

            outputs.append(
                output
            )

        return outputs


# ============================================================
# SimAM-FPN Feature Adapter
# ============================================================

class SimAMFPNRefinement(nn.Module):
    """
    用于 SimAM-FPN 的特征细化封装。

    功能：

        C2/C3/C4/C5
              ↓
          特征输入
              ↓
            SimAM
              ↓
          P3/P4/P5/P6

    注意：
        本类只负责 SimAM 特征细化，
        不负责 FPN 的 lateral convolution、
        top-down fusion 或 P6 生成。

    真正的 FPN 结构由 fpn.py 实现。
    """

    def __init__(
            self,
            e_lambda: float = 1e-4,
            num_levels: int = 4
    ):
        super().__init__()

        self.num_levels = num_levels

        self.simam = MultiScaleSimAM(
            num_levels=num_levels,
            e_lambda=e_lambda,
            use_residual=False
        )

    def forward(
            self,
            features: List[torch.Tensor]
    ) -> List[torch.Tensor]:

        return self.simam(
            features
        )


# ============================================================
# Factory Function
# ============================================================

def build_simam(
        e_lambda: float = 1e-4
) -> SimAM:
    """
    创建单尺度 SimAM。
    """

    return SimAM(
        e_lambda=e_lambda
    )


# ============================================================
# Multi-scale Factory
# ============================================================

def build_multiscale_simam(
        num_levels: int = 4,
        e_lambda: float = 1e-4,
        use_residual: bool = False
) -> MultiScaleSimAM:
    """
    创建多尺度 SimAM。
    """

    return MultiScaleSimAM(
        num_levels=num_levels,
        e_lambda=e_lambda,
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
# Single SimAM Test
# ============================================================

def test_single_simam():

    print("=" * 70)

    print(
        "Testing Single SimAM"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 创建模块
    # --------------------------------------------------------

    model = SimAM(
        e_lambda=1e-4
    )

    model.eval()

    # --------------------------------------------------------
    # 模拟 C4
    # --------------------------------------------------------

    x = torch.randn(
        2,
        384,
        40,
        40
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print(
        f"Input : {tuple(x.shape)}"
    )

    print(
        f"Output: {tuple(y.shape)}"
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model)}"
    )

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    assert (
        y.shape
        == x.shape
    )

    # --------------------------------------------------------
    # 数值检查
    # --------------------------------------------------------

    assert torch.isfinite(
        y
    ).all()

    print()

    print(
        "Shape test passed."
    )

    print(
        "Numerical stability test passed."
    )

    print("=" * 70)


# ============================================================
# Multi-scale SimAM Test
# ============================================================

def test_multiscale_simam():

    print("=" * 70)

    print(
        "Testing Multi-scale SimAM"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = build_multiscale_simam(
        num_levels=4,
        e_lambda=1e-4,
        use_residual=False
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
    # Shape check
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
            f"Level {i + 1}: "
            f"{tuple(feature.shape)} "
            f"→ "
            f"{tuple(output.shape)}"
        )

        assert (
            feature.shape
            == output.shape
        )

        assert torch.isfinite(
            output
        ).all()

    print()

    print(
        "Multi-scale shape test passed."
    )

    print(
        "Multi-scale numerical stability test passed."
    )

    print("=" * 70)


# ============================================================
# Gradient Test
# ============================================================

def test_gradient():

    print("=" * 70)

    print(
        "Testing SimAM Gradient"
    )

    print("=" * 70)

    model = SimAM(
        e_lambda=1e-4
    )

    model.train()

    x = torch.randn(
        2,
        256,
        32,
        32,
        requires_grad=True
    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    # --------------------------------------------------------
    # 检查输入梯度
    # --------------------------------------------------------

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

    print(
        "Gradient test passed."
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()

    test_single_simam()

    print()

    test_multiscale_simam()

    print()

    test_gradient()