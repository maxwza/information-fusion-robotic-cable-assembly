# -*- coding: utf-8 -*-


from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .simam import SimAM
except ImportError:
    from simam import SimAM


# ============================================================
# 基础卷积模块
# ============================================================

class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm2d + ReLU

    主要用于：
        - FPN lateral projection
        - FPN output refinement
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: bool = True,
        activation: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# FPN 单层
# ============================================================

class FPNLevel(nn.Module):
    """
    FPN 单层特征处理。

    包含：
        1. lateral 1×1 projection
        2. 3×3 feature refinement
        3. SimAM refinement
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_simam: bool = True,
        simam_e_lambda: float = 1e-4,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 横向连接
        # ----------------------------------------------------
        self.lateral = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=True,
            activation=False,
        )

        # ----------------------------------------------------
        # FPN 3×3 平滑卷积
        # ----------------------------------------------------
        self.output = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=True,
            activation=True,
        )

        # ----------------------------------------------------
        # SimAM
        # ----------------------------------------------------
        self.use_simam = use_simam

        if use_simam:
            self.simam = SimAM(
                e_lambda=simam_e_lambda
            )
        else:
            self.simam = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.lateral(x)

        x = self.output(x)

        if self.use_simam:
            x = self.simam(x)

        return x


# ============================================================
# P6 生成模块
# ============================================================

class P6Downsample(nn.Module):
    """
    从 P5 生成 P6。

    P5:
        stride = 32

    P6:
        stride = 64

    使用：
        3×3 stride=2 convolution
    """

    def __init__(
        self,
        channels: int,
        use_simam: bool = True,
        simam_e_lambda: float = 1e-4,
    ):
        super().__init__()

        self.conv = ConvBNAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=True,
            activation=True,
        )

        self.use_simam = use_simam

        if use_simam:
            self.simam = SimAM(
                e_lambda=simam_e_lambda
            )
        else:
            self.simam = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        if self.use_simam:
            x = self.simam(x)

        return x


# ============================================================
# SimAM-FPN
# ============================================================

class SimAMFPN(nn.Module):
    """
    SimAM-FPN。

    输入：
        C3
        C4
        C5

    输出：
        P3
        P4
        P5
        P6

    默认：
        C3 = 192
        C4 = 384
        C5 = 768

        FPN channels = 256

    对于 640×640 输入：

        C3 = [B, 192, 80, 80]
        C4 = [B, 384, 40, 40]
        C5 = [B, 768, 20, 20]

        P3 = [B, 256, 80, 80]
        P4 = [B, 256, 40, 40]
        P5 = [B, 256, 20, 20]
        P6 = [B, 256, 10, 10]
    """

    def __init__(
        self,
        in_channels: List[int] = None,
        out_channels: int = 256,
        use_simam: bool = True,
        simam_e_lambda: float = 1e-4,
        use_extra_p6: bool = True,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 默认 ConvNeXt V2 Tiny 的 C3/C4/C5
        # ----------------------------------------------------
        if in_channels is None:
            in_channels = [192, 384, 768]

        if len(in_channels) != 3:
            raise ValueError(
                "SimAMFPN 的 in_channels 必须包含 "
                "C3、C4、C5 三个特征层。"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_simam = use_simam
        self.use_extra_p6 = use_extra_p6

        # ====================================================
        # C3 -> P3
        # ====================================================

        self.p3 = FPNLevel(
            in_channels=in_channels[0],
            out_channels=out_channels,
            use_simam=use_simam,
            simam_e_lambda=simam_e_lambda,
        )

        # ====================================================
        # C4 -> P4
        # ====================================================

        self.p4 = FPNLevel(
            in_channels=in_channels[1],
            out_channels=out_channels,
            use_simam=use_simam,
            simam_e_lambda=simam_e_lambda,
        )

        # ====================================================
        # C5 -> P5
        # ====================================================

        self.p5 = FPNLevel(
            in_channels=in_channels[2],
            out_channels=out_channels,
            use_simam=use_simam,
            simam_e_lambda=simam_e_lambda,
        )

        # ====================================================
        # P5 -> P6
        # ====================================================

        if use_extra_p6:
            self.p6 = P6Downsample(
                channels=out_channels,
                use_simam=use_simam,
                simam_e_lambda=simam_e_lambda,
            )
        else:
            self.p6 = None

    # ========================================================
    # Top-down 融合
    # ========================================================

    @staticmethod
    def _upsample_add(
        higher: torch.Tensor,
        lateral: torch.Tensor,
    ) -> torch.Tensor:
        """
        将高层特征上采样后与低层 lateral feature 相加。

        higher：
            空间尺寸较小。

        lateral：
            空间尺寸较大。
        """

        higher = F.interpolate(
            higher,
            size=lateral.shape[-2:],
            mode="nearest",
        )

        return higher + lateral

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        features: Union[
            List[torch.Tensor],
            Tuple[torch.Tensor, ...],
            Dict[str, torch.Tensor],
        ],
    ) -> Dict[str, torch.Tensor]:
        """
        参数
        ----
        features:
            可以是：

            方式一：
                [C2, C3, C4, C5]

            方式二：
                [C3, C4, C5]

            方式三：
                {
                    "C2": C2,
                    "C3": C3,
                    "C4": C4,
                    "C5": C5
                }

            如果输入包含 C2，则自动忽略 C2。

        返回
        ----
        {
            "P3": P3,
            "P4": P4,
            "P5": P5,
            "P6": P6
        }
        """

        # ====================================================
        # 解析输入
        # ====================================================

        if isinstance(features, dict):

            if not all(
                key in features
                for key in ["C3", "C4", "C5"]
            ):
                raise KeyError(
                    "FPN 输入字典必须包含 C3、C4、C5。"
                )

            c3 = features["C3"]
            c4 = features["C4"]
            c5 = features["C5"]

        elif isinstance(features, (list, tuple)):

            if len(features) == 4:
                # ConvNeXt V2 标准输出：
                # C2, C3, C4, C5
                _, c3, c4, c5 = features

            elif len(features) == 3:
                # 直接输入：
                # C3, C4, C5
                c3, c4, c5 = features

            else:
                raise ValueError(
                    "FPN 输入必须包含 "
                    "3 个或 4 个特征层："
                    "C3,C4,C5 或 C2,C3,C4,C5。"
                )

        else:
            raise TypeError(
                "features 必须是 list、tuple 或 dict。"
            )

        # ====================================================
        # 输入维度检查
        # ====================================================

        self._check_feature(
            c3,
            self.in_channels[0],
            "C3",
        )

        self._check_feature(
            c4,
            self.in_channels[1],
            "C4",
        )

        self._check_feature(
            c5,
            self.in_channels[2],
            "C5",
        )

        # ====================================================
        # Bottom-up lateral projection
        # ====================================================

        p3_lateral = self.p3.lateral(c3)

        p4_lateral = self.p4.lateral(c4)

        p5_lateral = self.p5.lateral(c5)

        # ====================================================
        # Top-down pathway
        #
        # P5 -> P4 -> P3
        # ====================================================

        p4_td = self._upsample_add(
            p5_lateral,
            p4_lateral,
        )

        p3_td = self._upsample_add(
            p4_td,
            p3_lateral,
        )

        # ====================================================
        # 3×3 refinement + SimAM
        # ====================================================

        p3 = self.p3.output(p3_td)

        if self.use_simam:
            p3 = self.p3.simam(p3)

        p4 = self.p4.output(p4_td)

        if self.use_simam:
            p4 = self.p4.simam(p4)

        p5 = self.p5.output(p5_lateral)

        if self.use_simam:
            p5 = self.p5.simam(p5)

        # ====================================================
        # P6
        # ====================================================

        if self.use_extra_p6:

            p6 = self.p6(p5)

        else:

            p6 = p5

        # ====================================================
        # 输出
        # ====================================================

        outputs = {
            "P3": p3,
            "P4": p4,
            "P5": p5,
            "P6": p6,
        }

        return outputs

    # ========================================================
    # Feature validation
    # ========================================================

    @staticmethod
    def _check_feature(
        x: torch.Tensor,
        expected_channels: int,
        name: str,
    ):

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{name} 必须是 torch.Tensor。"
            )

        if x.ndim != 4:
            raise ValueError(
                f"{name} 必须为 4D Tensor，"
                f"当前 shape={tuple(x.shape)}"
            )

        if x.shape[1] != expected_channels:
            raise ValueError(
                f"{name} 通道数错误："
                f"期望 {expected_channels}，"
                f"实际 {x.shape[1]}。"
            )


# ============================================================
# 标准 FPN
# ============================================================

class FPN(SimAMFPN):
    """
    不使用 SimAM 的标准 FPN。

    用于：
        - 基线实验
        - 消融实验
        - 与 SimAM-FPN 进行对比
    """

    def __init__(
        self,
        in_channels: List[int] = None,
        out_channels: int = 256,
        use_extra_p6: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            use_simam=False,
            simam_e_lambda=1e-4,
            use_extra_p6=use_extra_p6,
        )


# ============================================================
# FPN 工厂函数
# ============================================================

def build_fpn(
    in_channels: List[int] = None,
    out_channels: int = 256,
    use_simam: bool = True,
    simam_e_lambda: float = 1e-4,
    use_extra_p6: bool = True,
) -> SimAMFPN:
    """
    构建 FPN。

    默认构建：
        SimAM-FPN
    """

    if in_channels is None:
        in_channels = [192, 384, 768]

    return SimAMFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        use_simam=use_simam,
        simam_e_lambda=simam_e_lambda,
        use_extra_p6=use_extra_p6,
    )


# ============================================================
# 输出信息
# ============================================================

def print_feature_shapes(
    features: Dict[str, torch.Tensor]
):
    """
    打印 P3-P6 的特征尺寸。
    """

    print("\n" + "=" * 70)
    print("SimAM-FPN Feature Shapes")
    print("=" * 70)

    for level in ["P3", "P4", "P5", "P6"]:

        if level not in features:
            continue

        x = features[level]

        print(
            f"{level}: "
            f"shape={tuple(x.shape)}, "
            f"dtype={x.dtype}, "
            f"device={x.device}"
        )

    print("=" * 70)


# ============================================================
# 单元测试
# ============================================================

def test_fpn():

    print("\n" + "=" * 70)
    print("Testing SimAM-FPN")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 2

    # --------------------------------------------------------
    # 模拟 640×640 输入经过 ConvNeXt V2 Tiny 后的特征
    # --------------------------------------------------------

    c2 = torch.randn(
        batch_size,
        96,
        160,
        160,
        device=device,
    )

    c3 = torch.randn(
        batch_size,
        192,
        80,
        80,
        device=device,
    )

    c4 = torch.randn(
        batch_size,
        384,
        40,
        40,
        device=device,
    )

    c5 = torch.randn(
        batch_size,
        768,
        20,
        20,
        device=device,
    )

    # --------------------------------------------------------
    # 构建 FPN
    # --------------------------------------------------------

    model = build_fpn(
        in_channels=[192, 384, 768],
        out_channels=256,
        use_simam=True,
        simam_e_lambda=1e-4,
        use_extra_p6=True,
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            [c2, c3, c4, c5]
        )

    # --------------------------------------------------------
    # 检查输出
    # --------------------------------------------------------

    print_feature_shapes(outputs)

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

    for level, expected in expected_shapes.items():

        actual = tuple(outputs[level].shape)

        assert actual == expected, (
            f"{level} shape 错误："
            f"expected={expected}, "
            f"actual={actual}"
        )

    print("\n所有输出尺寸检查通过。")

    # --------------------------------------------------------
    # 梯度测试
    # --------------------------------------------------------

    model.train()

    c3_test = torch.randn(
        1,
        192,
        80,
        80,
        device=device,
        requires_grad=True,
    )

    c4_test = torch.randn(
        1,
        384,
        40,
        40,
        device=device,
        requires_grad=True,
    )

    c5_test = torch.randn(
        1,
        768,
        20,
        20,
        device=device,
        requires_grad=True,
    )

    outputs = model(
        [c3_test, c4_test, c5_test]
    )

    loss = sum(
        value.mean()
        for value in outputs.values()
    )

    loss.backward()

    assert c3_test.grad is not None
    assert c4_test.grad is not None
    assert c5_test.grad is not None

    print("梯度反向传播检查通过。")

    print("\n" + "=" * 70)
    print("SimAM-FPN Test Passed")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_fpn()