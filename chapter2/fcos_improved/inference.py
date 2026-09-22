# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Local imports
# ============================================================

try:
    from postprocess import (
        FCOSPostProcessor,
        scale_boxes_to_original_image,
    )
except ImportError as exc:

    raise ImportError(
        "无法导入 postprocess.py。"
        "请确认 inference.py 与 postprocess.py 位于同一目录。"
    ) from exc


# ============================================================
# Configuration
# ============================================================

DEFAULT_IMAGE_SIZE = 640

DEFAULT_NUM_CLASSES = 12

DEFAULT_SCORE_THRESHOLD = 0.05

DEFAULT_NMS_THRESHOLD = 0.60

DEFAULT_TOP_K = 1000

DEFAULT_MAX_DETECTIONS = 100

DEFAULT_STRIDES = (
    8,
    16,
    32,
    64,
)


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


# ImageNet normalization.
# 如果 config.py 中采用了不同参数，
# 应与训练阶段保持完全一致。
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int = 42):
    """
    设置随机种子。

    推理阶段实际上没有随机训练操作，
    但保持环境稳定有助于复现实验。
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def get_device(
    device: Optional[str] = None,
) -> torch.device:
    """
    获取推理设备。
    """

    if device is not None:

        if device.lower() == "cuda":

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "指定使用 CUDA，但当前环境没有可用 GPU。"
                )

            return torch.device(
                "cuda"
            )

        return torch.device(
            device
        )

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# Image preprocessing
# ============================================================

class ImagePreprocessor:
    """
    图像预处理。

    默认：

        BGR image
            ↓
        RGB
            ↓
        Resize 640×640
            ↓
        [0,1]
            ↓
        ImageNet Normalize
            ↓
        Tensor [1,3,640,640]

    同时保存原图尺寸，
    便于将检测框映射回原始图像。
    """

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        self.image_size = int(
            image_size
        )

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(
            1,
            1,
            3,
        )

        self.std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(
            1,
            1,
            3,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[
        torch.Tensor,
        Tuple[int, int],
    ]:
        """
        输入：

            image:
                BGR uint8

        返回：

            tensor:
                [1,3,H,W]

            original_size:
                (H,W)
        """

        if image is None:

            raise ValueError(
                "输入 image 为 None。"
            )

        if image.ndim != 3:

            raise ValueError(
                "输入图像必须是 H×W×3。"
            )

        original_height = (
            image.shape[0]
        )

        original_width = (
            image.shape[1]
        )

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        resized = cv2.resize(
            rgb,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # ----------------------------------------------------
        # uint8 → float32
        # ----------------------------------------------------

        resized = (
            resized.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        resized = (
            resized
            - self.mean
        ) / self.std

        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            resized
        ).permute(
            2,
            0,
            1,
        ).contiguous()

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        tensor = tensor.unsqueeze(
            0
        )

        return (
            tensor,
            (
                original_height,
                original_width,
            ),
        )


# ============================================================
# Model wrapper
# ============================================================

class ConnectorDetector(nn.Module):
    """
    第二章完整检测模型。

    结构：

        backbone
            ↓
        EMA
            ↓
        FPN
            ↓
        PFNM
            ↓
        FCOS Head
    """

    def __init__(
        self,
        backbone: nn.Module,
        ema: nn.Module,
        fpn: nn.Module,
        pfnm: nn.Module,
        head: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone

        self.ema = ema

        self.fpn = fpn

        self.pfnm = pfnm

        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        process_info=None,
    ):
        """
        完整前向传播。

        process_info：

            工艺信息。

        PFNM 如果需要工艺信息，
        可以在这里传递。
        """

        # ====================================================
        # Backbone
        # ====================================================

        features = self.backbone(
            x
        )

        # ====================================================
        # EMA
        # ====================================================

        features = self.ema(
            features
        )

        # ====================================================
        # FPN
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
        # FCOS Head
        # ====================================================

        predictions = self.head(
            pyramid_features
        )

        return predictions


# ============================================================
# Model builder
# ============================================================

def build_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    构建第二章完整检测网络。

    --------------------------------------------------------
    重要：
    --------------------------------------------------------

    由于你目前是逐个文件建立程序，
    不同文件最终确定的类名可能略有区别。

    因此这里采用“候选类名自动查找”的方式，
    尽量减少后续修改量。
    """

    # --------------------------------------------------------
    # Import modules
    # --------------------------------------------------------

    try:

        import convnextv2

    except ImportError as exc:

        raise ImportError(
            "无法导入 convnextv2.py"
        ) from exc

    try:

        import ema_improved

    except ImportError as exc:

        raise ImportError(
            "无法导入 ema_improved.py"
        ) from exc

    try:

        import fpn

    except ImportError as exc:

        raise ImportError(
            "无法导入 fpn.py"
        ) from exc

    try:

        import pfnm

    except ImportError as exc:

        raise ImportError(
            "无法导入 pfnm.py"
        ) from exc

    try:

        import fcos_head

    except ImportError as exc:

        raise ImportError(
            "无法导入 fcos_head.py"
        ) from exc

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def find_class(
        module,
        candidates: Sequence[str],
        module_name: str,
    ):

        for name in candidates:

            if hasattr(
                module,
                name,
            ):

                cls = getattr(
                    module,
                    name,
                )

                if isinstance(
                    cls,
                    type,
                ):

                    return cls

        raise AttributeError(
            f"{module_name}.py 中没有找到可用类。"
            f"\n尝试过：{list(candidates)}"
        )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    BackboneClass = find_class(
        convnextv2,
        [
            "ConvNeXtV2",
            "ConvNeXtV2Backbone",
            "ConvNeXtV2Tiny",
            "ConvNeXtV2Small",
        ],
        "convnextv2",
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    EMAClass = find_class(
        ema_improved,
        [
            "EMA",
            "EMAImproved",
            "EfficientMultiScaleAttention",
        ],
        "ema_improved",
    )

    # --------------------------------------------------------
    # FPN
    # --------------------------------------------------------

    FPNClass = find_class(
        fpn,
        [
            "FPN",
            "FeaturePyramidNetwork",
            "SimAMFPN",
        ],
        "fpn",
    )

    # --------------------------------------------------------
    # PFNM
    # --------------------------------------------------------

    PFNMClass = find_class(
        pfnm,
        [
            "PFNM",
            "ProcessFeatureNormalizationModule",
            "ProcessAwareFeatureModulation",
        ],
        "pfnm",
    )

    # --------------------------------------------------------
    # FCOS Head
    # --------------------------------------------------------

    HeadClass = find_class(
        fcos_head,
        [
            "FCOSHead",
            "FCOSHeadImproved",
            "ProcessAwareFCOSHead",
        ],
        "fcos_head",
    )

    # --------------------------------------------------------
    # Instantiate backbone
    # --------------------------------------------------------

    backbone = None

    backbone_constructors = [
        lambda: BackboneClass(),
        lambda: BackboneClass(
            num_classes=num_classes
        ),
    ]

    backbone_error = None

    for constructor in (
        backbone_constructors
    ):

        try:

            backbone = constructor()

            break

        except Exception as exc:

            backbone_error = exc

    if backbone is None:

        raise RuntimeError(
            "无法实例化 ConvNeXt V2。"
            f"\n最后错误：{backbone_error}"
        )

    # --------------------------------------------------------
    # Instantiate EMA
    # --------------------------------------------------------

    ema = None

    ema_constructors = [
        lambda: EMAClass(),
        lambda: EMAClass(
            channels=768
        ),
        lambda: EMAClass(
            channel=768
        ),
        lambda: EMAClass(
            channels=[96, 192, 384, 768]
        ),
    ]

    ema_error = None

    for constructor in (
        ema_constructors
    ):

        try:

            ema = constructor()

            break

        except Exception as exc:

            ema_error = exc

    if ema is None:

        raise RuntimeError(
            "无法实例化 EMA。"
            f"\n最后错误：{ema_error}"
        )

    # --------------------------------------------------------
    # Instantiate FPN
    # --------------------------------------------------------

    fpn_module = None

    fpn_constructors = [
        lambda: FPNClass(),
        lambda: FPNClass(
            in_channels=[
                96,
                192,
                384,
                768,
            ]
        ),
        lambda: FPNClass(
            in_channels=[
                96,
                192,
                384,
                768,
            ],
            out_channels=256,
        ),
    ]

    fpn_error = None

    for constructor in (
        fpn_constructors
    ):

        try:

            fpn_module = constructor()

            break

        except Exception as exc:

            fpn_error = exc

    if fpn_module is None:

        raise RuntimeError(
            "无法实例化 FPN。"
            f"\n最后错误：{fpn_error}"
        )

    # --------------------------------------------------------
    # Instantiate PFNM
    # --------------------------------------------------------

    pfnm_module = None

    pfnm_constructors = [
        lambda: PFNMClass(),
        lambda: PFNMClass(
            channels=256
        ),
        lambda: PFNMClass(
            feature_channels=256
        ),
        lambda: PFNMClass(
            in_channels=256,
            out_channels=256,
        ),
    ]

    pfnm_error = None

    for constructor in (
        pfnm_constructors
    ):

        try:

            pfnm_module = constructor()

            break

        except Exception as exc:

            pfnm_error = exc

    if pfnm_module is None:

        raise RuntimeError(
            "无法实例化 PFNM。"
            f"\n最后错误：{pfnm_error}"
        )

    # --------------------------------------------------------
    # Instantiate FCOS Head
    # --------------------------------------------------------

    head = None

    head_constructors = [
        lambda: HeadClass(
            num_classes=num_classes
        ),
        lambda: HeadClass(
            num_classes=num_classes,
            in_channels=256,
        ),
        lambda: HeadClass(
            num_classes=num_classes,
            in_channels=256,
            num_levels=4,
        ),
    ]

    head_error = None

    for constructor in (
        head_constructors
    ):

        try:

            head = constructor()

            break

        except Exception as exc:

            head_error = exc

    if head is None:

        raise RuntimeError(
            "无法实例化 FCOS Head。"
            f"\n最后错误：{head_error}"
        )

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    model = ConnectorDetector(
        backbone=backbone,
        ema=ema,
        fpn=fpn_module,
        pfnm=pfnm_module,
        head=head,
    )

    if device is not None:

        model = model.to(
            device
        )

    return model


# ============================================================
# Checkpoint loading
# ============================================================

def extract_state_dict(
    checkpoint,
):
    """
    从不同格式 checkpoint 中提取 state_dict。

    支持：

        checkpoint
        checkpoint["state_dict"]
        checkpoint["model"]
        checkpoint["model_state_dict"]
    """

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ):

            if key in checkpoint:

                candidate = (
                    checkpoint[key]
                )

                if isinstance(
                    candidate,
                    dict,
                ):

                    return candidate

        # 有些保存方式直接就是 state_dict
        if all(
            isinstance(
                value,
                torch.Tensor,
            )
            for value in checkpoint.values()
        ):

            return checkpoint

    raise RuntimeError(
        "无法从 checkpoint 中找到 state_dict。"
    )


def remove_module_prefix(
    state_dict: Dict[str, torch.Tensor],
):
    """
    去除 DataParallel 保存产生的：

        module.

    前缀。
    """

    new_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            new_key = key[
                len("module.") :
            ]

        else:

            new_key = key

        new_state_dict[
            new_key
        ] = value

    return new_state_dict


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
):
    """
    加载模型权重。
    """

    checkpoint_path = str(
        checkpoint_path
    )

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"找不到 checkpoint："
            f"{checkpoint_path}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Loading checkpoint"
    )

    print(
        "=" * 70
    )

    print(
        f"Path: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )

    state_dict = (
        remove_module_prefix(
            state_dict
        )
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=strict,
        )
    )

    if missing_keys:

        print(
            "\nMissing keys:"
        )

        for key in missing_keys:

            print(
                f"  {key}"
            )

    if unexpected_keys:

        print(
            "\nUnexpected keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

    if not missing_keys and not unexpected_keys:

        print(
            "\nCheckpoint loaded successfully."
        )

    else:

        print(
            "\nCheckpoint loaded with warnings."
        )

    print(
        "=" * 70
    )

    return model


# ============================================================
# Process information
# ============================================================

class ProcessInfoProvider:
    """
    工艺信息接口。

    PFNM 使用的工艺信息包括：

        assembly stage
        process parameter ranges
        contact requirements
        environmental conditions

    这里提供统一接口。

    如果实际部署时工艺信息由数据库、
    process_factory_info.py 或配置文件提供，
    可以在这里替换。

    当前默认返回 None，
    表示由 PFNM 使用内部默认处理。
    """

    def __init__(
        self,
        process_info: Optional[dict] = None,
    ):
        self.process_info = (
            process_info
        )

    def get(
        self,
        image_path: Optional[str] = None,
    ):
        return self.process_info


# ============================================================
# Main inference engine
# ============================================================

class InferenceEngine:
    """
    FCOS Connector Detection 推理引擎。
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_size: int = DEFAULT_IMAGE_SIZE,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        process_info_provider: Optional[
            ProcessInfoProvider
        ] = None,
    ):
        self.model = model

        self.device = device

        self.image_size = int(
            image_size
        )

        self.preprocessor = (
            ImagePreprocessor(
                image_size=self.image_size
            )
        )

        self.postprocessor = (
            FCOSPostProcessor(
                num_classes=DEFAULT_NUM_CLASSES,
                strides=DEFAULT_STRIDES,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                max_detections=max_detections,
                regression_in_feature_units=False,
                class_names=CLASS_NAMES,
            )
        )

        self.process_info_provider = (
            process_info_provider
        )

        self.model.eval()

    # ========================================================
    # Warmup
    # ========================================================

    @torch.no_grad()
    def warmup(
        self,
        iterations: int = 10,
    ):
        """
        GPU warmup。

        用于正式 FPS 测试之前，
        避免 CUDA initialization 影响时间。
        """

        if self.device.type != "cuda":

            return

        dummy = torch.zeros(
            (
                1,
                3,
                self.image_size,
                self.image_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(
            iterations
        ):

            predictions = self.model(
                dummy
            )

            _ = predictions

        torch.cuda.synchronize()

    # ========================================================
    # Forward
    # ========================================================

    @torch.no_grad()
    def forward(
        self,
        tensor: torch.Tensor,
        process_info=None,
    ):
        """
        执行模型前向传播。
        """

        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        predictions = self.model(
            tensor,
            process_info=process_info,
        )

        return predictions

    # ========================================================
    # Single image inference
    # ========================================================

    @torch.no_grad()
    def infer_image(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
    ):
        """
        单张图像推理。

        返回：

            detections
            inference_time
            original_image
        """

        tensor, original_size = (
            self.preprocessor(
                image
            )
        )

        process_info = None

        if (
            self.process_info_provider
            is not None
        ):

            process_info = (
                self.process_info_provider.get(
                    image_path
                )
            )

        # ----------------------------------------------------
        # CUDA synchronize before timing
        # ----------------------------------------------------

        if self.device.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        predictions = self.forward(
            tensor,
            process_info=process_info,
        )

        if self.device.type == "cuda":

            torch.cuda.synchronize()

        inference_time = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Postprocess
        # ----------------------------------------------------

        results = (
            self.postprocessor(
                predictions,
                image_size=(
                    self.image_size,
                    self.image_size,
                ),
            )
        )

        detections = results[0]

        # ----------------------------------------------------
        # Map boxes back to original image
        # ----------------------------------------------------

        detections[
            "boxes"
        ] = scale_boxes_to_original_image(
            detections[
                "boxes"
            ],
            original_size=original_size,
            current_size=(
                self.image_size,
                self.image_size,
            ),
        )

        return (
            detections,
            inference_time,
            original_size,
        )

    # ========================================================
    # File inference
    # ========================================================

    @torch.no_grad()
    def infer_file(
        self,
        image_path: str,
    ):
        """
        从文件读取并推理。
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"无法读取图像："
                f"{image_path}"
            )

        return self.infer_image(
            image,
            image_path=image_path,
        )


# ============================================================
# Visualization
# ============================================================

def draw_detections(
    image: np.ndarray,
    detections: Dict[str, torch.Tensor],
    class_names: Sequence[str] = CLASS_NAMES,
    score_digits: int = 2,
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制检测结果。
    """

    output = image.copy()

    boxes = (
        detections["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        detections["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        detections["labels"]
        .detach()
        .cpu()
        .numpy()
    )

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):

        x1, y1, x2, y2 = (
            [
                int(round(value))
                for value in box
            ]
        )

        label_id = int(
            label
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

        text = (
            f"{class_name} "
            f"{score:.{score_digits}f}"
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (0, 255, 0),
            thickness,
        )

        # ----------------------------------------------------
        # Text background
        # ----------------------------------------------------

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )

        text_x = x1

        text_y = max(
            y1 - 5,
            text_height + baseline + 2,
        )

        cv2.rectangle(
            output,
            (
                text_x,
                text_y
                - text_height
                - baseline,
            ),
            (
                text_x
                + text_width,
                text_y
                + baseline,
            ),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            output,
            text,
            (
                text_x,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# Save JSON
# ============================================================

def save_detection_json(
    detections: Dict[str, torch.Tensor],
    output_path: str,
    class_names: Sequence[str] = CLASS_NAMES,
):
    """
    将检测结果保存为 JSON。
    """

    boxes = (
        detections["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        detections["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        detections["labels"]
        .detach()
        .cpu()
        .numpy()
    )

    records = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):

        label_id = int(
            label
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

        records.append(
            {
                "bbox": [
                    float(
                        box[0]
                    ),
                    float(
                        box[1]
                    ),
                    float(
                        box[2]
                    ),
                    float(
                        box[3]
                    ),
                ],
                "score": float(
                    score
                ),
                "label": label_id,
                "class_name": class_name,
            }
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# Single image command
# ============================================================

def run_single_image(
    engine: InferenceEngine,
    image_path: str,
    output_dir: str,
    save_visualization: bool = True,
    save_json: bool = True,
):
    """
    单张图像推理。
    """

    image_path = Path(
        image_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Single Image Inference"
    )

    print(
        "=" * 70
    )

    print(
        f"Image: {image_path}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"无法读取："
            f"{image_path}"
        )

    (
        detections,
        inference_time,
        original_size,
    ) = engine.infer_image(
        image,
        image_path=str(
            image_path
        ),
    )

    num_detections = (
        detections["boxes"].shape[0]
    )

    fps = (
        1.0
        / max(
            inference_time,
            1e-9,
        )
    )

    print(
        f"Original size: "
        f"{original_size[1]} × "
        f"{original_size[0]}"
    )

    print(
        f"Detections: "
        f"{num_detections}"
    )

    print(
        f"Inference time: "
        f"{inference_time * 1000:.2f} ms"
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    boxes = (
        detections["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        detections["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        detections["labels"]
        .detach()
        .cpu()
        .numpy()
    )

    for index, (
        box,
        score,
        label,
    ) in enumerate(
        zip(
            boxes,
            scores,
            labels,
        )
    ):

        label_id = int(
            label
        )

        class_name = (
            CLASS_NAMES[
                label_id
            ]
            if 0 <= label_id
            < len(CLASS_NAMES)
            else f"class_{label_id}"
        )

        print(
            f"  [{index:02d}] "
            f"{class_name:<6} "
            f"score={score:.4f} "
            f"box="
            f"["
            f"{box[0]:.1f}, "
            f"{box[1]:.1f}, "
            f"{box[2]:.1f}, "
            f"{box[3]:.1f}"
            f"]"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if save_visualization:

        visualization = (
            draw_detections(
                image,
                detections,
                CLASS_NAMES,
            )
        )

        output_image_path = (
            output_dir
            / (
                image_path.stem
                + "_result"
                + image_path.suffix
            )
        )

        cv2.imwrite(
            str(
                output_image_path
            ),
            visualization,
        )

        print(
            f"\nSaved visualization:"
            f"\n{output_image_path}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if save_json:

        output_json_path = (
            output_dir
            / (
                image_path.stem
                + "_result.json"
            )
        )

        save_detection_json(
            detections,
            str(
                output_json_path
            ),
            CLASS_NAMES,
        )

        print(
            f"Saved JSON:"
            f"\n{output_json_path}"
        )

    print(
        "=" * 70
    )

    return detections


# ============================================================
# Collect image files
# ============================================================

def collect_images(
    input_dir: str,
) -> List[Path]:
    """
    收集文件夹中的图像。
    """

    input_dir = Path(
        input_dir
    )

    if not input_dir.exists():

        raise FileNotFoundError(
            f"输入目录不存在："
            f"{input_dir}"
        )

    image_paths = []

    for path in sorted(
        input_dir.iterdir()
    ):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            image_paths.append(
                path
            )

    return image_paths


# ============================================================
# Batch inference
# ============================================================

def run_batch(
    engine: InferenceEngine,
    input_dir: str,
    output_dir: str,
):
    """
    文件夹批量推理。
    """

    image_paths = collect_images(
        input_dir
    )

    if not image_paths:

        print(
            "输入目录中没有找到图像。"
        )

        return

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Batch Inference"
    )

    print(
        "=" * 70
    )

    print(
        f"Input directory : "
        f"{input_dir}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print(
        f"Images          : "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    inference_times = []

    total_detections = 0

    successful_images = 0

    failed_images = 0

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            "\n"
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                raise RuntimeError(
                    "cv2.imread 返回 None"
                )

            (
                detections,
                inference_time,
                _,
            ) = engine.infer_image(
                image,
                image_path=str(
                    image_path
                ),
            )

            inference_times.append(
                inference_time
            )

            num_detections = (
                detections[
                    "boxes"
                ].shape[0]
            )

            total_detections += (
                num_detections
            )

            successful_images += 1

            # ------------------------------------------------
            # Save visualization
            # ------------------------------------------------

            visualization = (
                draw_detections(
                    image,
                    detections,
                    CLASS_NAMES,
                )
            )

            output_image_path = (
                output_dir
                / (
                    image_path.stem
                    + "_result"
                    + image_path.suffix
                )
            )

            cv2.imwrite(
                str(
                    output_image_path
                ),
                visualization,
            )

            # ------------------------------------------------
            # Save JSON
            # ------------------------------------------------

            output_json_path = (
                output_dir
                / (
                    image_path.stem
                    + "_result.json"
                )
            )

            save_detection_json(
                detections,
                str(
                    output_json_path
                ),
                CLASS_NAMES,
            )

            fps = (
                1.0
                / max(
                    inference_time,
                    1e-9,
                )
            )

            print(
                f"  detections: "
                f"{num_detections}"
            )

            print(
                f"  time: "
                f"{inference_time * 1000:.2f} ms"
            )

            print(
                f"  FPS: "
                f"{fps:.2f}"
            )

        except Exception as exc:

            failed_images += 1

            print(
                f"  ERROR: {exc}"
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Batch Inference Summary"
    )

    print(
        "=" * 70
    )

    print(
        f"Total images      : "
        f"{len(image_paths)}"
    )

    print(
        f"Successful images : "
        f"{successful_images}"
    )

    print(
        f"Failed images     : "
        f"{failed_images}"
    )

    print(
        f"Total detections  : "
        f"{total_detections}"
    )

    if inference_times:

        mean_time = (
            float(
                np.mean(
                    inference_times
                )
            )
        )

        std_time = (
            float(
                np.std(
                    inference_times
                )
            )
        )

        mean_fps = (
            1.0
            / max(
                mean_time,
                1e-9,
            )
        )

        print(
            f"Mean inference    : "
            f"{mean_time * 1000:.2f} ms"
        )

        print(
            f"Std inference     : "
            f"{std_time * 1000:.2f} ms"
        )

        print(
            f"Mean FPS          : "
            f"{mean_fps:.2f}"
        )

    print(
        "=" * 70
    )


# ============================================================
# Argument parser
# ============================================================

def build_argument_parser():
    """
    构建命令行参数。
    """

    parser = argparse.ArgumentParser(
        description=(
            "FCOS connector detection inference"
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "输入图像路径或图像目录"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./inference_results",
        help=(
            "推理结果保存目录"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "模型 checkpoint 路径"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "推理设备，例如 cuda / cpu"
        ),
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help=(
            "网络输入尺寸"
        ),
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help=(
            "检测置信度阈值"
        ),
    )

    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.60,
        help=(
            "NMS IoU 阈值"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help=(
            "NMS 前保留的候选框数量"
        ),
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help=(
            "最终最大检测数量"
        ),
    )

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help=(
            "不保存可视化结果"
        ),
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help=(
            "不保存 JSON"
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help=(
            "CUDA warmup 次数"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "随机种子"
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():
    """
    inference.py 主入口。
    """

    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device(
        args.device
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Process-Aware FCOS Inference"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Image size: "
        f"{args.image_size} × "
        f"{args.image_size}"
    )

    print(
        f"Classes: "
        f"{DEFAULT_NUM_CLASSES}"
    )

    print(
        f"Strides: "
        f"{DEFAULT_STRIDES}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        "\nBuilding model..."
    )

    model = build_model(
        num_classes=DEFAULT_NUM_CLASSES,
        device=device,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
        strict=False,
    )

    model.eval()

    # --------------------------------------------------------
    # Engine
    # --------------------------------------------------------

    engine = InferenceEngine(
        model=model,
        device=device,
        image_size=args.image_size,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        max_detections=args.max_detections,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if args.warmup > 0:

        print(
            f"\nCUDA warmup: "
            f"{args.warmup} iterations"
        )

        try:

            engine.warmup(
                args.warmup
            )

        except Exception as exc:

            print(
                "Warmup skipped:"
                f" {exc}"
            )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_path = Path(
        args.input
    )

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    if input_path.is_file():

        run_single_image(
            engine=engine,
            image_path=str(
                input_path
            ),
            output_dir=args.output,
            save_visualization=(
                not args.no_visualization
            ),
            save_json=(
                not args.no_json
            ),
        )

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    elif input_path.is_dir():

        run_batch(
            engine=engine,
            input_dir=str(
                input_path
            ),
            output_dir=args.output,
        )

    else:

        raise FileNotFoundError(
            f"输入路径不存在："
            f"{input_path}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Inference finished."
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()