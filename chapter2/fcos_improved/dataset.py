# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import torch
from torch.utils.data import Dataset


# ============================================================
# Constants
# ============================================================

DEFAULT_CLASSES = [
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


DEFAULT_MEAN = (
    0.485,
    0.456,
    0.406,
)

DEFAULT_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# Utility functions
# ============================================================

def load_json(
    json_path: str,
) -> Dict[str, Any]:
    """
    读取 JSON 文件。
    """

    json_path = str(json_path)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"COCO annotation 文件不存在：{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def clip_boxes(
    boxes: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """
    将 bounding boxes 限制在图像范围内。

    输入：
        [N,4]

    输出：
        [N,4]
    """

    if boxes.size == 0:
        return boxes.reshape(
            0,
            4,
        )

    boxes = boxes.copy()

    boxes[:, 0] = np.clip(
        boxes[:, 0],
        0,
        width,
    )

    boxes[:, 1] = np.clip(
        boxes[:, 1],
        0,
        height,
    )

    boxes[:, 2] = np.clip(
        boxes[:, 2],
        0,
        width,
    )

    boxes[:, 3] = np.clip(
        boxes[:, 3],
        0,
        height,
    )

    return boxes


def remove_invalid_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    min_box_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    删除无效 bounding boxes。

    条件：

        width > min_box_size
        height > min_box_size
    """

    if boxes.size == 0:

        return (
            boxes.reshape(0, 4),
            labels.reshape(0),
        )

    widths = (
        boxes[:, 2]
        - boxes[:, 0]
    )

    heights = (
        boxes[:, 3]
        - boxes[:, 1]
    )

    valid = (
        (widths >= min_box_size)
        & (heights >= min_box_size)
    )

    return (
        boxes[valid],
        labels[valid],
    )


def xywh_to_xyxy(
    box: Sequence[float],
) -> List[float]:
    """
    COCO：

        [x, y, width, height]

    转换为：

        [x1, y1, x2, y2]
    """

    x, y, w, h = [
        float(v)
        for v in box[:4]
    ]

    return [
        x,
        y,
        x + w,
        y + h,
    ]


# ============================================================
# Image transformations
# ============================================================

class Compose:
    """
    简单的检测数据增强组合器。

    每个 transform 接收：

        image, boxes, labels

    返回：

        image, boxes, labels
    """

    def __init__(
        self,
        transforms: Sequence,
    ):
        self.transforms = list(
            transforms
        )

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):
        for transform in self.transforms:

            image, boxes, labels = transform(
                image,
                boxes,
                labels,
            )

        return (
            image,
            boxes,
            labels,
        )


class RandomHorizontalFlip:
    """
    随机水平翻转。
    """

    def __init__(
        self,
        probability: float = 0.5,
    ):
        self.probability = probability

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        if random.random() >= self.probability:
            return (
                image,
                boxes,
                labels,
            )

        width, _ = image.size

        image = ImageOps.mirror(
            image
        )

        if boxes.size > 0:

            boxes = boxes.copy()

            old_x1 = boxes[:, 0].copy()
            old_x2 = boxes[:, 2].copy()

            boxes[:, 0] = (
                width - old_x2
            )

            boxes[:, 2] = (
                width - old_x1
            )

        return (
            image,
            boxes,
            labels,
        )


class RandomColorJitter:
    """
    随机颜色增强。

    对应 config.py：

        color_jitter = True

    不改变 bounding box。
    """

    def __init__(
        self,
        probability: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        self.probability = probability
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _adjust_hue(
        self,
        image: Image.Image,
        factor: float,
    ) -> Image.Image:
        """
        使用 HSV 进行简单 hue 调整。
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        hsv = np.asarray(
            image.convert("HSV")
        ).astype(
            np.uint8
        )

        hue_shift = int(
            factor * 255
        )

        hsv[:, :, 0] = (
            hsv[:, :, 0].astype(
                np.int16
            )
            + hue_shift
        ) % 256

        hsv = hsv.astype(
            np.uint8
        )

        return Image.fromarray(
            hsv,
            mode="HSV",
        ).convert("RGB")

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        if random.random() >= self.probability:

            return (
                image,
                boxes,
                labels,
            )

        if self.brightness > 0:

            factor = random.uniform(
                max(
                    0.0,
                    1.0 - self.brightness,
                ),
                1.0 + self.brightness,
            )

            image = ImageEnhance.Brightness(
                image
            ).enhance(
                factor
            )

        if self.contrast > 0:

            factor = random.uniform(
                max(
                    0.0,
                    1.0 - self.contrast,
                ),
                1.0 + self.contrast,
            )

            image = ImageEnhance.Contrast(
                image
            ).enhance(
                factor
            )

        if self.saturation > 0:

            factor = random.uniform(
                max(
                    0.0,
                    1.0 - self.saturation,
                ),
                1.0 + self.saturation,
            )

            image = ImageEnhance.Color(
                image
            ).enhance(
                factor
            )

        if self.hue > 0:

            factor = random.uniform(
                -self.hue,
                self.hue,
            )

            image = self._adjust_hue(
                image,
                factor,
            )

        return (
            image,
            boxes,
            labels,
        )


class RandomRotation:
    """
    随机旋转。

    默认：

        ±5°

    使用 expand=True。

    bounding box 通过四个角点变换后重新计算。
    """

    def __init__(
        self,
        degrees: float = 5.0,
        probability: float = 0.5,
    ):
        self.degrees = float(
            degrees
        )

        self.probability = (
            probability
        )

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        if (
            random.random()
            >= self.probability
        ):
            return (
                image,
                boxes,
                labels,
            )

        angle = random.uniform(
            -self.degrees,
            self.degrees,
        )

        if abs(angle) < 1e-8:

            return (
                image,
                boxes,
                labels,
            )

        width, height = image.size

        # ----------------------------------------------------
        # Rotate image
        # ----------------------------------------------------

        rotated = image.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=True,
            fillcolor=(0, 0, 0),
        )

        new_width, new_height = (
            rotated.size
        )

        # ----------------------------------------------------
        # Rotation matrix
        # ----------------------------------------------------

        theta = math.radians(
            angle
        )

        cos_theta = math.cos(
            theta
        )

        sin_theta = math.sin(
            theta
        )

        cx = width / 2.0
        cy = height / 2.0

        new_cx = (
            new_width / 2.0
        )

        new_cy = (
            new_height / 2.0
        )

        if boxes.size == 0:

            return (
                rotated,
                boxes,
                labels,
            )

        transformed_boxes = []

        for box in boxes:

            x1, y1, x2, y2 = box

            corners = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.float32,
            )

            transformed_points = []

            for px, py in corners:

                dx = (
                    px - cx
                )

                dy = (
                    py - cy
                )

                # PIL rotate uses the
                # mathematical positive
                # rotation convention.
                rx = (
                    dx * cos_theta
                    + dy * sin_theta
                )

                ry = (
                    -dx * sin_theta
                    + dy * cos_theta
                )

                nx = (
                    rx + new_cx
                )

                ny = (
                    ry + new_cy
                )

                transformed_points.append(
                    [
                        nx,
                        ny,
                    ]
                )

            transformed_points = np.asarray(
                transformed_points,
                dtype=np.float32,
            )

            new_x1 = transformed_points[
                :, 0
            ].min()

            new_y1 = transformed_points[
                :, 1
            ].min()

            new_x2 = transformed_points[
                :, 0
            ].max()

            new_y2 = transformed_points[
                :, 1
            ].max()

            transformed_boxes.append(
                [
                    new_x1,
                    new_y1,
                    new_x2,
                    new_y2,
                ]
            )

        boxes = np.asarray(
            transformed_boxes,
            dtype=np.float32,
        )

        boxes = clip_boxes(
            boxes,
            new_width,
            new_height,
        )

        return (
            rotated,
            boxes,
            labels,
        )


class RandomScale:
    """
    随机缩放。

    默认范围：

        0.8 ~ 1.2

    缩放后将图像放置在同样比例的新画布中。

    注意：
        这里不是最终 640×640 resize。
        最终尺寸由 ResizeAndNormalize 完成。
    """

    def __init__(
        self,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        probability: float = 0.5,
    ):
        self.scale_min = float(
            scale_min
        )

        self.scale_max = float(
            scale_max
        )

        self.probability = (
            probability
        )

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        if (
            random.random()
            >= self.probability
        ):

            return (
                image,
                boxes,
                labels,
            )

        scale = random.uniform(
            self.scale_min,
            self.scale_max,
        )

        width, height = image.size

        new_width = max(
            1,
            int(round(width * scale)),
        )

        new_height = max(
            1,
            int(round(height * scale)),
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            resample=Image.BILINEAR,
        )

        if boxes.size > 0:

            boxes = boxes.copy()

            boxes *= scale

        return (
            image,
            boxes,
            labels,
        )


class Resize:
    """
    将图像直接 resize 到固定尺寸。

    默认：

        640 × 640

    同时按 x/y 比例调整 bbox。
    """

    def __init__(
        self,
        size: Tuple[int, int] = (
            640,
            640,
        ),
    ):
        self.size = (
            int(size[0]),
            int(size[1]),
        )

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        old_width, old_height = (
            image.size
        )

        new_width, new_height = (
            self.size
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            resample=Image.BILINEAR,
        )

        if boxes.size > 0:

            boxes = boxes.copy()

            scale_x = (
                new_width
                / old_width
            )

            scale_y = (
                new_height
                / old_height
            )

            boxes[:, [0, 2]] *= (
                scale_x
            )

            boxes[:, [1, 3]] *= (
                scale_y
            )

        return (
            image,
            boxes,
            labels,
        )


class ToTensor:
    """
    PIL Image -> torch.Tensor。

    输出：

        [3,H,W]

    范围：

        0~1
    """

    def __call__(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        image = image.convert(
            "RGB"
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array = array / 255.0

        tensor = torch.from_numpy(
            array
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        return (
            tensor,
            boxes,
            labels,
        )


class Normalize:
    """
    ImageNet normalization。
    """

    def __init__(
        self,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
    ):
        self.mean = torch.tensor(
            mean,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        self.std = torch.tensor(
            std,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

    def __call__(
        self,
        image: torch.Tensor,
        boxes: np.ndarray,
        labels: np.ndarray,
    ):

        image = (
            image - self.mean
        ) / self.std

        return (
            image,
            boxes,
            labels,
        )


# ============================================================
# COCO Dataset
# ============================================================

class ConnectorCocoDataset(Dataset):
    """
    连接器 COCO Dataset。

    目录示例：

        dataset/
        ├── train/
        │   ├── images/
        │   └── annotations.json
        │
        ├── val/
        │   ├── images/
        │   └── annotations.json
        │
        └── test/
            ├── images/
            └── annotations.json

    或：

        images/
            xxx.jpg
            yyy.jpg

        annotations/
            instances_train.json
            instances_val.json
            instances_test.json

    只要 image_dir 和 annotation_file
    指向正确位置即可。
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        image_size: int = 640,
        classes: Optional[
            Sequence[str]
        ] = None,
        transforms=None,
        training: bool = False,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        min_box_size: float = 1.0,
        filter_crowd: bool = True,
    ):
        super().__init__()

        self.image_dir = Path(
            image_dir
        )

        self.annotation_file = Path(
            annotation_file
        )

        self.image_size = int(
            image_size
        )

        self.classes = list(
            classes
            if classes is not None
            else DEFAULT_CLASSES
        )

        self.training = bool(
            training
        )

        self.min_box_size = float(
            min_box_size
        )

        self.filter_crowd = bool(
            filter_crowd
        )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"图像目录不存在："
                f"{self.image_dir}"
            )

        if not self.annotation_file.exists():

            raise FileNotFoundError(
                f"COCO annotation 不存在："
                f"{self.annotation_file}"
            )

        # ----------------------------------------------------
        # Load COCO JSON
        # ----------------------------------------------------

        self.coco = load_json(
            str(
                self.annotation_file
            )
        )

        # ----------------------------------------------------
        # Build category mapping
        # ----------------------------------------------------

        self.category_id_to_label = (
            self._build_category_mapping()
        )

        self.label_to_category_id = {
            label: category_id
            for category_id, label
            in self.category_id_to_label.items()
        }

        # ----------------------------------------------------
        # Build image index
        # ----------------------------------------------------

        self.images = self._build_image_index()

        # ----------------------------------------------------
        # Build annotation index
        # ----------------------------------------------------

        self.annotations = (
            self._build_annotation_index()
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if transforms is None:

            self.transforms = (
                build_default_transforms(
                    image_size=self.image_size,
                    training=self.training,
                    mean=mean,
                    std=std,
                )
            )

        else:

            self.transforms = transforms

    # ========================================================
    # Category mapping
    # ========================================================

    def _build_category_mapping(
        self,
    ) -> Dict[int, int]:
        """
        建立：

            COCO category_id
                ↓
            contiguous label 0~11

        优先按照 classes 中的类别名称顺序映射。

        例如：

            C1-2 -> 0
            C1-4 -> 1
            ...
            C4-6 -> 11
        """

        categories = self.coco.get(
            "categories",
            [],
        )

        if not categories:

            raise ValueError(
                "COCO JSON 中不存在 categories。"
            )

        category_name_to_id = {}

        for category in categories:

            category_id = int(
                category["id"]
            )

            name = str(
                category["name"]
            )

            category_name_to_id[
                name
            ] = category_id

        mapping = {}

        # ----------------------------------------------------
        # Strict class check
        # ----------------------------------------------------

        missing_classes = [
            name
            for name in self.classes
            if name not in category_name_to_id
        ]

        if missing_classes:

            raise ValueError(
                "COCO categories 缺少以下论文定义的类别："
                f"{missing_classes}\n"
                f"当前 categories："
                f"{list(category_name_to_id.keys())}"
            )

        for label, class_name in enumerate(
            self.classes
        ):

            category_id = (
                category_name_to_id[
                    class_name
                ]
            )

            mapping[
                category_id
            ] = label

        return mapping

    # ========================================================
    # Image index
    # ========================================================

    def _build_image_index(
        self,
    ) -> List[Dict[str, Any]]:
        """
        建立 image index。
        """

        images = self.coco.get(
            "images",
            [],
        )

        if not images:

            raise ValueError(
                "COCO JSON 中不存在 images。"
            )

        result = []

        for image_info in images:

            image_id = int(
                image_info["id"]
            )

            file_name = str(
                image_info["file_name"]
            )

            width = int(
                image_info.get(
                    "width",
                    0,
                )
            )

            height = int(
                image_info.get(
                    "height",
                    0,
                )
            )

            result.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )

        return result

    # ========================================================
    # Annotation index
    # ========================================================

    def _build_annotation_index(
        self,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        建立：

            image_id
                ↓
            annotations
        """

        annotation_index = {}

        for image_info in self.images:

            annotation_index[
                image_info["id"]
            ] = []

        annotations = self.coco.get(
            "annotations",
            [],
        )

        for annotation in annotations:

            image_id = int(
                annotation["image_id"]
            )

            if image_id not in annotation_index:
                continue

            # ------------------------------------------------
            # Ignore crowd
            # ------------------------------------------------

            if self.filter_crowd:

                if int(
                    annotation.get(
                        "iscrowd",
                        0,
                    )
                ) == 1:

                    continue

            category_id = int(
                annotation["category_id"]
            )

            # ------------------------------------------------
            # Ignore categories outside
            # the 12 defined classes
            # ------------------------------------------------

            if (
                category_id
                not in self.category_id_to_label
            ):
                continue

            annotation_index[
                image_id
            ].append(
                annotation
            )

        return annotation_index

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.images
        )

    # ========================================================
    # Image path
    # ========================================================

    def _get_image_path(
        self,
        file_name: str,
    ) -> Path:
        """
        支持 COCO file_name：

            xxx.jpg

        或：

            subfolder/xxx.jpg
        """

        path = (
            self.image_dir
            / file_name
        )

        if path.exists():
            return path

        # ----------------------------------------------------
        # Windows / Linux path separator
        # ----------------------------------------------------

        normalized = file_name.replace(
            "\\",
            "/",
        )

        path = (
            self.image_dir
            / normalized
        )

        if path.exists():
            return path

        raise FileNotFoundError(
            f"找不到图像：{path}"
        )

    # ========================================================
    # Read annotations
    # ========================================================

    def _load_target(
        self,
        image_info: Dict[str, Any],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        读取单张图片 GT。

        返回：

            boxes:
                [N,4]

            labels:
                [N]
        """

        image_id = image_info[
            "id"
        ]

        annotations = self.annotations.get(
            image_id,
            [],
        )

        boxes = []
        labels = []

        for annotation in annotations:

            bbox = annotation.get(
                "bbox",
                None,
            )

            if bbox is None:
                continue

            if len(bbox) < 4:
                continue

            box = xywh_to_xyxy(
                bbox
            )

            category_id = int(
                annotation[
                    "category_id"
                ]
            )

            if (
                category_id
                not in self.category_id_to_label
            ):
                continue

            label = self.category_id_to_label[
                category_id
            ]

            boxes.append(
                box
            )

            labels.append(
                label
            )

        if boxes:

            boxes = np.asarray(
                boxes,
                dtype=np.float32,
            )

            labels = np.asarray(
                labels,
                dtype=np.int64,
            )

        else:

            boxes = np.empty(
                (
                    0,
                    4,
                ),
                dtype=np.float32,
            )

            labels = np.empty(
                (
                    0,
                ),
                dtype=np.int64,
            )

        return (
            boxes,
            labels,
        )

    # ========================================================
    # __getitem__
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ):
        """
        获取单张图片。

        返回：

            image
            target
        """

        image_info = self.images[
            index
        ]

        image_id = image_info[
            "id"
        ]

        file_name = image_info[
            "file_name"
        ]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_path = (
            self._get_image_path(
                file_name
            )
        )

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as exc:

            raise RuntimeError(
                f"读取图像失败："
                f"{image_path}"
            ) from exc

        orig_width, orig_height = (
            image.size
        )

        # ----------------------------------------------------
        # Load GT
        # ----------------------------------------------------

        boxes, labels = (
            self._load_target(
                image_info
            )
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        image, boxes, labels = (
            self.transforms(
                image,
                boxes,
                labels,
            )
        )

        # ----------------------------------------------------
        # Ensure tensor
        # ----------------------------------------------------

        if not isinstance(
            image,
            torch.Tensor,
        ):

            raise TypeError(
                "Dataset transform 最终必须输出 torch.Tensor。"
            )

        # ----------------------------------------------------
        # Ensure boxes
        # ----------------------------------------------------

        if boxes.size > 0:

            boxes = clip_boxes(
                boxes,
                self.image_size,
                self.image_size,
            )

            boxes, labels = (
                remove_invalid_boxes(
                    boxes,
                    labels,
                    min_box_size=self.min_box_size,
                )
            )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        boxes_tensor = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        ).reshape(
            -1,
            4,
        )

        labels_tensor = torch.as_tensor(
            labels,
            dtype=torch.long,
        )

        image_id_tensor = torch.tensor(
            [image_id],
            dtype=torch.long,
        )

        orig_size_tensor = torch.tensor(
            [
                orig_height,
                orig_width,
            ],
            dtype=torch.long,
        )

        current_size_tensor = torch.tensor(
            [
                image.shape[1],
                image.shape[2],
            ],
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "orig_size": orig_size_tensor,
            "size": current_size_tensor,
            "file_name": file_name,
        }

        return (
            image,
            target,
        )

    # ========================================================
    # Dataset information
    # ========================================================

    def get_class_names(
        self,
    ) -> List[str]:

        return list(
            self.classes
        )

    def get_category_mapping(
        self,
    ) -> Dict[int, int]:

        return dict(
            self.category_id_to_label
        )

    def get_class_distribution(
        self,
    ) -> Dict[str, int]:
        """
        统计当前 split 的类别实例数量。
        """

        distribution = {
            name: 0
            for name in self.classes
        }

        for image_info in self.images:

            image_id = image_info[
                "id"
            ]

            for annotation in self.annotations.get(
                image_id,
                [],
            ):

                category_id = int(
                    annotation[
                        "category_id"
                    ]
                )

                if (
                    category_id
                    in self.category_id_to_label
                ):

                    label = (
                        self.category_id_to_label[
                            category_id
                        ]
                    )

                    distribution[
                        self.classes[label]
                    ] += 1

        return distribution


# ============================================================
# Default transform builder
# ============================================================

def build_default_transforms(
    image_size: int = 640,
    training: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
):
    """
    根据训练/验证/测试状态构建 transform。

    Train：

        RandomHorizontalFlip
        RandomRotation
        RandomScale
        ColorJitter
        Resize
        ToTensor
        Normalize

    Val/Test：

        Resize
        ToTensor
        Normalize
    """

    transforms = []

    if training:

        transforms.append(
            RandomHorizontalFlip(
                probability=0.5,
            )
        )

        transforms.append(
            RandomRotation(
                degrees=5.0,
                probability=0.5,
            )
        )

        transforms.append(
            RandomScale(
                scale_min=0.8,
                scale_max=1.2,
                probability=0.5,
            )
        )

        transforms.append(
            RandomColorJitter(
                probability=0.5,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )
        )

    transforms.append(
        Resize(
            size=(
                image_size,
                image_size,
            )
        )
    )

    transforms.append(
        ToTensor()
    )

    transforms.append(
        Normalize(
            mean=mean,
            std=std,
        )
    )

    return Compose(
        transforms
    )


# ============================================================
# Dataset factory
# ============================================================

def build_connector_dataset(
    image_dir: str,
    annotation_file: str,
    image_size: int = 640,
    training: bool = False,
    classes: Optional[
        Sequence[str]
    ] = None,
):
    """
    构建 Connector COCO Dataset。
    """

    return ConnectorCocoDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        image_size=image_size,
        classes=classes,
        training=training,
    )


# ============================================================
# DataLoader collate function
# ============================================================

def detection_collate_fn(
    batch,
):
    """
    Detection Dataset 专用 collate_fn。

    普通 torch.stack 无法处理：

        image 1:
            N1 objects

        image 2:
            N2 objects

    因此：

        images -> stack

        targets -> list

    返回：

        images:
            [B,3,H,W]

        targets:
            List[Dict]
    """

    images = []
    targets = []

    for image, target in batch:

        images.append(
            image
        )

        targets.append(
            target
        )

    images = torch.stack(
        images,
        dim=0,
    )

    return (
        images,
        targets,
    )


# ============================================================
# Dataset statistics
# ============================================================

def print_dataset_statistics(
    dataset: ConnectorCocoDataset,
):
    """
    打印数据集统计信息。
    """

    print("\n")
    print("=" * 70)
    print("Connector Dataset Statistics")
    print("=" * 70)

    print(
        f"Images       : {len(dataset)}"
    )

    print(
        f"Classes      : {len(dataset.classes)}"
    )

    print(
        f"Image size   : "
        f"{dataset.image_size} × "
        f"{dataset.image_size}"
    )

    print("\nCategory mapping:")

    for category_id, label in sorted(
        dataset.category_id_to_label.items()
    ):

        print(
            f"  category_id {category_id:>3} "
            f"-> label {label:>2} "
            f"-> {dataset.classes[label]}"
        )

    print("\nInstance distribution:")

    distribution = (
        dataset.get_class_distribution()
    )

    total_instances = 0

    for class_name, count in (
        distribution.items()
    ):

        print(
            f"  {class_name:<6}: "
            f"{count}"
        )

        total_instances += count

    print(
        f"\nTotal instances: "
        f"{total_instances}"
    )

    print("=" * 70)


# ============================================================
# Sample checker
# ============================================================

def check_sample(
    dataset: ConnectorCocoDataset,
    index: int = 0,
):
    """
    检查单个 Dataset sample。
    """

    image, target = dataset[
        index
    ]

    print("\n")
    print("=" * 70)
    print("Dataset Sample Check")
    print("=" * 70)

    print(
        f"Index      : {index}"
    )

    print(
        f"Image      : "
        f"{tuple(image.shape)}"
    )

    print(
        f"Image dtype: "
        f"{image.dtype}"
    )

    print(
        f"Image min  : "
        f"{image.min().item():.4f}"
    )

    print(
        f"Image max  : "
        f"{image.max().item():.4f}"
    )

    print(
        f"Boxes      : "
        f"{tuple(target['boxes'].shape)}"
    )

    print(
        f"Labels     : "
        f"{tuple(target['labels'].shape)}"
    )

    print(
        f"Image ID   : "
        f"{target['image_id'].tolist()}"
    )

    print(
        f"Original   : "
        f"{target['orig_size'].tolist()}"
    )

    print(
        f"Current    : "
        f"{target['size'].tolist()}"
    )

    if target["boxes"].numel() > 0:

        print("\nBoxes:")

        print(
            target["boxes"]
        )

        print("\nLabels:")

        print(
            target["labels"]
        )

    else:

        print(
            "\nNo objects in this image."
        )

    print("=" * 70)


# ============================================================
# DataLoader test
# ============================================================

def test_dataloader(
    dataset: ConnectorCocoDataset,
    batch_size: int = 2,
    num_workers: int = 0,
):
    """
    测试 Dataset + DataLoader。
    """

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    images, targets = next(
        iter(loader)
    )

    print("\n")
    print("=" * 70)
    print("DataLoader Test")
    print("=" * 70)

    print(
        f"Images shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Targets: "
        f"{len(targets)}"
    )

    for i, target in enumerate(
        targets
    ):

        print(
            f"  Sample {i}: "
            f"boxes="
            f"{tuple(target['boxes'].shape)}, "
            f"labels="
            f"{tuple(target['labels'].shape)}"
        )

    print("=" * 70)


# ============================================================
# Synthetic dataset test
# ============================================================

def test_transform_pipeline():
    """
    不依赖真实数据集的 transform 测试。
    """

    print("\n")
    print("=" * 70)
    print("Testing Transform Pipeline")
    print("=" * 70)

    image = Image.new(
        "RGB",
        (
            1280,
            720,
        ),
        color=(
            120,
            120,
            120,
        ),
    )

    boxes = np.asarray(
        [
            [
                200.0,
                150.0,
                500.0,
                450.0,
            ],
            [
                700.0,
                250.0,
                1000.0,
                550.0,
            ],
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [
            0,
            5,
        ],
        dtype=np.int64,
    )

    transforms = (
        build_default_transforms(
            image_size=640,
            training=True,
        )
    )

    image_out, boxes_out, labels_out = (
        transforms(
            image,
            boxes,
            labels,
        )
    )

    print(
        f"Output image: "
        f"{tuple(image_out.shape)}"
    )

    print(
        f"Output boxes: "
        f"{boxes_out.shape}"
    )

    print(
        f"Output labels: "
        f"{labels_out.shape}"
    )

    assert image_out.shape == (
        3,
        640,
        640,
    )

    assert boxes_out.shape[1] == 4

    assert labels_out.ndim == 1

    assert np.all(
        boxes_out[:, 0]
        <= boxes_out[:, 2]
    )

    assert np.all(
        boxes_out[:, 1]
        <= boxes_out[:, 3]
    )

    assert np.all(
        boxes_out >= 0
    )

    assert np.all(
        boxes_out[:, [0, 2]]
        <= 640
    )

    assert np.all(
        boxes_out[:, [1, 3]]
        <= 640
    )

    print(
        "\nTransform test passed."
    )

    print("=" * 70)


# ============================================================
# Empty target test
# ============================================================

def test_empty_target():
    """
    测试无目标图像的处理逻辑。
    """

    print("\n")
    print("=" * 70)
    print("Testing Empty Target")
    print("=" * 70)

    image = Image.new(
        "RGB",
        (
            640,
            480,
        ),
        color=(
            100,
            100,
            100,
        ),
    )

    boxes = np.empty(
        (
            0,
            4,
        ),
        dtype=np.float32,
    )

    labels = np.empty(
        (
            0,
        ),
        dtype=np.int64,
    )

    transforms = (
        build_default_transforms(
            image_size=640,
            training=False,
        )
    )

    image_out, boxes_out, labels_out = (
        transforms(
            image,
            boxes,
            labels,
        )
    )

    print(
        f"Image: "
        f"{tuple(image_out.shape)}"
    )

    print(
        f"Boxes: "
        f"{boxes_out.shape}"
    )

    print(
        f"Labels: "
        f"{labels_out.shape}"
    )

    assert image_out.shape == (
        3,
        640,
        640,
    )

    assert boxes_out.shape == (
        0,
        4,
    )

    assert labels_out.shape == (
        0,
    )

    print(
        "\nEmpty target test passed."
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Independent transform tests
    # --------------------------------------------------------

    test_transform_pipeline()

    test_empty_target()

    # --------------------------------------------------------
    # Real dataset test
    # --------------------------------------------------------
    #
    # 修改下面路径后取消注释。
    #
    # dataset = build_connector_dataset(
    #     image_dir=r"D:\your_dataset\train\images",
    #     annotation_file=r"D:\your_dataset\train\annotations.json",
    #     image_size=640,
    #     training=True,
    # )
    #
    # print_dataset_statistics(
    #     dataset
    # )
    #
    # check_sample(
    #     dataset,
    #     index=0,
    # )
    #
    # test_dataloader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=0,
    # )

    print("\n")
    print("=" * 70)
    print("dataset.py standalone tests passed.")
    print("=" * 70)