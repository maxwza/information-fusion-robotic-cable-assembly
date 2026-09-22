# -*- coding: utf-8 -*-


from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 类型定义
# ============================================================

Number = Union[int, float, np.number]

VectorLike = Union[
    Sequence[Number],
    np.ndarray,
    torch.Tensor,
]


# ============================================================
# 工艺信息数据结构
# ============================================================

@dataclass
class ProcessFactoryInfo:
    """
    单个样本的工艺工厂信息。

    参数
    ----
    assembly_stage:
        装配阶段，4维。

    process_parameters:
        工艺参数范围，4维。

    contact_requirements:
        接触要求，4维。

    environment:
        环境条件，4维。

    sample_id:
        可选样本编号。

    metadata:
        可选辅助信息。

    注意：
        metadata 仅用于记录，不参与 PFNM 数值编码。
    """

    assembly_stage: Sequence[Number]
    process_parameters: Sequence[Number]
    contact_requirements: Sequence[Number]
    environment: Sequence[Number]

    sample_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。
        """

        return {
            "sample_id": self.sample_id,
            "assembly_stage": list(
                self.assembly_stage
            ),
            "process_parameters": list(
                self.process_parameters
            ),
            "contact_requirements": list(
                self.contact_requirements
            ),
            "environment": list(
                self.environment
            ),
            "metadata": self.metadata,
        }


# ============================================================
# 工艺信息编码器
# ============================================================

class ProcessFactoryInfoEncoder:
    """
    工艺工厂信息编码器。

    将四类工艺语义信息：

        Assembly Stage
        Process Parameters
        Contact Requirements
        Environment

    拼接为统一的 16 维向量。

    默认：

        stage_dim = 4
        parameter_dim = 4
        contact_dim = 4
        environment_dim = 4

        total_dim = 16
    """

    def __init__(
        self,
        assembly_stage_dim: int = 4,
        process_parameter_dim: int = 4,
        contact_requirement_dim: int = 4,
        environment_dim: int = 4,
        normalize: bool = True,
        normalization_mode: str = "minmax",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.assembly_stage_dim = (
            assembly_stage_dim
        )

        self.process_parameter_dim = (
            process_parameter_dim
        )

        self.contact_requirement_dim = (
            contact_requirement_dim
        )

        self.environment_dim = (
            environment_dim
        )

        self.normalize = normalize

        self.normalization_mode = (
            normalization_mode
        )

        self.eps = eps

        self.total_dim = (
            assembly_stage_dim
            + process_parameter_dim
            + contact_requirement_dim
            + environment_dim
        )

        if self.total_dim <= 0:
            raise ValueError(
                "工艺信息总维度必须大于 0。"
            )

        if normalization_mode not in {
            "minmax",
            "standard",
            "none",
        }:
            raise ValueError(
                "normalization_mode 必须为 "
                "'minmax'、'standard' 或 'none'。"
            )

        if not normalize:
            self.normalization_mode = "none"

        # ----------------------------------------------------
        # 归一化参数
        #
        # shape:
        #     [16]
        # ----------------------------------------------------

        self.registered_min = np.zeros(
            self.total_dim,
            dtype=np.float32,
        )

        self.registered_max = np.ones(
            self.total_dim,
            dtype=np.float32,
        )

        self.registered_mean = np.zeros(
            self.total_dim,
            dtype=np.float32,
        )

        self.registered_std = np.ones(
            self.total_dim,
            dtype=np.float32,
        )

        self.is_fitted = False

    # ========================================================
    # 向量转换
    # ========================================================

    @staticmethod
    def _to_numpy(
        value: VectorLike,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        将输入转换为一维 numpy 数组。
        """

        if isinstance(value, torch.Tensor):

            value = (
                value.detach()
                .cpu()
                .numpy()
            )

        array = np.asarray(
            value,
            dtype=dtype,
        )

        if array.ndim == 0:
            array = array.reshape(1)

        return array.reshape(-1)

    # ========================================================
    # 维度检查
    # ========================================================

    @staticmethod
    def _validate_dimension(
        value: np.ndarray,
        expected_dim: int,
        name: str,
    ):
        """
        检查单项工艺信息维度。
        """

        if value.size != expected_dim:

            raise ValueError(
                f"{name} 维度错误："
                f"期望 {expected_dim}，"
                f"实际 {value.size}。"
            )

    # ========================================================
    # 单样本拼接
    # ========================================================

    def encode_single(
        self,
        process_info: ProcessFactoryInfo,
    ) -> np.ndarray:
        """
        编码单个工艺信息样本。

        返回：

            [16]
        """

        stage = self._to_numpy(
            process_info.assembly_stage
        )

        parameters = self._to_numpy(
            process_info.process_parameters
        )

        contact = self._to_numpy(
            process_info.contact_requirements
        )

        environment = self._to_numpy(
            process_info.environment
        )

        # ----------------------------------------------------
        # 维度检查
        # ----------------------------------------------------

        self._validate_dimension(
            stage,
            self.assembly_stage_dim,
            "assembly_stage",
        )

        self._validate_dimension(
            parameters,
            self.process_parameter_dim,
            "process_parameters",
        )

        self._validate_dimension(
            contact,
            self.contact_requirement_dim,
            "contact_requirements",
        )

        self._validate_dimension(
            environment,
            self.environment_dim,
            "environment",
        )

        # ----------------------------------------------------
        # 拼接
        # ----------------------------------------------------

        vector = np.concatenate(
            [
                stage,
                parameters,
                contact,
                environment,
            ],
            axis=0,
        ).astype(
            np.float32
        )

        if vector.size != self.total_dim:

            raise RuntimeError(
                "工艺信息编码后的维度错误："
                f"expected={self.total_dim}, "
                f"actual={vector.size}"
            )

        return vector

    # ========================================================
    # 从字典编码
    # ========================================================

    def encode_dict(
        self,
        data: Mapping[str, Any],
    ) -> np.ndarray:
        """
        从字典编码单个样本。

        支持：

        {
            "assembly_stage": [...],
            "process_parameters": [...],
            "contact_requirements": [...],
            "environment": [...]
        }
        """

        required_keys = [
            "assembly_stage",
            "process_parameters",
            "contact_requirements",
            "environment",
        ]

        missing = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing:
            raise KeyError(
                "缺少工艺信息字段："
                + ", ".join(missing)
            )

        info = ProcessFactoryInfo(
            assembly_stage=data[
                "assembly_stage"
            ],
            process_parameters=data[
                "process_parameters"
            ],
            contact_requirements=data[
                "contact_requirements"
            ],
            environment=data[
                "environment"
            ],
            sample_id=data.get(
                "sample_id",
                None,
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

        return self.encode_single(info)

    # ========================================================
    # 批量编码
    # ========================================================

    def encode_batch(
        self,
        batch: Sequence[
            Union[
                ProcessFactoryInfo,
                Mapping[str, Any],
            ]
        ],
    ) -> np.ndarray:
        """
        批量编码。

        输入：
            N 个工艺信息样本。

        输出：
            [N, 16]
        """

        vectors = []

        for item in batch:

            if isinstance(
                item,
                ProcessFactoryInfo,
            ):
                vector = self.encode_single(
                    item
                )

            elif isinstance(
                item,
                Mapping,
            ):
                vector = self.encode_dict(
                    item
                )

            else:
                raise TypeError(
                    "batch 中的元素必须是 "
                    "ProcessFactoryInfo 或 dict。"
                )

            vectors.append(vector)

        if len(vectors) == 0:

            return np.empty(
                (
                    0,
                    self.total_dim,
                ),
                dtype=np.float32,
            )

        return np.stack(
            vectors,
            axis=0,
        )

    # ========================================================
    # 拟合归一化参数
    # ========================================================

    def fit(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Sequence[
                Union[
                    ProcessFactoryInfo,
                    Mapping[str, Any],
                ]
            ],
        ],
    ):
        """
        根据训练集工艺信息计算归一化参数。

        注意：
            归一化参数只能使用训练集计算，
            不应使用验证集或测试集统计量。
        """

        matrix = self._prepare_matrix(data)

        if matrix.shape[0] == 0:
            raise ValueError(
                "fit() 接收到空工艺信息数据。"
            )

        if matrix.shape[1] != self.total_dim:
            raise ValueError(
                "工艺信息矩阵维度错误："
                f"expected={self.total_dim}, "
                f"actual={matrix.shape[1]}"
            )

        if self.normalization_mode == "minmax":

            self.registered_min = np.min(
                matrix,
                axis=0,
            )

            self.registered_max = np.max(
                matrix,
                axis=0,
            )

        elif self.normalization_mode == "standard":

            self.registered_mean = np.mean(
                matrix,
                axis=0,
            )

            self.registered_std = np.std(
                matrix,
                axis=0,
            )

            self.registered_std = np.maximum(
                self.registered_std,
                self.eps,
            )

        self.is_fitted = True

        return self

    # ========================================================
    # 数据矩阵准备
    # ========================================================

    def _prepare_matrix(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Sequence[
                Union[
                    ProcessFactoryInfo,
                    Mapping[str, Any],
                ]
            ],
        ],
    ) -> np.ndarray:

        if isinstance(
            data,
            torch.Tensor,
        ):

            matrix = (
                data.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        elif isinstance(
            data,
            np.ndarray,
        ):

            matrix = data.astype(
                np.float32
            )

        elif isinstance(
            data,
            Sequence,
        ):

            matrix = self.encode_batch(
                data
            )

        else:

            raise TypeError(
                "不支持的工艺信息数据类型。"
            )

        if matrix.ndim == 1:

            matrix = matrix.reshape(
                1,
                -1,
            )

        if matrix.ndim != 2:

            raise ValueError(
                "工艺信息矩阵必须为二维："
                "[N, D]"
            )

        return matrix

    # ========================================================
    # 归一化
    # ========================================================

    def transform(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Sequence[
                Union[
                    ProcessFactoryInfo,
                    Mapping[str, Any],
                ]
            ],
        ],
    ) -> np.ndarray:
        """
        对工艺信息进行归一化。

        返回：
            [N, 16]
        """

        matrix = self._prepare_matrix(data)

        if matrix.shape[1] != self.total_dim:

            raise ValueError(
                "输入工艺信息维度错误："
                f"expected={self.total_dim}, "
                f"actual={matrix.shape[1]}"
            )

        if self.normalization_mode == "none":

            return matrix.astype(
                np.float32
            )

        if not self.is_fitted:

            raise RuntimeError(
                "归一化器尚未 fit。"
                "请先使用训练集调用 fit()。"
            )

        if self.normalization_mode == "minmax":

            denominator = (
                self.registered_max
                - self.registered_min
            )

            denominator = np.maximum(
                denominator,
                self.eps,
            )

            matrix = (
                matrix
                - self.registered_min
            ) / denominator

            matrix = np.clip(
                matrix,
                0.0,
                1.0,
            )

        elif self.normalization_mode == "standard":

            matrix = (
                matrix
                - self.registered_mean
            ) / self.registered_std

        return matrix.astype(
            np.float32
        )

    # ========================================================
    # fit + transform
    # ========================================================

    def fit_transform(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Sequence[
                Union[
                    ProcessFactoryInfo,
                    Mapping[str, Any],
                ]
            ],
        ],
    ) -> np.ndarray:
        """
        在训练数据上拟合并转换。
        """

        self.fit(data)

        return self.transform(data)

    # ========================================================
    # Tensor 输出
    # ========================================================

    def encode_tensor(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Sequence[
                Union[
                    ProcessFactoryInfo,
                    Mapping[str, Any],
                ]
            ],
        ],
        device: Optional[
            Union[str, torch.device]
        ] = None,
    ) -> torch.Tensor:
        """
        编码为 PyTorch Tensor。

        输出：

            [B, 16]
        """

        matrix = self.transform(data)

        tensor = torch.from_numpy(
            matrix
        ).float()

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    # ========================================================
    # 保存归一化参数
    # ========================================================

    def save(
        self,
        path: str,
    ):
        """
        保存编码器配置及归一化参数。
        """

        directory = os.path.dirname(
            os.path.abspath(path)
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        state = {
            "assembly_stage_dim":
                self.assembly_stage_dim,

            "process_parameter_dim":
                self.process_parameter_dim,

            "contact_requirement_dim":
                self.contact_requirement_dim,

            "environment_dim":
                self.environment_dim,

            "total_dim":
                self.total_dim,

            "normalize":
                self.normalize,

            "normalization_mode":
                self.normalization_mode,

            "eps":
                self.eps,

            "registered_min":
                self.registered_min.tolist(),

            "registered_max":
                self.registered_max.tolist(),

            "registered_mean":
                self.registered_mean.tolist(),

            "registered_std":
                self.registered_std.tolist(),

            "is_fitted":
                self.is_fitted,
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                state,
                f,
                ensure_ascii=False,
                indent=4,
            )

    # ========================================================
    # 加载归一化参数
    # ========================================================

    def load(
        self,
        path: str,
    ):
        """
        加载编码器配置及归一化参数。
        """

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"找不到工艺信息编码器文件：{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            state = json.load(f)

        if state["total_dim"] != self.total_dim:

            raise ValueError(
                "加载的工艺信息维度与当前编码器不一致："
                f"saved={state['total_dim']}, "
                f"current={self.total_dim}"
            )

        self.registered_min = np.asarray(
            state["registered_min"],
            dtype=np.float32,
        )

        self.registered_max = np.asarray(
            state["registered_max"],
            dtype=np.float32,
        )

        self.registered_mean = np.asarray(
            state["registered_mean"],
            dtype=np.float32,
        )

        self.registered_std = np.asarray(
            state["registered_std"],
            dtype=np.float32,
        )

        self.is_fitted = bool(
            state["is_fitted"]
        )

        return self


# ============================================================
# 工艺信息默认构造器
# ============================================================

class DefaultProcessFactory:
    """
    默认工艺信息构造器。

    用于在数据集没有直接提供结构化工艺参数时，
    根据预先定义的工艺语义生成固定长度的工艺向量。

    注意：
        这里的默认值只是代码接口示例。

        实际训练时，应将这些值替换为你的真实工艺数据库
        或与你论文实验对应的工艺条件。

    默认输出：

        assembly_stage        -> 4
        process_parameters    -> 4
        contact_requirements  -> 4
        environment           -> 4
    """

    def __init__(self):

        self.encoder = ProcessFactoryInfoEncoder(
            assembly_stage_dim=4,
            process_parameter_dim=4,
            contact_requirement_dim=4,
            environment_dim=4,
            normalize=False,
        )

    # ========================================================
    # 默认装配阶段
    # ========================================================

    @staticmethod
    def assembly_stage(
        stage: Union[int, float]
    ) -> np.ndarray:
        """
        将装配阶段编码为 4 维 one-hot。

        stage:
            0 / 1 / 2 / 3

        如果输入为 1~4，则自动转换为 0~3。
        """

        stage_value = int(stage)

        if 1 <= stage_value <= 4:
            stage_value -= 1

        if not 0 <= stage_value < 4:

            raise ValueError(
                "assembly stage 必须属于 "
                "{0,1,2,3} 或 {1,2,3,4}。"
            )

        vector = np.zeros(
            4,
            dtype=np.float32,
        )

        vector[stage_value] = 1.0

        return vector

    # ========================================================
    # 默认工艺参数
    # ========================================================

    @staticmethod
    def process_parameters(
        insertion_depth: float = 10.0,
        approach_speed: float = 0.05,
        retract_distance: float = 10.0,
        alignment_tolerance: float = 5.0,
    ) -> np.ndarray:
        """
        构造 4 维工艺参数。

        参数：

            insertion_depth
                插入深度。

            approach_speed
                接近速度。

            retract_distance
                异常接触后的回撤距离。

            alignment_tolerance
                对准角度容差。

        注意：
            单位需要与你实际实验系统保持一致。
        """

        return np.asarray(
            [
                insertion_depth,
                approach_speed,
                retract_distance,
                alignment_tolerance,
            ],
            dtype=np.float32,
        )

    # ========================================================
    # 默认接触要求
    # ========================================================

    @staticmethod
    def contact_requirements(
        normal_force: float = 15.0,
        force_lower_limit: float = 10.0,
        force_upper_limit: float = 20.0,
        lateral_force_limit: float = 5.0,
    ) -> np.ndarray:
        """
        构造 4 维接触要求。

        参数：

            normal_force
                标称法向接触力。

            force_lower_limit
                法向力下限。

            force_upper_limit
                法向力上限。

            lateral_force_limit
                横向接触力阈值。
        """

        return np.asarray(
            [
                normal_force,
                force_lower_limit,
                force_upper_limit,
                lateral_force_limit,
            ],
            dtype=np.float32,
        )

    # ========================================================
    # 默认环境条件
    # ========================================================

    @staticmethod
    def environment(
        illumination: float = 0.5,
        clutter: float = 0.5,
        visibility: float = 1.0,
        workspace_constraint: float = 0.5,
    ) -> np.ndarray:
        """
        构造 4 维环境条件。

        所有参数建议在实际数据库中统一定义量纲。

        默认：

            illumination
            clutter
            visibility
            workspace_constraint
        """

        return np.asarray(
            [
                illumination,
                clutter,
                visibility,
                workspace_constraint,
            ],
            dtype=np.float32,
        )

    # ========================================================
    # 创建完整工艺信息
    # ========================================================

    def create(
        self,
        stage: Union[int, float] = 1,
        insertion_depth: float = 10.0,
        approach_speed: float = 0.05,
        retract_distance: float = 10.0,
        alignment_tolerance: float = 5.0,
        normal_force: float = 15.0,
        force_lower_limit: float = 10.0,
        force_upper_limit: float = 20.0,
        lateral_force_limit: float = 5.0,
        illumination: float = 0.5,
        clutter: float = 0.5,
        visibility: float = 1.0,
        workspace_constraint: float = 0.5,
    ) -> ProcessFactoryInfo:
        """
        创建一个完整的 ProcessFactoryInfo。
        """

        return ProcessFactoryInfo(
            assembly_stage=self.assembly_stage(
                stage
            ),

            process_parameters=self.process_parameters(
                insertion_depth=insertion_depth,
                approach_speed=approach_speed,
                retract_distance=retract_distance,
                alignment_tolerance=alignment_tolerance,
            ),

            contact_requirements=self.contact_requirements(
                normal_force=normal_force,
                force_lower_limit=force_lower_limit,
                force_upper_limit=force_upper_limit,
                lateral_force_limit=lateral_force_limit,
            ),

            environment=self.environment(
                illumination=illumination,
                clutter=clutter,
                visibility=visibility,
                workspace_constraint=workspace_constraint,
            ),
        )


# ============================================================
# PFNM 输入接口
# ============================================================

class ProcessFactoryInfoModule(nn.Module):
    """
    PyTorch 工艺信息模块。

    作用：
        在训练或推理过程中，将工艺信息统一转换为 Tensor。

    输出：

        [B, 16]

    注意：
        该模块本身不负责复杂的特征学习。

        具体的工艺语义映射和特征调制由后面的
        fpnm.py 完成。
    """

    def __init__(
        self,
        feature_dim: int = 16,
        normalize: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.normalize = normalize

        # ----------------------------------------------------
        # 输入检查
        # ----------------------------------------------------

        if feature_dim <= 0:

            raise ValueError(
                "feature_dim 必须大于 0。"
            )

    def forward(
        self,
        process_info: Union[
            torch.Tensor,
            np.ndarray,
            Sequence[Number],
        ],
    ) -> torch.Tensor:
        """
        输入：

            [B, 16]
            或
            [16]

        输出：

            [B, 16]
        """

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        if isinstance(
            process_info,
            torch.Tensor,
        ):

            x = process_info

        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        elif isinstance(
            process_info,
            np.ndarray,
        ):

            x = torch.from_numpy(
                process_info
            )

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        else:

            x = torch.tensor(
                process_info,
                dtype=torch.float32,
            )

        # ----------------------------------------------------
        # dtype
        # ----------------------------------------------------

        x = x.float()

        # ----------------------------------------------------
        # 维度
        # ----------------------------------------------------

        if x.ndim == 1:

            x = x.unsqueeze(0)

        if x.ndim != 2:

            raise ValueError(
                "process_info 必须是 "
                "[B, D] 或 [D]。"
            )

        if x.shape[-1] != self.feature_dim:

            raise ValueError(
                "process_info 特征维度错误："
                f"expected={self.feature_dim}, "
                f"actual={x.shape[-1]}"
            )

        # ----------------------------------------------------
        # 可选归一化
        #
        # 注意：
        #     这里只进行样本内部的尺度约束。
        #     正式训练时推荐使用
        #     ProcessFactoryInfoEncoder.fit()
        #     得到训练集统计量。
        # ----------------------------------------------------

        if self.normalize:

            x_min = x.min(
                dim=-1,
                keepdim=True,
            ).values

            x_max = x.max(
                dim=-1,
                keepdim=True,
            ).values

            denominator = (
                x_max - x_min
            ).clamp_min(1e-6)

            x = (
                x - x_min
            ) / denominator

        return x


# ============================================================
# 工艺信息数据库
# ============================================================

class ProcessFactoryDatabase:
    """
    简单的工艺信息数据库。

    作用：
        根据 sample_id 保存和读取工艺信息。

    示例：

        database.add(
            "sample_001",
            process_info
        )

        info = database.get(
            "sample_001"
        )
    """

    def __init__(self):

        self.database: Dict[
            str,
            ProcessFactoryInfo
        ] = {}

    # ========================================================
    # 添加
    # ========================================================

    def add(
        self,
        sample_id: str,
        process_info: ProcessFactoryInfo,
    ):
        if not sample_id:

            raise ValueError(
                "sample_id 不能为空。"
            )

        process_info.sample_id = sample_id

        self.database[
            sample_id
        ] = process_info

    # ========================================================
    # 获取
    # ========================================================

    def get(
        self,
        sample_id: str,
    ) -> ProcessFactoryInfo:

        if sample_id not in self.database:

            raise KeyError(
                f"数据库中不存在 sample_id："
                f"{sample_id}"
            )

        return self.database[
            sample_id
        ]

    # ========================================================
    # 判断
    # ========================================================

    def contains(
        self,
        sample_id: str,
    ) -> bool:

        return sample_id in self.database

    # ========================================================
    # 删除
    # ========================================================

    def remove(
        self,
        sample_id: str,
    ):

        if sample_id in self.database:

            del self.database[
                sample_id
            ]

    # ========================================================
    # 数量
    # ========================================================

    def __len__(self):

        return len(
            self.database
        )

    # ========================================================
    # 保存 JSON
    # ========================================================

    def save(
        self,
        path: str,
    ):
        """
        保存整个工艺数据库。
        """

        directory = os.path.dirname(
            os.path.abspath(path)
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        data = {
            sample_id: info.to_dict()
            for sample_id, info
            in self.database.items()
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
            )

    # ========================================================
    # 加载 JSON
    # ========================================================

    def load(
        self,
        path: str,
    ):
        """
        加载工艺数据库。
        """

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"找不到工艺数据库：{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        self.database.clear()

        for sample_id, item in data.items():

            info = ProcessFactoryInfo(
                assembly_stage=item[
                    "assembly_stage"
                ],

                process_parameters=item[
                    "process_parameters"
                ],

                contact_requirements=item[
                    "contact_requirements"
                ],

                environment=item[
                    "environment"
                ],

                sample_id=sample_id,

                metadata=item.get(
                    "metadata",
                    {},
                ),
            )

            self.add(
                sample_id,
                info,
            )

        return self


# ============================================================
# 默认 PFNM 工艺信息
# ============================================================

def build_default_process_info(
    stage: int = 1,
) -> ProcessFactoryInfo:
    """
    快速构造默认工艺信息。
    """

    factory = DefaultProcessFactory()

    return factory.create(
        stage=stage,
    )


# ============================================================
# 工艺信息转 Tensor
# ============================================================

def process_info_to_tensor(
    process_info: ProcessFactoryInfo,
    encoder: Optional[
        ProcessFactoryInfoEncoder
    ] = None,
    device: Optional[
        Union[str, torch.device]
    ] = None,
) -> torch.Tensor:
    """
    将单个 ProcessFactoryInfo 转换成 Tensor。

    输出：

        [1, 16]
    """

    if encoder is None:

        encoder = ProcessFactoryInfoEncoder(
            normalize=False
        )

    tensor = encoder.encode_tensor(
        [process_info],
        device=device,
    )

    return tensor


# ============================================================
# 测试
# ============================================================

def test_process_factory_info():
    """
    模块完整测试。
    """

    print("\n" + "=" * 70)
    print("Testing process_factory_info.py")
    print("=" * 70)

    # ========================================================
    # 1. 创建默认工艺信息
    # ========================================================

    factory = DefaultProcessFactory()

    info = factory.create(
        stage=1,
        insertion_depth=10.0,
        approach_speed=0.05,
        retract_distance=10.0,
        alignment_tolerance=5.0,
        normal_force=15.0,
        force_lower_limit=10.0,
        force_upper_limit=20.0,
        lateral_force_limit=5.0,
        illumination=0.5,
        clutter=0.5,
        visibility=1.0,
        workspace_constraint=0.5,
    )

    print("\n单个工艺信息：")
    print(info.to_dict())

    # ========================================================
    # 2. 创建编码器
    # ========================================================

    encoder = ProcessFactoryInfoEncoder(
        assembly_stage_dim=4,
        process_parameter_dim=4,
        contact_requirement_dim=4,
        environment_dim=4,
        normalize=False,
    )

    # ========================================================
    # 3. 单样本编码
    # ========================================================

    vector = encoder.encode_single(
        info
    )

    print("\n单样本编码：")
    print(
        "shape =",
        vector.shape
    )

    print(
        "vector =",
        vector
    )

    assert vector.shape == (16,)

    # ========================================================
    # 4. 字典编码
    # ========================================================

    dictionary = {
        "assembly_stage": [1, 0, 0, 0],

        "process_parameters": [
            10.0,
            0.05,
            10.0,
            5.0,
        ],

        "contact_requirements": [
            15.0,
            10.0,
            20.0,
            5.0,
        ],

        "environment": [
            0.5,
            0.5,
            1.0,
            0.5,
        ],
    }

    vector_dict = encoder.encode_dict(
        dictionary
    )

    assert vector_dict.shape == (16,)

    print(
        "字典编码检查通过。"
    )

    # ========================================================
    # 5. 批量编码
    # ========================================================

    batch = [
        factory.create(stage=1),
        factory.create(stage=2),
        factory.create(stage=3),
        factory.create(stage=4),
    ]

    matrix = encoder.encode_batch(
        batch
    )

    print("\n批量编码：")
    print(
        "shape =",
        matrix.shape
    )

    assert matrix.shape == (4, 16)

    print(
        "批量编码检查通过。"
    )

    # ========================================================
    # 6. 归一化测试
    # ========================================================

    normalized_encoder = (
        ProcessFactoryInfoEncoder(
            assembly_stage_dim=4,
            process_parameter_dim=4,
            contact_requirement_dim=4,
            environment_dim=4,
            normalize=True,
            normalization_mode="minmax",
        )
    )

    normalized = (
        normalized_encoder.fit_transform(
            matrix
        )
    )

    print("\nMin-Max 归一化后：")
    print(normalized)

    assert normalized.shape == (
        4,
        16,
    )

    assert np.all(
        normalized >= -1e-6
    )

    assert np.all(
        normalized <= 1.0 + 1e-6
    )

    print(
        "归一化检查通过。"
    )

    # ========================================================
    # 7. Tensor 测试
    # ========================================================

    tensor = normalized_encoder.encode_tensor(
        matrix
    )

    print("\nTensor：")
    print(
        "shape =",
        tuple(tensor.shape)
    )

    assert tensor.shape == (
        4,
        16,
    )

    assert tensor.dtype == torch.float32

    print(
        "Tensor 转换检查通过。"
    )

    # ========================================================
    # 8. PyTorch Module 测试
    # ========================================================

    module = ProcessFactoryInfoModule(
        feature_dim=16,
        normalize=False,
    )

    output = module(
        tensor
    )

    print("\nProcessFactoryInfoModule：")
    print(
        "shape =",
        tuple(output.shape)
    )

    assert output.shape == (
        4,
        16,
    )

    print(
        "PyTorch Module 检查通过。"
    )

    # ========================================================
    # 9. 数据库测试
    # ========================================================

    database = ProcessFactoryDatabase()

    for index, item in enumerate(batch):

        database.add(
            f"sample_{index:03d}",
            item,
        )

    print("\n数据库数量：")
    print(
        len(database)
    )

    assert len(database) == 4

    retrieved = database.get(
        "sample_000"
    )

    assert isinstance(
        retrieved,
        ProcessFactoryInfo,
    )

    print(
        "数据库检查通过。"
    )

    # ========================================================
    # 10. PFNM 输入检查
    # ========================================================

    pfnm_input = process_info_to_tensor(
        info
    )

    print("\nPFNM 输入：")
    print(
        "shape =",
        tuple(pfnm_input.shape)
    )

    assert pfnm_input.shape == (
        1,
        16,
    )

    print(
        "PFNM 输入接口检查通过。"
    )

    # ========================================================
    # 完成
    # ========================================================

    print("\n" + "=" * 70)
    print("process_factory_info.py Test Passed")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_process_factory_info()