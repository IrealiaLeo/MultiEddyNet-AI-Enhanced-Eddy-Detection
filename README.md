# MultiEddyNet-AI-Enhanced-Eddy-Detection
# 多尺度海洋涡旋的智能识别与分析

本项目是学位论文《多尺度涡旋的智能识别与分析》（龚博儒，上海交通大学，2025）的代码实现。项目旨在提供一个**结合深度学习与优化追踪算法的端到端海洋涡旋分析流程**。

该工作流的核心是专为海洋涡旋识别任务设计的深度学习模型 **MultiEddyNet**，以及一个基于匈牙利算法的鲁棒多目标追踪（Multi-Object Tracking, MOT）系统（暂未加入仓库）。整套系统能够从高分辨率海洋模式数据中自动识别中尺度和亚中尺度涡旋，精确追踪其完整生命周期，并为后续的科学分析提供高质量的轨迹数据集。

## 项目核心功能

-   **智能涡旋识别 (`MultiEddyNet`)**:
    -   采用一个改进的 **U-Net** 架构，能够同时接收多个物理场（如海面高度、流速、Okubo-Weiss参数等）作为输入，充分利用多源信息的互补性。
    -   在网络瓶颈层集成了**空洞空间金字塔池化 (ASPP)** 模块，显著增强了模型对不同尺度涡旋的感知能力，使其能同时精准识别大尺度和小尺度涡旋。

-   **精确涡旋追踪 (MOT 系统)**:
    -   遵循“检测后跟踪”（Tracking-by-Detection）范式，将涡旋识别与追踪解耦。
    -   使用**匈牙利算法**进行数据关联，实现跨时间帧的全局最优匹配，有效处理涡旋密集和交互等复杂场景。
    -   集成了**候选池管理机制**，能够容忍短暂的检测丢失，显著提升了生成轨迹的连续性和完整性。

-   **系统的统计分析**:
    -   基于生成的涡旋轨迹数据集，项目演示了如何对涡旋的运动学特征（如数量、生命周期、半径、传播距离）、空间分布和季节性变化进行深入的统计分析。

## 文件结构

本项目代码遵循模块化设计，主要文件功能如下：

-   `hybrid_detection/`: 包含复现的传统混合涡旋检测算法，用于生成训练标签。
    -   `main.py`: 传统算法主程序。
    -   `eddy.py`, `ow.py`, `extrema.py`, `dedup.py`, `contour.py`: 传统算法的各个模块。
-   `intelligent_identification/`: 包含 `MultiEddyNet` 的实现、训练和评估。
    -   `config.py`: **全局配置文件**，用于设置路径、超参数等。
    -   `dataset.py`: 数据加载器，负责从HDF5文件读取数据。
    -   `model.py`: **`MultiEddyNet`** 模型架构的定义。
    -   `train.py`: **主训练脚本**，启动模型训练流程。
    -   `evaluate.py`: **评估脚本**，用于可视化模型性能和生成对比图。
    -   `trainer.py` & `utils.py`: 训练器和通用工具函数。
-   `tracking/`: （可选）多目标追踪算法的实现。
-   `analysis/`: （可选）用于统计分析的脚本。

## 如何使用

### 1. 环境配置

建议使用 `conda` 或 `virtualenv` 创建一个独立的Python环境，并安装所有必要的依赖项。

```bash
# 克隆仓库
git clone https://github.com/your-username/multiscale-eddy-analysis.git
cd multiscale-eddy-analysis

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

-   **原始数据**: 将高分辨率的海洋模式NetCDF数据（如MOM6输出）放置在指定目录。
-   **标签生成**: 运行 `train/main.py` 来生成用于训练的涡旋标签，并将其与原始物理场一同打包成HDF5文件。请确保HDF5文件结构与 `dataset.py` 中的要求一致。
-   **路径配置**: 更新 `train/config.py` 文件中的 `HDF5_FILE` 路径，使其指向您生成的HDF5数据文件。

### 3. 模型训练

配置好 `config.py` 中的训练参数（如年份、批大小、学习率等）后，运行主训练脚本：

```bash
python train/train.py
```
训练日志、TensorBoard记录和最佳模型权重将保存在 `runs/` 和 `checkpoints/` 目录下。

### 4. 模型评估与可视化

训练完成后，可使用评估脚本对模型性能进行可视化分析。在 `evaluate.py` 中设置您想要评估的日期 (`TARGET_DATE_STR`)，然后运行：

```bash
python eval/evaluate.py
```
脚本将生成对比图，展示模型识别结果与真实标签之间的差异，保存在 `evaluation_plots/` 目录中。

## 引用

若本研究对您的工作有所帮助，请考虑引用：

> 龚博儒. (2025). *多尺度涡旋的智能识别与分析* [学士学位论文]. 上海交通大学.
