# Fashion-MNIST 神经网络实验集合（PyTorch）

本项目基于 **Fashion-MNIST** 数据集，使用 **PyTorch** 实现并对比多种神经网络结构，支持 **CPU/GPU 自动切换**、**数据增强**、以及**训练完成自动生成文本报告**，方便做实验记录与模型横向对比。

## 特性一览

- ✅ 多模型实现：MLP / CNN / LeNet-5 / ResNet / Wide ResNet / DenseNet / Capsule Network
- ✅ 一键运行：交互菜单 `run.py`，也支持命令行自动跑全套
- ✅ 自动生成训练报告：每个模型训练后在 `reports/` 生成详细报告
- ✅ 汇总对比报告：自动解析每个模型结果，输出对比总结
- ✅ 自动使用 GPU：检测 CUDA 可用则使用 GPU（见 `gpu_utils.py`）
- ✅ 数据增强：Random Erasing、常用增强策略（见 `data_augmentation.py`）

---

## 项目结构

> 目录树请务必放在代码块里，否则 GitHub 会压缩空格导致格式错乱。

```text
.
├── run.py                      # 入口：交互式菜单 / 自动运行所有模型
├── run_new_models.py           # 入口：一键运行高准确率模型组合（5-7）
├── utils.py                    # 数据加载、batch生成、类别名、报告工具等
├── gpu_utils.py                # GPU 检测与设备管理
├── data_augmentation.py        # 数据增强策略 + label smoothing 等
├── mlp.py                      # MLP 模型
├── cnn.py                      # CNN 模型
├── lenet.py                    # LeNet-5 模型
├── resnet.py                   # ResNet 模型
├── wide_resnet.py              # Wide ResNet 模型
├── densenet.py                 # DenseNet-BC 模型
├── capsule_network.py          # Capsule Network 模型
├── test_data.py                # 数据加载测试脚本（检查shape/范围/类别分布等）
├── generate_summary_report.py  # 汇总报告脚本（可独立运行）
├── requirements.txt            # 依赖
└── reports/                    # 训练输出报告（自动生成，初次可能不存在）

dataset/                        # 需要自行准备 Fashion-MNIST 原始 gzip 文件
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz   # 标准命名（推荐）
└── t10k-labels-idx1-ubyte.gz


