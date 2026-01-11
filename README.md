Fashion-MNIST 神经网络实验集合（PyTorch）

本项目基于 Fashion-MNIST 数据集，使用 PyTorch 实现并对比多种经典/高性能神经网络结构，支持 CPU/GPU 自动切换、数据增强、以及训练完成自动生成文本报告，便于做实验记录与模型对比。

适合：想用一个小而完整的项目，系统性跑通“数据加载 → 训练 → 评估 → 报告 → 多模型横向对比”的同学。

主要特性

多模型实现与对比：MLP / CNN / LeNet-5 / ResNet / Wide ResNet / DenseNet / Capsule Network

一键运行 & 交互菜单：run.py 支持交互选择或自动跑全套模型

自动生成实验报告：每个模型训练后生成单独报告，并可生成汇总对比报告（保存在 reports/）

GPU 自动加速：自动检测 CUDA，可用则使用 GPU（见 gpu_utils.py）

数据增强支持：旋转、平移、随机裁剪、水平翻转、Random Erasing 等（见 data_augmentation.py）

可复现实验：统一随机种子（见 utils.set_random_seed）

模型列表

仓库内包含以下模型脚本（均可单独运行）：

mlp.py：多层感知机（MLP，全连接 + Dropout 等）

cnn.py：标准卷积网络（Conv + BN + Pool + Dropout 等）

lenet.py：LeNet-5 风格网络

resnet.py：残差网络（ResNet）

wide_resnet.py：Wide ResNet（宽残差网络，支持更强正则化/增强策略）

densenet.py：DenseNet-BC

capsule_network.py：Capsule Network（胶囊网络）

说明：run_new_models.py 专门用于跑 WideResNet / DenseNet / Capsule 这类高准确率模型组合，并在最后生成对比汇总报告。

项目结构
.
├── run.py                      # 入口：交互式菜单 / 自动运行全模型
├── run_new_models.py           # 入口：运行高准确率模型组合（WideResNet/DenseNet/CapsNet）
├── utils.py                    # 数据集加载、batch生成、类别名、训练报告/汇总报告工具
├── gpu_utils.py                # GPU 检测、设备管理、显存信息等
├── data_augmentation.py        # 数据增强策略 + label smoothing loss
├── mlp.py                      # MLP 模型
├── cnn.py                      # CNN 模型
├── lenet.py                    # LeNet-5 模型
├── resnet.py                   # ResNet 模型
├── wide_resnet.py              # Wide ResNet 模型
├── densenet.py                 # DenseNet-BC 模型
├── capsule_network.py          # Capsule Network 模型
├── test_data.py                # 数据加载测试（检查shape/范围/类别分布等）
├── generate_summary_report.py  # 汇总报告脚本（与 utils 内功能类似，可独立用）
├── requirements.txt            # 依赖
└── reports/                    # 训练输出报告（每次训练生成txt，带时间戳）

环境依赖与安装
1) 安装依赖
pip install -r requirements.txt


依赖主要是：

numpy

torch

torchvision

如果你只需要 CPU 版本 PyTorch，可按 requirements.txt 中注释提示安装 CPU-only 版本。

数据集准备（重要）

本项目通过 utils.load_fashion_mnist() 读取 Fashion-MNIST 的 gzip 原始文件（idx 格式 .gz），并会自动在以下路径尝试寻找数据集目录：

./dataset

../dataset

（以及与 utils.py 同目录/上级目录下的 dataset）

你需要准备一个 dataset/ 文件夹，并放入以下文件（标准命名）：

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

兼容性说明：

测试集文件名也兼容 test-images-idx3-ubyte.gz / test-labels-idx1-ubyte.gz（代码会自动 fallback）。

快速开始
方式 A：交互菜单运行（推荐）
python run.py


你可以在菜单中选择：

单独训练某个模型

跑“全部模型”（会逐个训练并生成汇总报告）

测试数据加载（确认 dataset 是否正确）

方式 B：自动跑全模型（无人值守）
python run.py --auto


自动模式默认会依次运行：
MLP → CNN → LeNet-5 → ResNet → WideResNet → DenseNet → Capsule Network

如需在自动模式中额外包含“数据加载测试”：

python run.py --auto --test

方式 C：只跑高准确率模型组合
python run_new_models.py


会依次运行：

Wide ResNet

DenseNet-BC

Capsule Network

并在最后生成一个总结报告用于横向对比。

方式 D：单独运行某个模型脚本

例如：

python mlp.py
python cnn.py
python resnet.py

输出：训练报告与汇总报告

训练结束后会在 reports/ 下生成文本报告（带时间戳），内容通常包括：

模型信息（结构描述、关键超参如学习率、训练轮数等）

数据集信息（样本数、输入维度、类别数等）

训练过程摘要（初始/最高/最终准确率、提升幅度等）

训练耗时（秒/分钟）

当你运行“全部模型”或 run_new_models.py 时，还会生成一个汇总报告，对比各模型：

训练准确率 / 测试准确率

耗时

最佳模型标记

生成了哪些报告文件

常见问题
1) 提示找不到数据集目录

请确认：

你已创建 dataset/ 文件夹（在项目根目录或上级目录）

dataset/ 下存在 4 个 .gz 文件（命名见上文）

你也可以先运行：

python test_data.py


它会输出数据 shape、取值范围、类别分布，帮助定位问题。

2) GPU 显存不足 / 训练太慢

可尝试：

关闭其他占用 GPU 的程序

调小 batch size（各模型脚本内一般有默认配置可改）

先在 CPU 上验证流程跑通，再切 GPU 跑完整实验

参考与致谢（可选）

Fashion-MNIST（Zalando Research）

LeNet-5 / ResNet / DenseNet / Wide ResNet / Capsule Network 相关论文与经典实现思路

许可证

本项目代码仅用于学习与实验对比用途。
