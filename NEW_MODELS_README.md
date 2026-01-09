# Fashion-MNIST 高准确率模型

本项目新增了3个来自Fashion-MNIST官方Benchmark的高准确率深度学习模型。

## 📊 模型概览

根据 [Fashion-MNIST GitHub Benchmark](https://github.com/zalandoresearch/fashion-mnist)，我们实现了以下3个表现最优的模型：

| 模型 | Benchmark准确率 | 主要特点 |
|------|----------------|----------|
| **Wide ResNet-28-10 + Random Erasing** | 96.3% | 增加网络宽度，Random Erasing数据增强 |
| **DenseNet-BC** | 95.4% | Dense连接，特征重用，参数高效 |
| **Capsule Network** | 93.6% | 向量神经元，动态路由，保留空间关系 |

## 🚀 快速开始

### 方式1: 运行单个模型

```bash
# Wide ResNet
python wide_resnet.py

# DenseNet
python densenet.py

# Capsule Network
python capsule_network.py
```

### 方式2: 使用运行脚本

```bash
# 交互式选择模型
python run.py

# 选项 5: Wide ResNet
# 选项 6: DenseNet
# 选项 7: Capsule Network
# 选项 9: 运行所有新模型
```

### 方式3: 一键运行所有新模型

```bash
python run_new_models.py
```

这个脚本会：
1. 依次训练所有3个新模型
2. 为每个模型生成单独的训练报告
3. 最后生成一个综合对比报告

## 📁 模型详细说明

### 1. Wide ResNet-28-10 + Random Erasing

**论文**: "Wide Residual Networks" (Zagoruyko & Komodakis, 2016)

**核心思想**:
- 增加网络宽度（通道数）而非深度
- 使用Random Erasing数据增强技术
- 宽度因子为10，相比标准ResNet增加10倍通道数

**技术特点**:
- 28层深度，宽度因子10
- Batch Normalization
- Dropout正则化 (0.3)
- Random Erasing (随机擦除图像区域)

**实现文件**: `wide_resnet.py`

### 2. DenseNet-BC

**论文**: "Densely Connected Convolutional Networks" (Huang et al., 2017)

**核心思想**:
- Dense Connection: 每层与前面所有层直接相连
- Bottleneck结构 (BC): 使用1x1卷积降低计算量
- Compression: 在Transition层压缩特征数量

**技术特点**:
- Growth rate k=12
- 3个Dense Block
- Bottleneck层（4k特征图）
- Compression率 0.5
- 参数效率高，减少梯度消失

**实现文件**: `densenet.py`

### 3. Capsule Network

**论文**: "Dynamic Routing Between Capsules" (Sabour, Hinton et al., 2017)

**核心思想**:
- 使用向量（Capsule）替代标量神经元
- 向量长度表示实体存在概率
- 动态路由算法进行信息传递
- 更好地保留空间层次关系

**技术特点**:
- Primary Capsule: 32个8维capsule
- Digit Capsule: 10个16维capsule (对应10个类别)
- Dynamic Routing 算法 (3次迭代)
- Squash激活函数
- Margin Loss

**实现文件**: `capsule_network.py`

## 📈 训练配置

所有模型的默认配置:

```python
epochs = 10          # 训练轮数
batch_size = 128     # 批次大小
learning_rate = 0.1  # 学习率（Wide ResNet和DenseNet）
learning_rate = 0.001 # 学习率（Capsule Network）
```

## 📊 报告生成

每个模型训练完成后会在 `reports/` 目录下生成详细报告，包括：

- ✅ 模型配置信息
- ✅ 训练和测试准确率
- ✅ 每个epoch的性能变化
- ✅ 各类别识别准确率
- ✅ 混淆矩阵分析
- ✅ 训练时间统计

### 生成总结报告

运行所有模型后，可以生成综合对比报告：

```bash
python generate_summary_report.py
```

总结报告包含：
- 📊 所有模型性能对比表
- 🏆 最佳模型分析
- ⚡ 各类别最优模型（速度、准确率、性价比）
- 📈 与官方Benchmark的对比
- 📉 统计分析
- 💡 架构特点分析
- 🔬 优化建议

## 🎯 与官方Benchmark对比

| 模型 | 官方Benchmark | 本实现目标 | 说明 |
|------|--------------|-----------|------|
| Wide ResNet-28-10 + RE | 96.3% | ~90%+ | 简化版，减少层数以加快训练 |
| DenseNet-BC | 95.4% | ~90%+ | 简化版，Growth rate=12 |
| Capsule Network | 93.6% | ~90%+ | 8M参数版本 |

**注意**: 由于实现为从零开始的NumPy/CuPy版本，且为了加快训练速度进行了简化，实际准确率可能略低于官方benchmark。但架构和核心思想完全一致。

## ⚙️ GPU加速支持

所有模型都支持GPU加速（通过CuPy）:

```bash
# 检查GPU是否可用
python -c "from gpu_utils import is_gpu_available; print('GPU可用' if is_gpu_available() else 'GPU不可用')"
```

如果GPU可用，模型会自动使用GPU加速训练。

## 📚 参考文献

1. **Fashion-MNIST数据集**:
   - Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747

2. **Wide ResNet**:
   - Zagoruyko, S., & Komodakis, N. (2016). Wide Residual Networks. arXiv:1605.07146

3. **Random Erasing**:
   - Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random Erasing Data Augmentation. arXiv:1708.04896

4. **DenseNet**:
   - Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR 2017.

5. **Capsule Network**:
   - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules. NIPS 2017.

## 🔧 故障排除

### 内存不足 (Out of Memory)

**⚠️ 重要：代码已进行内存优化！**

如果遇到内存不足错误，已实施以下优化措施：

1. **GPU内存自动清理**：训练过程中定期清理GPU内存
2. **优化的im2col函数**：减少卷积操作的内存占用
3. **最小化缓存使用**：只保存反向传播必需的中间结果
4. **优化的Random Erasing**：减少内存复制
5. **模型规模调整**：默认使用较小的模型参数以适应内存限制

**进一步优化建议**：

如果仍然遇到OOM错误，可以：
1. **减小batch_size**：从2改为1（在模型文件的main函数中修改）
   ```python
   batch_size = 1  # 最小batch size
   ```
2. **进一步减小模型规模**：
   ```python
   # Wide ResNet: widen_factor 从 2 改为 1
   # DenseNet: growth_rate 从 4 改为 2
   ```
3. **减少训练轮数**：从10改为5
4. **使用CPU模式**：如果GPU内存非常有限
5. **查看详细优化指南**：参考 `MEMORY_OPTIMIZATION.md`

**查看GPU内存使用**：
训练过程中会显示GPU内存使用情况，如果接近上限（>80%），建议减小batch_size或模型规模。

### 训练速度慢

Wide ResNet和DenseNet训练时间较长（可能需要30-60分钟或更长）。建议：
1. 使用GPU加速
2. 减少训练轮数（如改为5轮）
3. 先在小数据集上测试

### 准确率不理想

1. 增加训练轮数（epochs）
2. 调整学习率
3. 使用更多的数据增强
4. 调整模型结构参数

## 💻 系统要求

- Python 3.6+
- NumPy
- CuPy (可选，用于GPU加速)
- 内存: 至少4GB RAM
- GPU内存: 推荐至少4GB（代码已优化，可在更小的GPU上运行）
- 存储: 至少500MB可用空间

**注意**：代码已进行内存优化，可以在较小的GPU内存限制下运行。如果您的GPU内存非常有限，请参考 `MEMORY_OPTIMIZATION.md` 获取详细的内存优化建议。

## 📞 联系方式

如有问题或建议，请查看代码注释或提交issue。

---

**Happy Training! 🚀**

