"""
Fashion-MNIST数据集加载工具
支持读取和解压.gz格式的Fashion-MNIST数据
数据默认加载到 CPU（NumPy），可在需要时传输到 GPU (PyTorch)
"""
import gzip
import numpy as np
import os
import torch


def set_random_seed(seed=42):
    """
    设置随机种子以保证实验可重复性（PyTorch版本）
    
    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子已设置: {seed} (NumPy + PyTorch)")


# 注意：PyTorch 使用内置的卷积操作，不再需要 im2col 和 col2im 函数
# 保留这些函数作为占位符以保持向后兼容，但它们不会被使用
def im2col(X, kernel_h, kernel_w, stride=1, padding=0):
    """
    已弃用：PyTorch使用内置卷积操作，不再需要im2col
    保留此函数仅为向后兼容
    """
    raise NotImplementedError("im2col is not needed in PyTorch. Use torch.nn.Conv2d instead.")


def col2im(col, X_shape, kernel_h, kernel_w, stride=1, padding=0):
    """
    已弃用：PyTorch使用内置卷积操作，不再需要col2im
    保留此函数仅为向后兼容
    """
    raise NotImplementedError("col2im is not needed in PyTorch. Use torch.nn.ConvTranspose2d instead.")


def load_mnist_images(filename):
    """
    加载Fashion-MNIST图像数据
    
    参数:
        filename: 图像文件路径 (.gz格式)
    
    返回:
        images: numpy数组, shape为(N, 784), N为图像数量
    """
    with gzip.open(filename, 'rb') as f:
        # 读取magic number和元数据
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # 读取图像数据
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        
    return data.astype(np.float32) / 255.0  # 归一化到[0, 1]


def load_mnist_labels(filename):
    """
    加载Fashion-MNIST标签数据
    
    参数:
        filename: 标签文件路径 (.gz格式)
    
    返回:
        labels: numpy数组, shape为(N,), N为标签数量
    """
    with gzip.open(filename, 'rb') as f:
        # 读取magic number和元数据
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # 读取标签数据
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        
    return labels


def load_fashion_mnist(data_dir=None):
    """
    加载完整的Fashion-MNIST数据集
    
    参数:
        data_dir: 数据集目录路径，如果为None则自动检测
    
    返回:
        (train_images, train_labels, test_images, test_labels)
    """
    # 自动检测数据集路径
    if data_dir is None:
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 尝试几个可能的路径
        possible_paths = [
            os.path.join(current_dir, 'dataset'),  # 同目录下的dataset
            os.path.join(current_dir, '..', 'dataset'),  # 上级目录下的dataset
            './dataset',  # 相对路径
            '../dataset'  # 上级相对路径
        ]
        
        for path in possible_paths:
            test_file = os.path.join(path, 'train-images-idx3-ubyte.gz')
            if os.path.exists(test_file):
                data_dir = path
                break
        
        if data_dir is None:
            raise FileNotFoundError(
                f"找不到数据集目录。请确保数据集文件存在于以下路径之一：\n" +
                "\n".join([f"  - {os.path.abspath(p)}" for p in possible_paths])
            )
    
    # Fashion-MNIST标准命名使用t10k前缀，不是test
    train_images = load_mnist_images(
        os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(
        os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    # 尝试两种命名方式：先尝试t10k（标准命名），再尝试test（兼容性）
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    if not os.path.exists(test_images_path):
        test_images_path = os.path.join(data_dir, 'test-images-idx3-ubyte.gz')
    test_images = load_mnist_images(test_images_path)
    
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    if not os.path.exists(test_labels_path):
        test_labels_path = os.path.join(data_dir, 'test-labels-idx1-ubyte.gz')
    test_labels = load_mnist_labels(test_labels_path)
    
    return train_images, train_labels, test_images, test_labels


def one_hot_encode(labels, num_classes=10):
    """
    将标签转换为one-hot编码
    
    参数:
        labels: 标签数组
        num_classes: 类别数量
    
    返回:
        one_hot: one-hot编码的标签, shape为(N, num_classes)
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def create_mini_batches(X, y, batch_size):
    """
    创建mini-batch（PyTorch版本）
    支持numpy数组或PyTorch张量
    
    参数:
        X: 输入数据（numpy数组或PyTorch张量）
        y: 标签（numpy数组）
        batch_size: batch大小
    
    返回:
        batches: [(X_batch, y_batch), ...]，返回numpy数组格式以保持兼容性
    """
    import numpy as np
    
    # 转换为numpy数组（如果输入是PyTorch张量）
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    m = X.shape[0]
    batches = []
    
    # 打乱数据 - 生成随机索引
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    
    # 创建完整的batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        X_batch = X_shuffled[k * batch_size:(k + 1) * batch_size]
        y_batch = y_shuffled[k * batch_size:(k + 1) * batch_size]
        batches.append((X_batch, y_batch))
    
    # 处理剩余数据
    if m % batch_size != 0:
        X_batch = X_shuffled[num_complete_batches * batch_size:]
        y_batch = y_shuffled[num_complete_batches * batch_size:]
        batches.append((X_batch, y_batch))
    
    return batches


# Fashion-MNIST类别名称
CLASS_NAMES = [
    'T-shirt/top',  # 0
    'Trouser',      # 1
    'Pullover',     # 2
    'Dress',        # 3
    'Coat',         # 4
    'Sandal',       # 5
    'Shirt',        # 6
    'Sneaker',      # 7
    'Bag',          # 8
    'Ankle boot'    # 9
]


def get_class_name(label):
    """获取类别名称"""
    return CLASS_NAMES[label]


def generate_training_report(model_name, history, train_acc, test_acc, X_train, y_train, 
                             X_test, y_test, model, layer_info, learning_rate, training_time=None):
    """
    生成训练报告并保存到文件
    
    参数:
        model_name: 模型名称
        history: 训练历史
        train_acc: 最终训练准确率
        test_acc: 最终测试准确率
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        model: 训练好的模型
        layer_info: 网络层信息
        learning_rate: 学习率
        training_time: 训练时间（秒）
    """
    import numpy as np
    from datetime import datetime
    
    # 创建报告内容
    report_lines = []
    
    def add_line(text):
        """添加一行到报告并打印"""
        print(text)
        report_lines.append(text)
    
    add_line("\n" + "=" * 70)
    add_line(" " * 20 + f"{model_name} 训练报告")
    add_line("=" * 70)
    
    # 添加生成时间
    add_line(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 模型配置信息
    add_line("\n【模型配置】")
    add_line("-" * 70)
    add_line(f"  网络结构: {layer_info}")
    add_line(f"  训练轮数: {history['epochs']}")
    add_line(f"  批次大小: {history['batch_size']}")
    add_line(f"  学习率: {learning_rate}")
    if training_time:
        add_line(f"  训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    
    # 2. 数据集信息
    add_line("\n【数据集信息】")
    add_line("-" * 70)
    add_line(f"  训练样本数: {X_train.shape[0]:,}")
    add_line(f"  测试样本数: {X_test.shape[0]:,}")
    add_line(f"  输入维度: {X_train.shape[1]}")
    add_line(f"  类别数量: 10")
    
    # 3. 训练过程
    add_line("\n【训练过程】")
    add_line("-" * 70)
    add_line(f"  初始训练准确率: {history['train_acc'][0]:.4f} ({history['train_acc'][0]*100:.2f}%)")
    add_line(f"  初始测试准确率: {history['test_acc'][0]:.4f} ({history['test_acc'][0]*100:.2f}%)")
    add_line(f"  最高训练准确率: {max(history['train_acc']):.4f} ({max(history['train_acc'])*100:.2f}%)")
    add_line(f"  最高测试准确率: {max(history['test_acc']):.4f} ({max(history['test_acc'])*100:.2f}%)")
    add_line(f"  准确率提升: {(history['test_acc'][-1] - history['test_acc'][0]):.4f} "
          f"({(history['test_acc'][-1] - history['test_acc'][0])*100:.2f}%)")
    
    # 添加每个epoch的详细数据
    add_line("\n  各轮次详细数据:")
    add_line("  " + "-" * 50)
    add_line(f"  {'Epoch':<8} {'训练准确率':<15} {'测试准确率':<15}")
    add_line("  " + "-" * 50)
    for i in range(len(history['train_acc'])):
        add_line(f"  {i+1:<8} {history['train_acc'][i]:.4f} ({history['train_acc'][i]*100:.2f}%)   "
                f"{history['test_acc'][i]:.4f} ({history['test_acc'][i]*100:.2f}%)")
    
    # 4. 最终性能
    add_line("\n【最终性能】")
    add_line("-" * 70)
    add_line(f"  训练准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    add_line(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    add_line(f"  过拟合程度: {(train_acc - test_acc):.4f} ({(train_acc - test_acc)*100:.2f}%)")
    
    # 5. 每个类别的性能
    add_line("\n【各类别性能】")
    add_line("-" * 70)
    predictions = model.predict(X_test)
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for i in range(len(y_test)):
        label = y_test[i]
        class_total[label] += 1
        if predictions[i] == label:
            class_correct[label] += 1
    
    add_line(f"  {'类别':<15} {'准确率':<10} {'正确/总数'}")
    add_line("  " + "-" * 40)
    for i in range(10):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        add_line(f"  {get_class_name(i):<15} {acc:.4f}     {int(class_correct[i])}/{int(class_total[i])}")
    
    # 6. 预测示例
    add_line("\n【预测示例】")
    add_line("-" * 70)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    correct_count = 0
    for idx in sample_indices:
        pred = model.predict(X_test[idx:idx+1])[0]
        true_label = y_test[idx]
        is_correct = pred == true_label
        if is_correct:
            correct_count += 1
        status = "✓" if is_correct else "✗"
        add_line(f"  样本 #{idx:5d}: 真实={get_class_name(true_label):15s} | "
              f"预测={get_class_name(pred):15s} {status}")
    add_line(f"\n  随机样本准确率: {correct_count}/10")
    
    # 7. 总结
    add_line("\n【总结】")
    add_line("-" * 70)
    if test_acc >= 0.90:
        performance = "优秀"
    elif test_acc >= 0.85:
        performance = "良好"
    elif test_acc >= 0.80:
        performance = "一般"
    else:
        performance = "需要改进"
    
    add_line(f"  模型性能评级: {performance}")
    add_line(f"  训练状态: {'可能存在过拟合' if (train_acc - test_acc) > 0.05 else '训练正常'}")
    
    suggestion_text = ""
    if (train_acc - test_acc) > 0.05:
        suggestion_text = "考虑添加正则化或dropout来减少过拟合"
    elif test_acc < 0.85:
        suggestion_text = "可以尝试增加网络深度、调整学习率或训练更多轮次"
    else:
        suggestion_text = "模型表现良好，可以投入使用"
    add_line(f"  建议: {suggestion_text}")
    
    add_line("\n" + "=" * 70)
    add_line(" " * 25 + "报告生成完成")
    add_line("=" * 70 + "\n")
    
    # 保存报告到文件
    # 创建reports目录（如果不存在）
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # 生成文件名：模型名称_时间戳.txt
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 清理模型名称，去除特殊字符
    clean_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    filename = os.path.join(reports_dir, f"{clean_model_name}_{timestamp}.txt")
    
    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ 报告已保存到: {filename}\n")
    
    return filename  # 返回报告文件路径


def parse_report_file(report_file):
    """
    从报告文件中解析训练结果
    
    参数:
        report_file: 报告文件路径
        
    返回:
        包含解析结果的字典，如果解析失败返回None
    """
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {}
        
        # 解析测试准确率（优先匹配【最终性能】部分的，如果没有则匹配其他部分）
        import re
        # 先尝试匹配【最终性能】部分的准确率（更准确）
        final_perf_match = re.search(r'【最终性能】.*?测试准确率:\s*([\d.]+)\s*\(([\d.]+)%\)', content, re.DOTALL)
        if final_perf_match:
            result['test_acc'] = float(final_perf_match.group(1))
        else:
            # 如果没有找到，尝试匹配其他部分的测试准确率
            test_acc_match = re.search(r'测试准确率:\s*([\d.]+)\s*\(([\d.]+)%\)', content)
            if test_acc_match:
                result['test_acc'] = float(test_acc_match.group(1))
        
        # 解析训练准确率（优先匹配【最终性能】部分的）
        final_train_match = re.search(r'【最终性能】.*?训练准确率:\s*([\d.]+)\s*\(([\d.]+)%\)', content, re.DOTALL)
        if final_train_match:
            result['train_acc'] = float(final_train_match.group(1))
        else:
            # 如果没有找到，尝试匹配其他部分的训练准确率
            train_acc_match = re.search(r'训练准确率:\s*([\d.]+)\s*\(([\d.]+)%\)', content)
            if train_acc_match:
                result['train_acc'] = float(train_acc_match.group(1))
        
        # 解析训练时间（匹配 "训练时间: X.XX 秒" 或 "训练时间: X.XX 秒 (X.XX 分钟)"）
        time_match = re.search(r'训练时间:\s*([\d.]+)\s*秒', content)
        if time_match:
            result['training_time'] = float(time_match.group(1))
        
        return result if result else None
    except Exception as e:
        print(f"解析报告文件 {report_file} 时出错: {e}")
        return None


def generate_summary_report(model_results, reports_dir="reports"):
    """
    生成所有模型的汇总报告
    
    参数:
        model_results: 字典列表，每个字典包含模型的结果信息
            [{
                'model_name': 模型名称,
                'train_acc': 训练准确率,
                'test_acc': 测试准确率,
                'training_time': 训练时间（秒）,
                'status': 'success' 或 'failed',
                'error': 错误信息（如果失败）
            }, ...]
        reports_dir: 报告目录
    """
    from datetime import datetime
    import glob
    
    # 创建报告内容
    report_lines = []
    
    def add_line(text):
        """添加一行到报告并打印"""
        print(text)
        report_lines.append(text)
    
    add_line("\n" + "=" * 70)
    add_line(" " * 20 + "所有模型训练汇总报告")
    add_line("=" * 70)
    
    # 添加生成时间
    add_line(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 统计信息
    total_models = len(model_results)
    success_models = [r for r in model_results if r.get('status') == 'success']
    failed_models = [r for r in model_results if r.get('status') == 'failed']
    
    add_line("\n【运行统计】")
    add_line("-" * 70)
    add_line(f"  总模型数: {total_models}")
    add_line(f"  成功: {len(success_models)} 个")
    add_line(f"  失败: {len(failed_models)} 个")
    
    # 在生成对比表之前，先尝试从报告文件中补充准确率信息
    if os.path.exists(reports_dir):
        report_files = sorted(glob.glob(os.path.join(reports_dir, "*.txt")), 
                             key=os.path.getmtime, reverse=True)
        
        # 建立模型名称映射（处理中文模型名称和英文文件名的匹配）
        model_name_mapping = {
            '多层感知机 (MLP)': ['多层感知机_MLP', 'MLP', '多层感知机'],
            '卷积神经网络 (CNN)': ['卷积神经网络_CNN', 'CNN', '卷积神经网络'],
            '残差网络 (ResNet)': ['残差神经网络_ResNet', 'ResNet', '残差网络', '残差神经网络'],
            'LeNet-5': ['LeNet', 'LeNet-5'],
            'Wide ResNet-28-10 + Random Erasing': ['Wide_ResNet', 'Wide ResNet'],
            'DenseNet-BC': ['DenseNet', 'DenseNet-BC'],
            'Capsule Network': ['Capsule_Network', 'Capsule', 'CapsNet']
        }
        
        # 为每个成功模型查找对应的报告文件并解析
        model_file_map = {}
        for file in report_files:
            basename = os.path.basename(file)
            # 跳过汇总报告和参数文件
            if '汇总报告' in basename or '参数' in basename:
                continue
            # 从文件名提取模型名称（去掉时间戳）
            parts = basename.rsplit('_', 2)
            if len(parts) >= 3:
                model_key = '_'.join(parts[:-2])
                if model_key not in model_file_map:
                    model_file_map[model_key] = file
        
        # 更新模型结果（优先使用报告文件中的准确率）
        for result in model_results:
            if result.get('status') == 'success':
                model_name = result.get('model_name', '')
                
                # 查找匹配的报告文件
                matched_file = None
                # 首先尝试直接匹配文件名中的关键部分（更精确的匹配）
                for key, file_path in model_file_map.items():
                    # 提取模型名称的关键词进行匹配（使用更精确的匹配逻辑）
                    matched = False
                    
                    if 'MLP' in model_name or '多层感知机' in model_name:
                        # MLP: 只匹配包含"MLP"或"多层感知机"但不包含其他复杂模型的名称
                        matched = ('MLP' in key or '多层感知机' in key) and 'Wide' not in key and 'Dense' not in key
                    elif 'CNN' in model_name or '卷积神经网络' in model_name:
                        # CNN: 只匹配包含"CNN"或"卷积神经网络"的文件
                        matched = ('CNN' in key or '卷积神经网络' in key) and 'Wide' not in key
                    elif '残差网络' in model_name or ('ResNet' in model_name and 'Wide' not in model_name):
                        # ResNet: 只匹配"残差神经网络"或只包含"ResNet"（不包含"Wide"）的文件
                        matched = ('残差' in key and 'Wide' not in key) or (key == 'ResNet' or (key.endswith('ResNet') and not key.startswith('Wide')))
                    elif 'LeNet' in model_name:
                        # LeNet: 只匹配LeNet相关的文件
                        matched = 'LeNet' in key
                    elif 'Wide' in model_name or 'WRN' in model_name:
                        # Wide ResNet: 只匹配Wide ResNet相关的文件
                        matched = 'Wide' in key or 'WRN' in key
                    elif 'DenseNet' in model_name or ('Dense' in model_name and 'DenseNet' in model_name):
                        # DenseNet: 只匹配DenseNet相关的文件
                        matched = 'DenseNet' in key or ('Dense' in key and 'Wide' not in key)
                    elif 'Capsule' in model_name or 'CapsNet' in model_name:
                        # Capsule Network: 只匹配Capsule相关的文件
                        matched = 'Capsule' in key or 'CapsNet' in key
                    
                    if matched:
                        matched_file = file_path
                        break
                
                # 如果没有找到匹配的文件，尝试使用映射表
                if not matched_file:
                    for full_name, keywords in model_name_mapping.items():
                        if full_name == model_name:
                            for kw in keywords:
                                for key, file_path in model_file_map.items():
                                    if kw in key:
                                        matched_file = file_path
                                        break
                                if matched_file:
                                    break
                        if matched_file:
                            break
                
                # 解析匹配的报告文件并更新结果
                if matched_file:
                    parsed = parse_report_file(matched_file)
                    if parsed:
                        # 优先使用报告文件中的数据（更准确）
                        if 'test_acc' in parsed:
                            result['test_acc'] = parsed['test_acc']
                        if 'train_acc' in parsed:
                            result['train_acc'] = parsed['train_acc']
                        if 'training_time' in parsed and parsed['training_time'] > 0:
                            result['training_time'] = parsed['training_time']
                        print(f"  已从报告文件更新 {model_name} 的准确率: 训练={result.get('train_acc', 0):.4f}, 测试={result.get('test_acc', 0):.4f}")
    
    # 成功模型对比
    if success_models:
        add_line("\n【模型性能对比】")
        add_line("-" * 70)
        add_line(f"  {'模型名称':<25} {'训练准确率':<15} {'测试准确率':<15} {'训练时间':<15}")
        add_line("  " + "-" * 68)
        
        # 按测试准确率排序
        sorted_models = sorted(success_models, key=lambda x: x.get('test_acc', 0), reverse=True)
        
        for result in sorted_models:
            model_name = result.get('model_name', '未知')
            train_acc = result.get('train_acc', 0)
            test_acc = result.get('test_acc', 0)
            training_time = result.get('training_time', 0)
            
            train_acc_str = f"{train_acc:.4f} ({train_acc*100:.2f}%)"
            test_acc_str = f"{test_acc:.4f} ({test_acc*100:.2f}%)"
            
            if training_time > 0:
                if training_time < 60:
                    time_str = f"{training_time:.1f}秒"
                else:
                    time_str = f"{training_time/60:.1f}分钟"
            else:
                time_str = "未知"
            
            add_line(f"  {model_name:<25} {train_acc_str:<15} {test_acc_str:<15} {time_str:<15}")
        
        # 最佳模型
        if sorted_models:
            best = sorted_models[0]
            add_line(f"\n  🏆 最佳模型: {best.get('model_name')} (测试准确率: {best.get('test_acc', 0):.4f})")
    
    # 失败模型
    if failed_models:
        add_line("\n【失败模型】")
        add_line("-" * 70)
        for result in failed_models:
            model_name = result.get('model_name', '未知')
            error = result.get('error', '未知错误')
            add_line(f"  ✗ {model_name}: {error}")
    
    # 报告文件列表
    add_line("\n【生成的报告文件】")
    add_line("-" * 70)
    
    # 查找所有报告文件
    if os.path.exists(reports_dir):
        report_files = sorted(glob.glob(os.path.join(reports_dir, "*.txt")), 
                             key=os.path.getmtime, reverse=True)
        
        # 只显示最近的报告（每个模型一个）
        model_files = {}
        for file in report_files:
            # 从文件名提取模型名称（去掉时间戳）
            basename = os.path.basename(file)
            # 跳过汇总报告（会在最后单独显示）
            if '汇总报告' in basename:
                continue
            # 找到最后一个下划线的位置（时间戳前）
            parts = basename.rsplit('_', 2)
            if len(parts) >= 3:
                model_key = '_'.join(parts[:-2])  # 去掉时间戳部分
                if model_key not in model_files:
                    model_files[model_key] = file
        
        if model_files:
            for model_key, file_path in sorted(model_files.items()):
                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024
                add_line(f"  ✓ {os.path.basename(file_path)} ({file_size_kb:.1f} KB)")
        else:
            add_line("  未找到报告文件")
    else:
        add_line("  报告目录不存在")
    
    # 保存汇总报告到文件
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = os.path.join(reports_dir, f"汇总报告_{timestamp}.txt")
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 在报告中添加汇总报告文件信息
    add_line(f"\n  📊 汇总报告: {os.path.basename(summary_filename)}")
    
    add_line("\n" + "=" * 70)
    add_line(" " * 25 + "汇总报告生成完成")
    add_line("=" * 70 + "\n")
    
    print(f"\n✓ 汇总报告已保存到: {summary_filename}\n")
    
    return summary_filename

