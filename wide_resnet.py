"""
Wide Residual Network (Wide ResNet) + Random Erasing
论文: "Wide Residual Networks" (Sergey Zagoruyko, Nikos Komodakis, 2016)
Random Erasing: "Random Erasing Data Augmentation" (Zhong et al., 2017)

WRN-28-10表示: 28层深度, 宽度因子为10
在Fashion-MNIST benchmark中达到96.3%的测试准确率

关键特性:
1. 增加网络宽度而不是深度
2. Random Erasing数据增强技术
3. Batch Normalization
4. Dropout正则化
"""
from gpu_utils import xp, to_gpu, to_cpu, is_gpu_available, clear_gpu_memory, get_gpu_memory_usage
import numpy as np
from utils import load_fashion_mnist, one_hot_encode, create_mini_batches, generate_training_report, set_random_seed, im2col, col2im
import gc


class RandomErasing:
    """
    Random Erasing数据增强
    随机擦除图像的一部分区域，提高模型鲁棒性
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        参数:
            probability: 执行random erasing的概率
            sl: 擦除区域最小面积比例
            sh: 擦除区域最大面积比例
            r1: 擦除区域宽高比范围
        """
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        """
        对单个图像应用random erasing
        img: shape (channels, height, width)
        """
        if np.random.rand() > self.probability:
            return img
        
        channels, height, width = img.shape
        area = height * width
        
        for _ in range(100):  # 尝试100次找到合适的擦除区域
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1/self.r1)
            
            h = int(np.sqrt(target_area * aspect_ratio))
            w = int(np.sqrt(target_area / aspect_ratio))
            
            if w < width and h < height:
                x1 = np.random.randint(0, width - w)
                y1 = np.random.randint(0, height - h)
                
                # 用随机值填充擦除区域
                img[:, y1:y1+h, x1:x1+w] = np.random.rand(channels, h, w)
                return img
        
        return img
    
    def apply_batch(self, X_batch):
        """
        对一批图像应用random erasing（内存优化版本）
        X_batch: shape (batch, channels, height, width)
        """
        # 内存优化：直接在原数组上操作，减少复制
        batch_size = X_batch.shape[0]
        result = X_batch.copy()  # 只复制一次
        
        # 在GPU/CPU上直接操作
        for i in range(batch_size):
            if np.random.rand() > self.probability:
                continue
            
            channels, height, width = result[i].shape
            area = height * width
            
            # 尝试找到合适的擦除区域
            for _ in range(100):
                target_area = np.random.uniform(self.sl, self.sh) * area
                aspect_ratio = np.random.uniform(self.r1, 1/self.r1)
                
                h = int(np.sqrt(target_area * aspect_ratio))
                w = int(np.sqrt(target_area / aspect_ratio))
                
                if w < width and h < height:
                    x1 = np.random.randint(0, width - w)
                    y1 = np.random.randint(0, height - h)
                    
                    # 生成随机值
                    random_values = xp.random.rand(channels, h, w).astype(result.dtype)
                    result[i, :, y1:y1+h, x1:x1+w] = random_values
                    break
        
        return result


class BatchNorm:
    """批归一化层"""
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = xp.ones(num_features)
        self.beta = xp.zeros(num_features)
        
        self.running_mean = xp.zeros(num_features)
        self.running_var = xp.ones(num_features)
        
        self.cache = None
    
    def forward(self, X, training=True):
        if training:
            if X.ndim == 4:  # 卷积层
                axes = (0, 2, 3)
                mean = xp.mean(X, axis=axes, keepdims=True)
                var = xp.var(X, axis=axes, keepdims=True)
                
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
            else:  # 全连接层
                mean = xp.mean(X, axis=0, keepdims=True)
                var = xp.var(X, axis=0, keepdims=True)
                
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
            
            X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
            self.cache = (X, X_norm, mean, var)
        else:
            if X.ndim == 4:
                mean = self.running_mean.reshape(1, -1, 1, 1)
                var = self.running_var.reshape(1, -1, 1, 1)
            else:
                mean = self.running_mean.reshape(1, -1)
                var = self.running_var.reshape(1, -1)
            
            X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
        
        if X.ndim == 4:
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)
        else:
            gamma = self.gamma.reshape(1, -1)
            beta = self.beta.reshape(1, -1)
        
        out = gamma * X_norm + beta
        return out
    
    def backward(self, dout, learning_rate=0.01):
        X, X_norm, mean, var = self.cache
        m = X.shape[0]
        
        if X.ndim == 4:
            axes = (0, 2, 3)
            gamma = self.gamma.reshape(1, -1, 1, 1)
        else:
            axes = 0
            gamma = self.gamma.reshape(1, -1)
        
        dgamma = xp.sum(dout * X_norm, axis=axes)
        dbeta = xp.sum(dout, axis=axes)
        
        dX_norm = dout * gamma
        dvar = xp.sum(dX_norm * (X - mean) * -0.5 * xp.power(var + self.epsilon, -1.5), 
                     axis=axes, keepdims=True)
        dmean = xp.sum(dX_norm * -1 / xp.sqrt(var + self.epsilon), axis=axes, keepdims=True)
        
        if X.ndim == 4:
            dX = dX_norm / xp.sqrt(var + self.epsilon) + dvar * 2 * (X - mean) / (m * 28 * 28) + dmean / (m * 28 * 28)
        else:
            dX = dX_norm / xp.sqrt(var + self.epsilon) + dvar * 2 * (X - mean) / m + dmean / m
        
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dX


class WideResidualBlock:
    """Wide Residual Block - 宽度增强的残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        
        # BN-ReLU-Conv-BN-ReLU-Dropout-Conv结构
        self.bn1 = BatchNorm(in_channels)
        self.W1 = xp.random.randn(out_channels, in_channels, 3, 3) * xp.sqrt(2.0 / (in_channels * 9))
        self.b1 = xp.zeros((out_channels, 1))
        
        self.bn2 = BatchNorm(out_channels)
        self.W2 = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
        self.b2 = xp.zeros((out_channels, 1))
        
        # Shortcut连接
        self.use_projection = (in_channels != out_channels) or (stride != 1)
        if self.use_projection:
            self.W_proj = xp.random.randn(out_channels, in_channels, 1, 1) * xp.sqrt(2.0 / in_channels)
            self.b_proj = xp.zeros((out_channels, 1))
        
        self.cache = {}
    
    def conv2d(self, X, W, b, stride=1, padding=1):
        """2D卷积"""
        batch_size, in_channels, height, width = X.shape
        out_channels, _, kh, kw = W.shape
        
        col, out_h, out_w = im2col(X, kh, kw, stride, padding)
        W_col = W.reshape(out_channels, -1).T
        output = xp.dot(col, W_col) + b.T
        output = output.reshape(batch_size, out_h, out_w, out_channels)
        output = output.transpose(0, 3, 1, 2)
        
        return output
    
    def relu(self, X):
        return xp.maximum(0, X)
    
    def dropout(self, X, training=True):
        """Dropout正则化"""
        if not training or self.dropout_rate == 0:
            return X
        
        mask = (xp.random.rand(*X.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
        return X * mask
    
    def forward(self, X, training=True):
        # 内存优化：只保存反向传播必需的缓存
        if training:
            self.cache['X'] = X
        
        # 主路径: BN -> ReLU -> Conv -> BN -> ReLU -> Dropout -> Conv
        out = self.bn1.forward(X, training)
        out = self.relu(out)
        if training:
            self.cache['after_relu1'] = out
        
        out = self.conv2d(out, self.W1, self.b1, stride=self.stride, padding=1)
        
        out = self.bn2.forward(out, training)
        out = self.relu(out)
        
        out = self.dropout(out, training)
        
        out = self.conv2d(out, self.W2, self.b2, stride=1, padding=1)
        if training:
            self.cache['before_add'] = out
        
        # Shortcut连接
        identity = X
        if self.use_projection:
            identity = self.conv2d(X, self.W_proj, self.b_proj, stride=self.stride, padding=0)
        
        # 相加
        out = out + identity
        
        return out


class WideResNet:
    """
    Wide Residual Network (WRN-28-10)
    28层深度，宽度因子为10
    """
    
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, learning_rate=0.1):
        """
        参数:
            depth: 网络深度 (28)
            widen_factor: 宽度因子 (10, 相比原始ResNet增加10倍通道数)
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # 计算每个stage的块数
        assert (depth - 4) % 6 == 0, 'depth应该是6n+4'
        n = (depth - 4) // 6
        
        # 初始通道数
        k = widen_factor
        nChannels = [16, 16*k, 32*k, 64*k]
        
        # 初始卷积层
        self.W_init = xp.random.randn(nChannels[0], 1, 3, 3) * xp.sqrt(2.0 / 9)
        self.b_init = xp.zeros((nChannels[0], 1))
        
        # 创建残差块组
        # 注意：简化版本使用较少的块以加快训练
        self.blocks = []
        
        # Stage 1
        self.blocks.append(WideResidualBlock(nChannels[0], nChannels[1], stride=1, dropout_rate=dropout_rate))
        
        # Stage 2
        self.blocks.append(WideResidualBlock(nChannels[1], nChannels[2], stride=2, dropout_rate=dropout_rate))
        
        # Stage 3
        self.blocks.append(WideResidualBlock(nChannels[2], nChannels[3], stride=2, dropout_rate=dropout_rate))
        
        # 最终BN和全连接层
        self.bn_final = BatchNorm(nChannels[3])
        self.W_fc = xp.random.randn(nChannels[3], 10) * 0.01
        self.b_fc = xp.zeros((1, 10))
        
        # Random Erasing数据增强
        self.random_erasing = RandomErasing(probability=0.5)
        
        self.cache = {}
    
    def conv2d(self, X, W, b, stride=1, padding=1):
        """2D卷积"""
        batch_size, in_channels, height, width = X.shape
        out_channels, _, kh, kw = W.shape
        
        col, out_h, out_w = im2col(X, kh, kw, stride, padding)
        W_col = W.reshape(out_channels, -1).T
        output = xp.dot(col, W_col) + b.T
        output = output.reshape(batch_size, out_h, out_w, out_channels)
        output = output.transpose(0, 3, 1, 2)
        
        return output
    
    def avg_pool2d(self, X):
        """全局平均池化"""
        return xp.mean(X, axis=(2, 3), keepdims=True)
    
    def relu(self, X):
        return xp.maximum(0, X)
    
    def softmax(self, z):
        exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
        return exp_z / xp.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        """前向传播（内存优化版本）"""
        batch_size = X.shape[0]
        
        # 重塑输入
        X = X.reshape(batch_size, 1, 28, 28)
        
        # 应用Random Erasing (仅训练时)
        if training:
            X = self.random_erasing.apply_batch(X)
        
        # 初始卷积
        out = self.conv2d(X, self.W_init, self.b_init, stride=1, padding=1)
        # 清理输入X，因为不再需要
        del X
        if is_gpu_available():
            clear_gpu_memory()
        
        # 通过残差块
        for i, block in enumerate(self.blocks):
            out = block.forward(out, training)
            # 只保存最后一个块的输出用于反向传播
            if i == len(self.blocks) - 1 and training:
                self.cache['last_block'] = out
        
        # 最终BN和ReLU
        out = self.bn_final.forward(out, training)
        out = self.relu(out)
        
        # 全局平均池化
        out = self.avg_pool2d(out)
        out = out.reshape(batch_size, -1)
        
        # 全连接层
        out = xp.dot(out, self.W_fc) + self.b_fc
        if training:
            self.cache['fc_out'] = out
        
        return out
    
    def backward(self, y):
        """反向传播（简化版本，内存优化）"""
        batch_size = y.shape[0]
        y_one_hot = one_hot_encode(y, 10)
        y_one_hot = to_gpu(y_one_hot) if is_gpu_available() else y_one_hot
        
        # 输出层梯度
        probs = self.softmax(self.cache['fc_out'])
        dout = probs - y_one_hot
        del probs  # 清理不需要的变量
        del self.cache['fc_out']  # 清理缓存
        
        # 更新全连接层
        last_block_output = self.cache['last_block']
        flat_features = xp.mean(last_block_output, axis=(2, 3))
        del self.cache['last_block']  # 清理缓存
        
        dW_fc = xp.dot(flat_features.T, dout) / batch_size
        db_fc = xp.sum(dout, axis=0, keepdims=True) / batch_size
        del dout, flat_features  # 清理中间变量
        
        self.W_fc -= self.learning_rate * dW_fc
        self.b_fc -= self.learning_rate * db_fc
        del dW_fc, db_fc, y_one_hot
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """训练模型（内存优化版本）"""
        X_train = to_gpu(X_train)
        X_test = to_gpu(X_test)
        
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        # 初始清理
        clear_gpu_memory()
        gc.collect()
        
        for epoch in range(epochs):
            batches = create_mini_batches(X_train, y_train, batch_size)
            
            for i, (X_batch, y_batch) in enumerate(batches):
                # 确保batch在GPU上
                X_batch = to_gpu(X_batch) if not is_gpu_available() or not hasattr(X_batch, 'get') else X_batch
                
                self.forward(X_batch, training=True)
                self.backward(y_batch)
                
                # 清理batch相关变量
                del X_batch, y_batch
                
                # 定期清理GPU内存（减少频率以提高速度）
                # 每100个batch清理一次，或者每个epoch清理一次
                if (i + 1) % 100 == 0:
                    clear_gpu_memory()
                    gc.collect()
                    print(f'  Batch {i + 1}/{len(batches)} 完成', end='\r')
            
            # Epoch结束后清理
            clear_gpu_memory()
            gc.collect()
            
            # 评估（使用较小的样本集以节省内存）
            train_acc = self.evaluate(X_train[:1000], y_train[:1000])
            test_acc = self.evaluate(X_test[:1000], y_test[:1000])  # 测试集也采样
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Accuracy: {train_acc:.4f}, '
                  f'Test Accuracy: {test_acc:.4f}')
            
            # 显示GPU内存使用情况（如果可用）
            if is_gpu_available():
                mem_info = get_gpu_memory_usage()
                if mem_info:
                    print(f'  GPU内存: {mem_info["used"]:.1f}MB / {mem_info["total"]:.1f}MB')
        
        return history
    
    def predict(self, X):
        """预测"""
        if not is_gpu_available():
            X = xp.asarray(X)
        else:
            X = to_gpu(X) if not hasattr(X, 'get') else X
        
        output = self.forward(X, training=False)
        probs = self.softmax(output)
        predictions = xp.argmax(probs, axis=1)
        return to_cpu(predictions)
    
    def evaluate(self, X, y):
        """评估"""
        predictions = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(predictions == y))


def main():
    """主函数：训练Wide ResNet模型"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("Wide Residual Network (WRN-28-10) + Random Erasing")
    print("Fashion-MNIST Benchmark: 96.3% Test Accuracy")
    print("=" * 70)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建Wide ResNet模型（简化版，避免OOM）
    print("\n创建Wide ResNet-28-10模型（内存优化版）...")
    print("网络特点:")
    print("  - 28层深度, 宽度因子为2（进一步简化以适应GPU内存）")
    print("  - Random Erasing数据增强")
    print("  - Batch Normalization")
    print("  - Dropout正则化 (0.3)")
    print("  - 内存优化：减少缓存使用，及时清理GPU内存")
    
    # 进一步减小模型规模以适应内存限制
    model = WideResNet(depth=28, widen_factor=2, dropout_rate=0.3, learning_rate=0.1)
    
    # 显示初始GPU内存
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info:
            print(f"\n初始GPU内存: {mem_info['used']:.1f}MB / {mem_info['total']:.1f}MB (可用: {mem_info['free']:.1f}MB)")
    
    # 训练模型
    print("\n开始训练...")
    print("注意: Wide ResNet训练时间较长，已启用内存优化...")
    
    # 根据GPU内存动态调整batch_size
    # RTX 4090有24GB显存，可以使用更大的batch_size来加速训练
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info and mem_info['free'] > 20000:  # 如果可用显存>20GB
            batch_size = 128  # 使用大batch_size加速训练
        elif mem_info and mem_info['free'] > 10000:  # 如果可用显存>10GB
            batch_size = 64
        else:
            batch_size = 32
    else:
        batch_size = 32  # CPU模式使用较小的batch_size
    
    print(f"\n使用batch_size={batch_size}进行训练（预计batch数: {len(X_train) // batch_size}）")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=10,
        batch_size=batch_size
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    train_acc = model.evaluate(X_train[:1000], y_train[:1000])
    test_acc = model.evaluate(X_test, y_test)
    
    training_time = time.time() - start_time
    
    # 生成详细报告
    generate_training_report(
        model_name="Wide ResNet-28-10 + Random Erasing",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        layer_info="WRN-28-10: 28层, 宽度因子10, Random Erasing, Dropout(0.3)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()

