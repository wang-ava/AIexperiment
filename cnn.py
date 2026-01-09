"""
卷积神经网络 (Convolutional Neural Network, CNN)
手写实现卷积、池化、全连接层
支持 GPU 加速（通过 CuPy）
使用 im2col 方法优化卷积计算
"""
from gpu_utils import xp, to_gpu, to_cpu, is_gpu_available
import numpy as np  # 仍需要 numpy 用于某些操作
from utils import load_fashion_mnist, one_hot_encode, create_mini_batches, get_class_name, generate_training_report, set_random_seed, im2col, col2im


class Conv2D:
    """2D卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        初始化卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（卷积核数量）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        self.weights = xp.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * xp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = xp.zeros((out_channels, 1))
        
        self.cache = None
    
    def forward(self, X):
        """
        前向传播（使用im2col优化）
        
        参数:
            X: 输入, shape (batch_size, in_channels, height, width)
        
        返回:
            output: 卷积结果
        """
        batch_size, _, height, width = X.shape
        
        # 使用im2col将输入转换为列矩阵
        col, out_h, out_w = im2col(X, self.kernel_size, self.kernel_size, 
                                    self.stride, self.padding)
        
        # 将卷积核reshape为矩阵
        W_col = self.weights.reshape(self.out_channels, -1).T
        
        # 矩阵乘法执行卷积
        output = xp.dot(col, W_col) + self.bias.T
        
        # Reshape回原始形状
        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        output = output.transpose(0, 3, 1, 2)
        
        self.cache = (X, col)
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """
        反向传播（使用im2col优化）
        
        参数:
            dout: 上游梯度
            learning_rate: 学习率
        
        返回:
            dx: 对输入的梯度
        """
        X, col = self.cache
        batch_size = X.shape[0]
        
        # 计算偏置梯度
        db = xp.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)
        
        # Reshape梯度
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # 计算权重梯度
        dw = xp.dot(col.T, dout)
        dw = dw.T.reshape(self.out_channels, self.in_channels, 
                         self.kernel_size, self.kernel_size)
        
        # 计算输入梯度
        W_col = self.weights.reshape(self.out_channels, -1).T
        dcol = xp.dot(dout, W_col.T)
        dx = col2im(dcol, X.shape, self.kernel_size, self.kernel_size, 
                   self.stride, self.padding)
        
        # 更新参数
        self.weights -= learning_rate * dw / batch_size
        self.bias -= learning_rate * db / batch_size
        
        return dx


class MaxPool2D:
    """2D最大池化层"""
    
    def __init__(self, pool_size=2, stride=2):
        """
        初始化池化层
        
        参数:
            pool_size: 池化窗口大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, X):
        """
        前向传播（向量化优化）
        
        参数:
            X: 输入, shape (batch_size, channels, height, width)
        
        返回:
            output: 池化结果
        """
        batch_size, channels, height, width = X.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # 使用im2col展开
        col, _, _ = im2col(X, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        # 对每个池化窗口取最大值
        output = xp.max(col, axis=1)
        
        # 保存最大值的索引用于反向传播
        self.max_idx = xp.argmax(col, axis=1)
        
        # Reshape回原始形状
        output = output.reshape(batch_size, out_height, out_width, channels)
        output = output.transpose(0, 3, 1, 2)
        
        self.cache = X
        return output
    
    def backward(self, dout):
        """
        反向传播（向量化优化）
        
        参数:
            dout: 上游梯度
        
        返回:
            dx: 对输入的梯度
        """
        X = self.cache
        batch_size, channels = X.shape[:2]
        
        # Reshape梯度
        dout = dout.transpose(0, 2, 3, 1).flatten()
        
        # 创建梯度矩阵
        dcol = xp.zeros((dout.size, self.pool_size * self.pool_size))
        dcol[xp.arange(dout.size), self.max_idx] = dout
        
        # 使用col2im转换回原始形状
        dx = col2im(dcol, X.shape, self.pool_size, self.pool_size, self.stride, 0)
        
        return dx


class ReLU:
    """ReLU激活函数"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, X):
        self.cache = X
        return xp.maximum(0, X)
    
    def backward(self, dout):
        X = self.cache
        return dout * (X > 0)


class Flatten:
    """展平层"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)


class Dense:
    """全连接层"""
    
    def __init__(self, input_size, output_size):
        self.weights = xp.random.randn(input_size, output_size) * xp.sqrt(2.0 / input_size)
        self.bias = xp.zeros((1, output_size))
        self.cache = None
    
    def forward(self, X):
        self.cache = X
        return xp.dot(X, self.weights) + self.bias
    
    def backward(self, dout, learning_rate=0.01):
        X = self.cache
        batch_size = X.shape[0]
        
        dw = xp.dot(X.T, dout) / batch_size
        db = xp.sum(dout, axis=0, keepdims=True) / batch_size
        dx = xp.dot(dout, self.weights.T)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return dx


class CNN:
    """
    卷积神经网络
    结构: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Dense
    """
    
    def __init__(self, learning_rate=0.01):
        """
        初始化CNN
        
        参数:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        
        # 构建网络层
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        self.dense1 = Dense(32 * 7 * 7, 128)
        self.relu3 = ReLU()
        self.dense2 = Dense(128, 10)
    
    def forward(self, X):
        """前向传播"""
        # 重塑输入为 (batch_size, 1, 28, 28)
        X = X.reshape(-1, 1, 28, 28)
        
        # 第一个卷积块
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # 第二个卷积块
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        # 全连接层
        out = self.flatten.forward(out)
        out = self.dense1.forward(out)
        out = self.relu3.forward(out)
        out = self.dense2.forward(out)
        
        return out
    
    def softmax(self, z):
        """Softmax函数"""
        exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
        return exp_z / xp.sum(exp_z, axis=1, keepdims=True)
    
    def backward(self, X, y, output):
        """反向传播"""
        batch_size = X.shape[0]
        
        # 输出层梯度（交叉熵+softmax）
        y_one_hot = one_hot_encode(y, 10)
        y_one_hot = to_gpu(y_one_hot)
        dout = self.softmax(output) - y_one_hot
        
        # 反向传播全连接层
        dout = self.dense2.backward(dout, self.learning_rate)
        dout = self.relu3.backward(dout)
        dout = self.dense1.backward(dout, self.learning_rate)
        dout = self.flatten.backward(dout)
        
        # 反向传播卷积层
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout, self.learning_rate)
        
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout, self.learning_rate)
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        """训练模型"""
        # 将数据转移到 GPU（如果使用 GPU）
        X_train = to_gpu(X_train)
        X_test = to_gpu(X_test)
        
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        for epoch in range(epochs):
            batches = create_mini_batches(X_train, y_train, batch_size)
            
            for i, (X_batch, y_batch) in enumerate(batches):
                # 前向传播
                output = self.forward(X_batch)
                
                # 反向传播
                self.backward(X_batch, y_batch, output)
                
                if (i + 1) % 100 == 0:
                    print(f'  Batch {i + 1}/{len(batches)} 完成', end='\r')
            
            # 评估
            train_acc = self.evaluate(X_train[:1000], y_train[:1000])  # 采样评估
            test_acc = self.evaluate(X_test, y_test)
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Accuracy: {train_acc:.4f}, '
                  f'Test Accuracy: {test_acc:.4f}')
        
        return history
    
    def predict(self, X):
        """预测"""
        # 确保输入在 GPU 上（如果使用 GPU）
        if not is_gpu_available():
            X = xp.asarray(X)
        else:
            X = to_gpu(X) if not hasattr(X, 'get') else X
        
        output = self.forward(X)
        probs = self.softmax(output)
        predictions = xp.argmax(probs, axis=1)
        return to_cpu(predictions)
    
    def evaluate(self, X, y):
        """评估"""
        predictions = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(predictions == y))


def main():
    """主函数：训练CNN模型"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("卷积神经网络 (CNN) - Fashion-MNIST分类")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建CNN模型
    print("\n创建CNN模型...")
    print("网络结构: Conv(16) -> ReLU -> MaxPool -> Conv(32) -> ReLU -> MaxPool -> Dense(128) -> Dense(10)")
    model = CNN(learning_rate=0.01)
    
    # 训练模型（使用较少的epoch，因为手写CNN较慢）
    print("\n开始训练...")
    print("注意: 手写CNN训练较慢，请耐心等待...")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=5,
        batch_size=64
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    train_acc = model.evaluate(X_train[:1000], y_train[:1000])  # 采样评估训练集
    test_acc = model.evaluate(X_test, y_test)
    
    training_time = time.time() - start_time
    
    # 生成详细报告
    generate_training_report(
        model_name="卷积神经网络 (CNN)",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        layer_info="Conv(16) -> ReLU -> MaxPool -> Conv(32) -> ReLU -> MaxPool -> Dense(128) -> Dense(10)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()

