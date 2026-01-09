"""
LeNet-5 神经网络
经典的卷积神经网络架构，由Yann LeCun于1998年提出
原始用于手写数字识别，这里应用于Fashion-MNIST
支持 GPU 加速（通过 CuPy）
使用 im2col 方法优化卷积计算
"""
from gpu_utils import xp, to_gpu, to_cpu, is_gpu_available
import numpy as np  # 仍需要 numpy 用于某些操作
from utils import load_fashion_mnist, one_hot_encode, create_mini_batches, get_class_name, generate_training_report, set_random_seed, im2col, col2im


class LeNet5:
    """
    LeNet-5卷积神经网络
    结构: Conv1(6) -> Sigmoid -> AvgPool -> Conv2(16) -> Sigmoid -> AvgPool -> 
          Flatten -> FC1(120) -> Sigmoid -> FC2(84) -> Sigmoid -> FC3(10)
    """
    
    def __init__(self, learning_rate=0.01):
        """
        初始化LeNet-5
        
        参数:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        
        # 卷积层1: 6个5x5卷积核
        self.W1 = xp.random.randn(6, 1, 5, 5) * 0.1
        self.b1 = xp.zeros((6, 1))
        
        # 卷积层2: 16个5x5卷积核
        self.W2 = xp.random.randn(16, 6, 5, 5) * 0.1
        self.b2 = xp.zeros((16, 1))
        
        # 全连接层1: 16*4*4 -> 120
        self.W3 = xp.random.randn(16 * 4 * 4, 120) * 0.1
        self.b3 = xp.zeros((1, 120))
        
        # 全连接层2: 120 -> 84
        self.W4 = xp.random.randn(120, 84) * 0.1
        self.b4 = xp.zeros((1, 84))
        
        # 输出层: 84 -> 10
        self.W5 = xp.random.randn(84, 10) * 0.1
        self.b5 = xp.zeros((1, 10))
        
        # 用于反向传播的缓存
        self.cache = {}
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + xp.exp(-xp.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Sigmoid导数"""
        return a * (1 - a)
    
    def softmax(self, z):
        """Softmax函数"""
        exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
        return exp_z / xp.sum(exp_z, axis=1, keepdims=True)
    
    def conv2d(self, X, W, b, stride=1, padding=0):
        """
        2D卷积操作（使用im2col优化）
        
        参数:
            X: 输入, shape (batch, in_channels, height, width)
            W: 卷积核, shape (out_channels, in_channels, kh, kw)
            b: 偏置, shape (out_channels, 1)
            stride: 步长
            padding: 填充
        
        返回:
            输出特征图
        """
        batch_size, in_channels, height, width = X.shape
        out_channels, _, kh, kw = W.shape
        
        # 使用im2col将输入转换为列矩阵
        col, out_h, out_w = im2col(X, kh, kw, stride, padding)
        
        # 将卷积核reshape为矩阵
        W_col = W.reshape(out_channels, -1).T
        
        # 矩阵乘法执行卷积
        output = xp.dot(col, W_col) + b.T
        
        # Reshape回原始形状
        output = output.reshape(batch_size, out_h, out_w, out_channels)
        output = output.transpose(0, 3, 1, 2)
        
        return output
    
    def avg_pool2d(self, X, pool_size=2, stride=2):
        """
        2D平均池化（向量化优化）
        
        参数:
            X: 输入
            pool_size: 池化窗口大小
            stride: 步长
        
        返回:
            池化输出
        """
        batch_size, channels, height, width = X.shape
        
        # 使用im2col展开
        col, out_h, out_w = im2col(X, pool_size, pool_size, stride, 0)
        col = col.reshape(-1, pool_size * pool_size)
        
        # 对每个池化窗口取平均值
        output = xp.mean(col, axis=1)
        
        # Reshape回原始形状
        output = output.reshape(batch_size, out_h, out_w, channels)
        output = output.transpose(0, 3, 1, 2)
        
        return output
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据, shape (batch_size, 784)
        
        返回:
            输出, shape (batch_size, 10)
        """
        batch_size = X.shape[0]
        
        # 重塑输入为图像格式 (batch, 1, 28, 28)
        X = X.reshape(batch_size, 1, 28, 28)
        self.cache['X'] = X
        
        # 第一层卷积 + Sigmoid + 平均池化
        # 输入: (batch, 1, 28, 28) -> 输出: (batch, 6, 24, 24)
        C1 = self.conv2d(X, self.W1, self.b1, stride=1, padding=0)
        self.cache['C1'] = C1
        
        A1 = self.sigmoid(C1)
        self.cache['A1'] = A1
        
        # 平均池化: (batch, 6, 24, 24) -> (batch, 6, 12, 12)
        S1 = self.avg_pool2d(A1, pool_size=2, stride=2)
        self.cache['S1'] = S1
        
        # 第二层卷积 + Sigmoid + 平均池化
        # 输入: (batch, 6, 12, 12) -> 输出: (batch, 16, 8, 8)
        C2 = self.conv2d(S1, self.W2, self.b2, stride=1, padding=0)
        self.cache['C2'] = C2
        
        A2 = self.sigmoid(C2)
        self.cache['A2'] = A2
        
        # 平均池化: (batch, 16, 8, 8) -> (batch, 16, 4, 4)
        S2 = self.avg_pool2d(A2, pool_size=2, stride=2)
        self.cache['S2'] = S2
        
        # 展平: (batch, 16, 4, 4) -> (batch, 256)
        F = S2.reshape(batch_size, -1)
        self.cache['F'] = F
        
        # 全连接层1: (batch, 256) -> (batch, 120)
        Z3 = xp.dot(F, self.W3) + self.b3
        A3 = self.sigmoid(Z3)
        self.cache['A3'] = A3
        
        # 全连接层2: (batch, 120) -> (batch, 84)
        Z4 = xp.dot(A3, self.W4) + self.b4
        A4 = self.sigmoid(Z4)
        self.cache['A4'] = A4
        
        # 输出层: (batch, 84) -> (batch, 10)
        Z5 = xp.dot(A4, self.W5) + self.b5
        self.cache['Z5'] = Z5
        
        return Z5
    
    def backward(self, y):
        """
        反向传播
        
        参数:
            y: 真实标签
        """
        batch_size = y.shape[0]
        
        # 将标签转换为one-hot编码
        y_one_hot = one_hot_encode(y, 10)
        y_one_hot = to_gpu(y_one_hot) if is_gpu_available() else y_one_hot
        
        # 输出层梯度（交叉熵+softmax）
        A5 = self.softmax(self.cache['Z5'])
        dZ5 = A5 - y_one_hot
        
        # 全连接层3的梯度
        dW5 = xp.dot(self.cache['A4'].T, dZ5) / batch_size
        db5 = xp.sum(dZ5, axis=0, keepdims=True) / batch_size
        dA4 = xp.dot(dZ5, self.W5.T)
        
        # 全连接层2的梯度
        dZ4 = dA4 * self.sigmoid_derivative(self.cache['A4'])
        dW4 = xp.dot(self.cache['A3'].T, dZ4) / batch_size
        db4 = xp.sum(dZ4, axis=0, keepdims=True) / batch_size
        dA3 = xp.dot(dZ4, self.W4.T)
        
        # 全连接层1的梯度
        dZ3 = dA3 * self.sigmoid_derivative(self.cache['A3'])
        dW3 = xp.dot(self.cache['F'].T, dZ3) / batch_size
        db3 = xp.sum(dZ3, axis=0, keepdims=True) / batch_size
        dF = xp.dot(dZ3, self.W3.T)
        
        # 反向传播到卷积层（简化版本）
        dS2 = dF.reshape(self.cache['S2'].shape)
        dA2 = self._avg_pool2d_backward(dS2, self.cache['A2'])
        dC2 = dA2 * self.sigmoid_derivative(self.cache['A2'])
        dW2, db2 = self._conv2d_backward(dC2, self.cache['S1'], self.W2)
        
        dS1 = self._conv2d_input_backward(dC2, self.cache['S1'], self.W2)
        dA1 = self._avg_pool2d_backward(dS1, self.cache['A1'])
        dC1 = dA1 * self.sigmoid_derivative(self.cache['A1'])
        dW1, db1 = self._conv2d_backward(dC1, self.cache['X'], self.W1)
        
        # 更新参数
        self.W5 -= self.learning_rate * dW5
        self.b5 -= self.learning_rate * db5
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def _conv2d_backward(self, dout, X, W):
        """卷积层反向传播（计算权重梯度）- 向量化优化"""
        batch_size, out_channels, out_h, out_w = dout.shape
        _, in_channels, kh, kw = W.shape
        
        # 计算偏置梯度
        db = xp.sum(dout, axis=(0, 2, 3)).reshape(-1, 1) / batch_size
        
        # 使用im2col将输入转换为列矩阵
        col, _, _ = im2col(X, kh, kw, stride=1, padding=0)
        
        # Reshape梯度
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        
        # 计算权重梯度（矩阵乘法）
        dW = xp.dot(col.T, dout_reshaped)
        dW = dW.T.reshape(out_channels, in_channels, kh, kw) / batch_size
        
        return dW, db
    
    def _conv2d_input_backward(self, dout, X, W):
        """卷积层反向传播（计算输入梯度）- 向量化优化"""
        batch_size, out_channels, out_h, out_w = dout.shape
        _, in_channels, kh, kw = W.shape
        
        # Reshape梯度
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        
        # 将卷积核reshape为矩阵
        W_col = W.reshape(out_channels, -1).T
        
        # 计算列矩阵的梯度
        dcol = xp.dot(dout_reshaped, W_col.T)
        
        # 使用col2im转换回原始形状
        dX = col2im(dcol, X.shape, kh, kw, stride=1, padding=0)
        
        return dX
    
    def _avg_pool2d_backward(self, dout, X, pool_size=2, stride=2):
        """平均池化反向传播 - 向量化优化"""
        batch_size, channels, out_h, out_w = dout.shape
        
        # Reshape梯度并扩展
        dout_expanded = dout.transpose(0, 2, 3, 1).reshape(-1, 1)
        
        # 创建梯度矩阵（每个元素平均分配）
        dcol = xp.tile(dout_expanded, (1, pool_size * pool_size)) / (pool_size * pool_size)
        
        # 使用col2im转换回原始形状
        dX = col2im(dcol, X.shape, pool_size, pool_size, stride, 0)
        
        return dX
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """
        训练模型
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            epochs: 训练轮数
            batch_size: 批次大小
        
        返回:
            history: 训练历史记录
        """
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
                self.forward(X_batch)
                
                # 反向传播
                self.backward(y_batch)
                
                if (i + 1) % 50 == 0:
                    print(f'  Batch {i + 1}/{len(batches)} 完成', end='\r')
            
            # 评估
            train_acc = self.evaluate(X_train[:1000], y_train[:1000])
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
    """主函数：训练LeNet-5模型"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("LeNet-5 神经网络 - Fashion-MNIST分类")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建LeNet-5模型
    print("\n创建LeNet-5模型...")
    print("网络结构: Conv(6) -> Sigmoid -> AvgPool -> Conv(16) -> Sigmoid -> AvgPool")
    print("          -> FC(120) -> Sigmoid -> FC(84) -> Sigmoid -> FC(10)")
    model = LeNet5(learning_rate=0.05)
    
    # 训练模型
    print("\n开始训练...")
    print("注意: LeNet-5训练需要一定时间...")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=10,
        batch_size=128
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    train_acc = model.evaluate(X_train[:1000], y_train[:1000])  # 采样评估训练集
    test_acc = model.evaluate(X_test, y_test)
    
    training_time = time.time() - start_time
    
    # 生成详细报告
    generate_training_report(
        model_name="LeNet-5",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        layer_info="Conv(6) -> Sigmoid -> AvgPool -> Conv(16) -> Sigmoid -> AvgPool -> FC(120) -> FC(84) -> FC(10)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()

