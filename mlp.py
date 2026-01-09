"""
多层感知机 (Multi-Layer Perceptron, MLP)
全连接神经网络实现，使用反向传播算法训练
支持 GPU 加速（通过 CuPy）
"""
from gpu_utils import xp, to_gpu, to_cpu, is_gpu_available
import numpy as np  # 仍需要 numpy 用于某些操作（如文件 I/O）
from utils import load_fashion_mnist, one_hot_encode, create_mini_batches, get_class_name, generate_training_report, set_random_seed


class MLP:
    """
    多层感知机神经网络
    支持任意层数和神经元数量的全连接网络
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu'):
        """
        初始化MLP
        
        参数:
            layer_sizes: 列表，每层的神经元数量，如[784, 128, 64, 10]
            learning_rate: 学习率
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.num_layers = len(layer_sizes)
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # He初始化（适用于ReLU）
        for i in range(self.num_layers - 1):
            w = xp.random.randn(layer_sizes[i], layer_sizes[i + 1]) * xp.sqrt(2.0 / layer_sizes[i])
            b = xp.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def activation(self, z):
        """激活函数"""
        if self.activation_type == 'relu':
            return xp.maximum(0, z)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + xp.exp(-xp.clip(z, -500, 500)))
        elif self.activation_type == 'tanh':
            return xp.tanh(z)
        else:
            return z
    
    def activation_derivative(self, z):
        """激活函数的导数"""
        if self.activation_type == 'relu':
            return (z > 0).astype(xp.float32)
        elif self.activation_type == 'sigmoid':
            s = self.activation(z)
            return s * (1 - s)
        elif self.activation_type == 'tanh':
            return 1 - xp.tanh(z) ** 2
        else:
            return xp.ones_like(z)
    
    def softmax(self, z):
        """Softmax函数（用于输出层）"""
        exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
        return exp_z / xp.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据, shape (batch_size, input_dim)
        
        返回:
            activations: 每层的激活值列表
            z_values: 每层的加权输入列表
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = xp.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # 最后一层使用softmax，其他层使用指定的激活函数
            if i == self.num_layers - 2:
                a = self.softmax(z)
            else:
                a = self.activation(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def backward(self, X, y, activations, z_values):
        """
        反向传播
        
        参数:
            X: 输入数据
            y: 标签（one-hot编码）
            activations: 前向传播的激活值
            z_values: 前向传播的加权输入
        
        返回:
            weight_gradients: 权重梯度列表
            bias_gradients: 偏置梯度列表
        """
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # 输出层的误差（交叉熵+softmax的导数）
        delta = activations[-1] - y
        
        # 反向传播
        for i in range(self.num_layers - 2, -1, -1):
            # 计算梯度
            dW = xp.dot(activations[i].T, delta) / m
            db = xp.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # 如果不是第一层，继续传播误差
            if i > 0:
                delta = xp.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """更新权重和偏置"""
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
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
        
        # 转换标签为one-hot编码（在 CPU 上，因为标签通常较小）
        y_train_encoded = one_hot_encode(y_train, self.layer_sizes[-1])
        y_train_encoded = to_gpu(y_train_encoded)
        
        # 记录训练历史
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        for epoch in range(epochs):
            # 创建mini-batches
            batches = create_mini_batches(X_train, y_train_encoded, batch_size)
            
            # 训练每个batch
            for X_batch, y_batch in batches:
                # 前向传播
                activations, z_values = self.forward(X_batch)
                
                # 反向传播
                weight_gradients, bias_gradients = self.backward(
                    X_batch, y_batch, activations, z_values)
                
                # 更新参数
                self.update_parameters(weight_gradients, bias_gradients)
            
            # 计算训练和测试准确率
            train_acc = self.evaluate(X_train, y_train)
            test_acc = self.evaluate(X_test, y_test)
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Accuracy: {train_acc:.4f}, '
                  f'Test Accuracy: {test_acc:.4f}')
        
        return history
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            predictions: 预测的类别（numpy 数组，在 CPU 上）
        """
        # 确保输入在 GPU 上（如果使用 GPU）
        if not is_gpu_available():
            X = xp.asarray(X)
        else:
            X = to_gpu(X) if not hasattr(X, 'get') else X
        
        activations, _ = self.forward(X)
        # 将结果传回 CPU 以便与标签比较
        predictions = xp.argmax(activations[-1], axis=1)
        return to_cpu(predictions)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X: 输入数据
            y: 真实标签
        
        返回:
            accuracy: 准确率
        """
        predictions = self.predict(X)
        # 确保 y 是 numpy 数组（在 CPU 上）
        y = np.asarray(y)
        return float(np.mean(predictions == y))


def main():
    """主函数：训练MLP模型"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("多层感知机 (MLP) - Fashion-MNIST分类")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"图像大小: {X_train.shape[1]} 像素 (28x28)")
    
    # 创建MLP模型
    print("\n创建MLP模型...")
    print("网络结构: 784 -> 256 -> 128 -> 10")
    model = MLP(
        layer_sizes=[784, 256, 128, 10],
        learning_rate=0.1,
        activation='relu'
    )
    
    # 训练模型
    print("\n开始训练...")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=128
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    
    training_time = time.time() - start_time
    
    # 生成详细报告
    generate_training_report(
        model_name="多层感知机 (MLP)",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        layer_info="784 -> 256 -> 128 -> 10",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()

