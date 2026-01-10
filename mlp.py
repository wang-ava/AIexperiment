"""
多层感知机 (Multi-Layer Perceptron, MLP)
使用 PyTorch 实现全连接神经网络
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed


class MLP(nn.Module):
    """
    多层感知机神经网络 (PyTorch版本)
    支持任意层数和神经元数量的全连接网络
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu', dropout_rate=0.4):
        """
        初始化MLP
        
        参数:
            layer_sizes: 列表，每层的神经元数量，如[784, 128, 64, 10]
            learning_rate: 学习率
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
            dropout_rate: Dropout比率，用于正则化减少过拟合（默认0.4）
        """
        super(MLP, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.num_layers = len(layer_sizes)
        self.dropout_rate = dropout_rate
        self.device = get_device()
        
        # 构建全连接层（添加Dropout正则化）
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # 除了最后一层，都添加激活函数和Dropout
            if i < self.num_layers - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                # 在激活函数后添加Dropout（最后一层前不添加）
                layers.append(nn.Dropout(p=dropout_rate))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        # 使用He初始化（适用于ReLU）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_type == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据, shape (batch_size, input_dim)
        
        返回:
            output: 网络输出（未经过softmax的logits）
        """
        return self.network(x)
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """
        训练模型
        
        参数:
            X_train: 训练数据 (numpy数组)
            y_train: 训练标签 (numpy数组)
            X_test: 测试数据 (numpy数组)
            y_test: 测试标签 (numpy数组)
            epochs: 训练轮数
            batch_size: 批次大小
        
        返回:
            history: 训练历史记录
        """
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 使用交叉熵损失（包含softmax）和权重衰减正则化
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # 添加学习率调度器（更激进的学习率衰减）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        
        # 记录训练历史
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        self.train()  # 设置为训练模式
        
        for epoch in range(epochs):
            # 创建mini-batches
            batches = create_mini_batches(X_train, y_train, batch_size)
            
            epoch_loss = 0
            num_batches = 0
            
            # 训练每个batch
            for X_batch, y_batch in batches:
                X_batch_tensor = torch.FloatTensor(X_batch).to(self.device)
                y_batch_tensor = torch.LongTensor(y_batch).to(self.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.forward(X_batch_tensor)
                loss = criterion(outputs, y_batch_tensor)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # 计算训练和测试准确率
            train_acc = self.evaluate(X_train_tensor, y_train_tensor)
            test_acc = self.evaluate(X_test_tensor, y_test_tensor)
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # 更新学习率调度器
            scheduler.step(test_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - '
                  f'Train Accuracy: {train_acc:.4f}, '
                  f'Test Accuracy: {test_acc:.4f} - '
                  f'LR: {current_lr:.6f}')
            
            # 定期清理GPU内存
            if (epoch + 1) % 5 == 0:
                clear_gpu_memory()
        
        return history
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据 (numpy数组或PyTorch张量)
        
        返回:
            predictions: 预测的类别（numpy 数组）
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 转换为PyTorch张量
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            
            # 前向传播
            outputs = self.forward(X)
            
            # 获取预测类别
            _, predictions = torch.max(outputs, 1)
            
            # 返回numpy数组
            return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X: 输入数据 (PyTorch张量)
            y: 真实标签 (PyTorch张量)
        
        返回:
            accuracy: 准确率
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(X)
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == y).float().mean().item()
        
        return accuracy


def main():
    """主函数：训练MLP模型"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("多层感知机 (MLP) - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"图像大小: {X_train.shape[1]} 像素 (28x28)")
    
    # 创建MLP模型（优化版：更深的网络、Dropout正则化和更小的学习率）
    print("\n创建MLP模型...")
    print("网络结构: 784 -> 512(ReLU+Dropout) -> 256(ReLU+Dropout) -> 128(ReLU+Dropout) -> 10 (带Dropout正则化)")
    model = MLP(
        layer_sizes=[784, 512, 256, 128, 10],
        learning_rate=0.0005,
        activation='relu',
        dropout_rate=0.4  # 添加40%的Dropout减少过拟合
    )
    
    # 显示GPU信息
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info:
            print(f"\nGPU内存: {mem_info['used']:.1f}MB / {mem_info['total']:.1f}MB (可用: {mem_info['free']:.1f}MB)")
    
    # 训练模型
    print("\n开始训练...")
    
    # 根据GPU内存动态调整batch_size
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info and mem_info['free'] > 80000:  # 如果可用显存>80GB
            batch_size = 1024
        elif mem_info and mem_info['free'] > 50000:
            batch_size = 512
        elif mem_info and mem_info['free'] > 20000:
            batch_size = 256
        elif mem_info and mem_info['free'] > 10000:
            batch_size = 128
        else:
            batch_size = 64
    else:
        batch_size = 128  # CPU模式
    
    print(f"使用batch_size={batch_size}进行训练")
    # 增加训练轮数以达到95%准确率
    history = model.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=100,
        batch_size=batch_size
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    X_train_tensor = torch.FloatTensor(X_train).to(model.device)
    y_train_tensor = torch.LongTensor(y_train).to(model.device)
    X_test_tensor = torch.FloatTensor(X_test).to(model.device)
    y_test_tensor = torch.LongTensor(y_test).to(model.device)
    
    train_acc = model.evaluate(X_train_tensor, y_train_tensor)
    test_acc = model.evaluate(X_test_tensor, y_test_tensor)
    
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
        layer_info="784 -> 512(ReLU+Dropout0.4) -> 256(ReLU+Dropout0.4) -> 128(ReLU+Dropout0.4) -> 10",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
