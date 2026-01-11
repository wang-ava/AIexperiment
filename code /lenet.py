"""
LeNet-5 神经网络 (PyTorch版本)
经典的卷积神经网络架构，由Yann LeCun于1998年提出
原始用于手写数字识别，这里应用于Fashion-MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed


class LeNet5(nn.Module):
    """
    LeNet-5卷积神经网络（PyTorch版本）
    结构: Conv1(6) -> ReLU -> AvgPool -> Conv2(16) -> ReLU -> AvgPool -> 
          Flatten -> FC1(120) -> ReLU -> FC2(84) -> ReLU -> FC3(10)
    使用 ReLU 激活函数和 He 初始化
    """
    
    def __init__(self, learning_rate=0.001):
        """
        初始化LeNet-5
        
        参数:
            learning_rate: 学习率（默认0.001）
        """
        super(LeNet5, self).__init__()
        
        self.learning_rate = learning_rate
        self.device = get_device()
        
        # 卷积层1: 1 -> 6, kernel=5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        
        # 平均池化层1: 2x2
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 6 -> 16, kernel=5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        # 平均池化层2: 2x2
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 全连接层1: 16*4*4 -> 120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        
        # 全连接层2: 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        
        # 输出层: 84 -> 10
        self.fc3 = nn.Linear(84, 10)
        
        # 初始化权重
        self._initialize_weights()
        
        self.to(self.device)
    
    def _initialize_weights(self):
        """使用He初始化（适用于ReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像, shape (batch, 1, 28, 28)
        
        返回:
            output: 输出logits, shape (batch, 10)
        """
        # Conv1 -> ReLU -> AvgPool
        x = self.avgpool1(F.relu(self.conv1(x)))  # (batch, 6, 12, 12)
        
        # Conv2 -> ReLU -> AvgPool
        x = self.avgpool2(F.relu(self.conv2(x)))  # (batch, 16, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 16*4*4)
        
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))  # (batch, 120)
        
        # FC2 -> ReLU
        x = F.relu(self.fc2(x))  # (batch, 84)
        
        # FC3 (输出层，不使用激活函数，使用交叉熵损失)
        x = self.fc3(x)  # (batch, 10)
        
        return x
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """
        训练模型
        
        参数:
            X_train: 训练数据 (numpy数组, shape: [N, 784])
            y_train: 训练标签 (numpy数组)
            X_test: 测试数据 (numpy数组, shape: [N, 784])
            y_test: 测试标签 (numpy数组)
            epochs: 训练轮数
            batch_size: 批次大小
        
        返回:
            history: 训练历史记录
        """
        # 将数据reshape为图像格式并转换为张量
        X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
        X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
        
        X_train_tensor = torch.FloatTensor(X_train_img).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_img).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 使用交叉熵损失和Adam优化器（带权重衰减）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # 添加学习率调度器（更激进的衰减）
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
                # Reshape为图像格式
                X_batch_img = X_batch.reshape(-1, 1, 28, 28).astype(np.float32)
                X_batch_tensor = torch.FloatTensor(X_batch_img).to(self.device)
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
            X: 输入数据 (numpy数组, shape: [N, 784] 或 [N, 1, 28, 28])
        
        返回:
            predictions: 预测的类别（numpy 数组）
        """
        self.eval()  # 设置为评估模式
        
        with torch.no_grad():
            # 转换为PyTorch张量并reshape
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            
            if X.ndim == 2:  # shape: [N, 784]
                X_img = X.reshape(-1, 1, 28, 28).astype(np.float32)
            else:  # shape: [N, 1, 28, 28]
                X_img = X.astype(np.float32)
            
            X_tensor = torch.FloatTensor(X_img).to(self.device)
            
            # 前向传播
            outputs = self.forward(X_tensor)
            
            # 获取预测类别
            _, predictions = torch.max(outputs, 1)
            
            # 返回numpy数组
            return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X: 输入数据 (PyTorch张量, shape: [N, 1, 28, 28])
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
    """主函数：训练LeNet-5模型"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("LeNet-5 卷积神经网络 - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 70)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建LeNet-5模型（优化版）
    print("\n创建LeNet-5模型...")
    print("网络结构: Conv1(6) -> ReLU -> AvgPool -> Conv2(16) -> ReLU -> AvgPool -> FC1(120) -> ReLU -> FC2(84) -> ReLU -> FC3(10)")
    model = LeNet5(learning_rate=0.0005)
    
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
        if mem_info and mem_info['free'] > 80000:
            batch_size = 512
        elif mem_info and mem_info['free'] > 50000:
            batch_size = 256
        elif mem_info and mem_info['free'] > 20000:
            batch_size = 128
        elif mem_info and mem_info['free'] > 10000:
            batch_size = 64
        else:
            batch_size = 32
    else:
        batch_size = 64  # CPU模式
    
    print(f"使用batch_size={batch_size}进行训练")
    # 增加训练轮数以达到95%准确率
    history = model.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=80,
        batch_size=batch_size
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    X_train_tensor = torch.FloatTensor(X_train_img).to(model.device)
    y_train_tensor = torch.LongTensor(y_train).to(model.device)
    X_test_tensor = torch.FloatTensor(X_test_img).to(model.device)
    y_test_tensor = torch.LongTensor(y_test).to(model.device)
    
    train_acc = model.evaluate(X_train_tensor, y_train_tensor)
    test_acc = model.evaluate(X_test_tensor, y_test_tensor)
    
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
        layer_info="Conv1(6) -> ReLU -> AvgPool -> Conv2(16) -> ReLU -> AvgPool -> FC1(120) -> ReLU -> FC2(84) -> ReLU -> FC3(10)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
