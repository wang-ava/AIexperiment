"""
卷积神经网络 (Convolutional Neural Network, CNN) - PyTorch版本
使用PyTorch实现标准的CNN架构
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed
from data_augmentation import DataAugmentation, label_smoothing_loss


class CNN(nn.Module):
    """
    卷积神经网络 (PyTorch版本)
    结构: Conv(16) -> ReLU -> MaxPool -> Conv(32) -> ReLU -> MaxPool -> Flatten -> Dense(128) -> Dense(10)
    """
    
    def __init__(self, learning_rate=0.01):
        """
        初始化CNN
        
        参数:
            learning_rate: 学习率
        """
        super(CNN, self).__init__()
        
        self.learning_rate = learning_rate
        self.device = get_device()
        
        # 第一个卷积块（增加通道数）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三个卷积块（新增）
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 全连接层（增加层数和dropout）
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像, shape (batch, 1, 28, 28)
        
        返回:
            output: 输出logits, shape (batch, 10)
        """
        # 第一个卷积块: Conv -> BN -> ReLU -> MaxPool -> Dropout
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))  # (batch, 32, 14, 14)
        
        # 第二个卷积块: Conv -> BN -> ReLU -> MaxPool -> Dropout
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))  # (batch, 64, 7, 7)
        
        # 第三个卷积块: Conv -> BN -> ReLU -> MaxPool -> Dropout
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))  # (batch, 128, 3, 3)
        
        # Flatten
        x = self.flatten(x)  # (batch, 128*3*3)
        
        # 全连接层
        x = F.relu(self.fc1(x))  # (batch, 256)
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))  # (batch, 128)
        x = self.fc3(x)  # (batch, 10)
        
        return x
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, use_augmentation=True, label_smoothing=0.1):
        """
        训练模型
        
        参数:
            X_train: 训练数据 (numpy数组, shape: [N, 784])
            y_train: 训练标签 (numpy数组)
            X_test: 测试数据 (numpy数组, shape: [N, 784])
            y_test: 测试标签 (numpy数组)
            epochs: 训练轮数
            batch_size: 批次大小
            use_augmentation: 是否使用数据增强
            label_smoothing: Label smoothing系数
        
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
        
        # 初始化数据增强
        aug = DataAugmentation(use_augmentation=use_augmentation)
        
        # 使用AdamW优化器和CosineAnnealingLR学习率调度器
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 记录训练历史
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        # 最佳测试准确率（用于early stopping）
        best_test_acc = 0.0
        patience_counter = 0
        patience = 15
        
        self.train()  # 设置为训练模式
        
        for epoch in range(epochs):
            # 创建mini-batches
            batches = create_mini_batches(X_train, y_train, batch_size)
            
            epoch_loss = 0
            num_batches = 0
            
            # 训练每个batch
            for i, (X_batch, y_batch) in enumerate(batches):
                # Reshape为图像格式
                X_batch_img = X_batch.reshape(-1, 1, 28, 28).astype(np.float32)
                X_batch_tensor = torch.FloatTensor(X_batch_img).to(self.device)
                y_batch_tensor = torch.LongTensor(y_batch).to(self.device)
                
                # 应用数据增强
                X_batch_tensor = aug.augment(X_batch_tensor)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.forward(X_batch_tensor)
                
                # 使用Label Smoothing损失
                if label_smoothing > 0:
                    loss = label_smoothing_loss(outputs, y_batch_tensor, num_classes=10, smoothing=label_smoothing)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, y_batch_tensor)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 100 == 0:
                    print(f'  Batch {i + 1}/{len(batches)} 完成', end='\r')
            
            # 更新学习率
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # 计算训练和测试准确率（不使用数据增强）
            self.eval()
            train_acc = self.evaluate(X_train_tensor[:5000], y_train_tensor[:5000])
            test_acc = self.evaluate(X_test_tensor, y_test_tensor)
            self.train()
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # Early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1} (best test acc: {best_test_acc:.4f})')
                    break
            
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
    """主函数：训练CNN模型"""
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("卷积神经网络 (CNN) - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载数据
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建CNN模型（优化版：更深的网络、BatchNorm、Dropout）
    print("\n创建CNN模型...")
    print("网络结构: Conv(32)->BN->ReLU->Pool->Dropout -> Conv(64)->BN->ReLU->Pool->Dropout -> Conv(128)->BN->ReLU->Pool->Dropout -> FC(256)->Dropout->FC(128)->FC(10)")
    print("优化策略: 数据增强 + Label Smoothing + CosineAnnealingLR + AdamW")
    model = CNN(learning_rate=0.001)
    
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
            batch_size = 256
        elif mem_info and mem_info['free'] > 50000:
            batch_size = 128
        elif mem_info and mem_info['free'] > 20000:
            batch_size = 64
        elif mem_info and mem_info['free'] > 10000:
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 32  # CPU模式
    
    print(f"使用batch_size={batch_size}进行训练")
    # 增加训练轮数并使用数据增强以提高准确率
    history = model.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=120,
        batch_size=batch_size,
        use_augmentation=True, label_smoothing=0.1
    )
    
    # 最终评估
    print("\n训练完成！正在生成报告...")
    X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    X_train_tensor = torch.FloatTensor(X_train_img).to(model.device)
    y_train_tensor = torch.LongTensor(y_train).to(model.device)
    X_test_tensor = torch.FloatTensor(X_test_img).to(model.device)
    y_test_tensor = torch.LongTensor(y_test).to(model.device)
    
    train_acc = model.evaluate(X_train_tensor[:5000], y_train_tensor[:5000])
    test_acc = model.evaluate(X_test_tensor, y_test_tensor)
    
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
        layer_info="Conv(32)->BN->ReLU->Pool->Dropout -> Conv(64)->BN->ReLU->Pool->Dropout -> Conv(128)->BN->ReLU->Pool->Dropout -> FC(256)->Dropout->FC(128)->FC(10) (数据增强+Label Smoothing)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
