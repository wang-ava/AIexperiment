"""
Wide Residual Network (Wide ResNet) - PyTorch版本
论文: "Wide Residual Networks" (Sergey Zagoruyko, Nikos Komodakis, 2016)

关键特性:
1. 增加网络宽度而不是深度
2. Batch Normalization
3. Dropout正则化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed
from data_augmentation import DataAugmentation, label_smoothing_loss


class WideBasicBlock(nn.Module):
    """Wide ResNet基础块"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """Wide Residual Network - 简化版本适用于Fashion-MNIST"""
    
    def __init__(self, depth=16, widen_factor=10, dropout_rate=0.3, learning_rate=0.1):
        super(WideResNet, self).__init__()
        
        self.learning_rate = learning_rate
        self.device = get_device()
        
        assert (depth - 4) % 6 == 0, 'depth应该是6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        nChannels = [16, 16*k, 32*k, 64*k]
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # 残差块
        self.layer1 = self._make_layer(nChannels[0], nChannels[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(nChannels[1], nChannels[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(nChannels[2], nChannels[3], n, stride=2, dropout_rate=dropout_rate)
        
        # BatchNorm和全连接层
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], 10)
        
        # 初始化权重
        self._initialize_weights()
        self.to(self.device)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(WideBasicBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, use_augmentation=True, label_smoothing=0.1):
        """训练模型"""
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
        
        self.train()
        
        for epoch in range(epochs):
            batches = create_mini_batches(X_train, y_train, batch_size)
            epoch_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in batches:
                X_batch_img = X_batch.reshape(-1, 1, 28, 28).astype(np.float32)
                X_batch_tensor = torch.FloatTensor(X_batch_img).to(self.device)
                y_batch_tensor = torch.LongTensor(y_batch).to(self.device)
                
                # 应用数据增强
                X_batch_tensor = aug.augment(X_batch_tensor)
                
                optimizer.zero_grad()
                outputs = self.forward(X_batch_tensor)
                
                # 使用Label Smoothing损失
                if label_smoothing > 0:
                    loss = label_smoothing_loss(outputs, y_batch_tensor, num_classes=10, smoothing=label_smoothing)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, y_batch_tensor)
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # 更新学习率
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # 评估（不使用数据增强）
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
            
            if (epoch + 1) % 5 == 0:
                clear_gpu_memory()
        
        return history
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            if X.ndim == 2:
                X_img = X.reshape(-1, 1, 28, 28).astype(np.float32)
            else:
                X_img = X.astype(np.float32)
            X_tensor = torch.FloatTensor(X_img).to(self.device)
            outputs = self.forward(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """评估模型"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == y).float().mean().item()
        return accuracy


def main():
    """主函数：训练WideResNet模型"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("Wide Residual Network (Wide ResNet) - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 70)
    
    set_random_seed(42)
    
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    print("\n创建WideResNet模型...")
    print("优化策略: 数据增强 + Label Smoothing + CosineAnnealingLR + AdamW")
    model = WideResNet(depth=16, widen_factor=10, dropout_rate=0.3, learning_rate=0.001)
    
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info:
            print(f"\nGPU内存: {mem_info['used']:.1f}MB / {mem_info['total']:.1f}MB (可用: {mem_info['free']:.1f}MB)")
    
    print("\n开始训练...")
    
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info and mem_info['free'] > 80000:
            batch_size = 128
        elif mem_info and mem_info['free'] > 50000:
            batch_size = 64
        else:
            batch_size = 32
    else:
        batch_size = 32
    
    print(f"使用batch_size={batch_size}进行训练")
    # 增加训练轮数并使用数据增强以提高准确率
    history = model.train_model(
        X_train, y_train, X_test, y_test,
        epochs=120, batch_size=batch_size,
        use_augmentation=True, label_smoothing=0.1
    )
    
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
    
    generate_training_report(
        model_name="Wide ResNet",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model=model,
        layer_info="WideResNet-16-10 (depth=16, widen_factor=10) (数据增强+Label Smoothing)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
