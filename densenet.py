"""
DenseNet-BC (Densely Connected Convolutional Networks) - PyTorch版本
论文: "Densely Connected Convolutional Networks" (Gao Huang et al., 2017)

关键特性:
- Dense连接减少梯度消失
- 特征重用，参数效率高
- Batch Normalization
- Growth rate控制网络宽度
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed
from data_augmentation import DataAugmentation, label_smoothing_loss


class DenseLayer(nn.Module):
    """DenseNet基本层（Bottleneck版本）"""
    
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0):
        super(DenseLayer, self).__init__()
        # Bottleneck: 1x1卷积
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        # 3x3卷积
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.Module):
    """DenseBlock: 包含多个DenseLayer"""
    
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Module):
    """Transition层：压缩特征图数量"""
    
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    """DenseNet-BC适用于Fashion-MNIST的简化版本"""
    
    def __init__(self, growth_rate=12, num_layers=[6, 6, 6], compression=0.5, learning_rate=0.1):
        super(DenseNet, self).__init__()
        
        self.learning_rate = learning_rate
        self.device = get_device()
        
        # 初始特征数
        num_features = 2 * growth_rate
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # DenseBlocks和Transition层
        for i, num_layer in enumerate(num_layers):
            block = DenseBlock(num_layer, num_features, 4, growth_rate, drop_rate=0.2)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layer * growth_rate
            
            if i != len(num_layers) - 1:
                trans = Transition(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        # 最终BatchNorm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, 10)
        
        # 初始化权重
        self._initialize_weights()
        self.to(self.device)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        # 移除inplace操作以避免梯度计算错误
        out = F.relu(features)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
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
    """主函数：训练DenseNet模型"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("DenseNet-BC - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 70)
    
    set_random_seed(42)
    
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 根据GPU内存调整层数
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info and mem_info['free'] > 80000:
            num_layers = [6, 6, 6]
        elif mem_info and mem_info['free'] > 50000:
            num_layers = [4, 4, 4]
        else:
            num_layers = [3, 3, 3]
    else:
        num_layers = [3, 3, 3]
    
    print(f"\n创建DenseNet模型 (layers: {num_layers})...")
    print("优化策略: 数据增强 + Label Smoothing + CosineAnnealingLR + AdamW")
    model = DenseNet(growth_rate=12, num_layers=num_layers, compression=0.5, learning_rate=0.001)
    
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
        model_name="DenseNet-BC",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model=model,
        layer_info=f"DenseNet-BC (growth_rate=12, layers={num_layers}) (数据增强+Label Smoothing)",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
