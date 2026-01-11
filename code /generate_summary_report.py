"""
生成所有模型的总结报告
对比分析所有深度学习方法的性能
"""
import os
import glob
from datetime import datetime
import re


def parse_report_file(filepath):
    """
    解析单个报告文件，提取关键信息
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取模型名称
        model_name_match = re.search(r'【(.+?)】训练报告', content)
        if not model_name_match:
            model_name_match = re.search(r'\s+(.+?)\s+训练报告', content)
        model_name = model_name_match.group(1).strip() if model_name_match else "未知模型"
        
        # 提取测试准确率
        test_acc_match = re.search(r'测试准确率:\s*([\d.]+)', content)
        test_acc = float(test_acc_match.group(1)) if test_acc_match else 0.0
        
        # 提取训练准确率
        train_acc_match = re.search(r'训练准确率.*?:\s*([\d.]+)', content)
        train_acc = float(train_acc_match.group(1)) if train_acc_match else 0.0
        
        # 提取训练时间
        time_match = re.search(r'训练时间:\s*([\d.]+)\s*秒', content)
        training_time = float(time_match.group(1)) if time_match else 0.0
        
        # 提取网络结构
        structure_match = re.search(r'网络结构:\s*(.+?)(?:\n|$)', content)
        structure = structure_match.group(1).strip() if structure_match else "未知"
        
        # 提取训练轮数
        epochs_match = re.search(r'训练轮数:\s*(\d+)', content)
        epochs = int(epochs_match.group(1)) if epochs_match else 0
        
        # 提取学习率
        lr_match = re.search(r'学习率:\s*([\d.]+)', content)
        learning_rate = float(lr_match.group(1)) if lr_match else 0.0
        
        # 提取文件修改时间作为训练时间
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        return {
            'model_name': model_name,
            'test_acc': test_acc,
            'train_acc': train_acc,
            'training_time': training_time,
            'structure': structure,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'file_time': file_time,
            'filepath': filepath
        }
    except Exception as e:
        print(f"警告: 解析文件 {filepath} 时出错: {e}")
        return None


def generate_summary_report(reports_dir='reports'):
    """
    生成所有模型的总结报告
    """
    # 查找所有报告文件
    report_files = glob.glob(os.path.join(reports_dir, '*.txt'))
    
    if not report_files:
        print(f"错误: 在 {reports_dir} 目录中没有找到报告文件")
        return
    
    print(f"找到 {len(report_files)} 个报告文件")
    
    # 解析所有报告
    models_data = []
    for filepath in report_files:
        data = parse_report_file(filepath)
        if data:
            models_data.append(data)
    
    if not models_data:
        print("错误: 没有成功解析任何报告")
        return
    
    # 按测试准确率排序
    models_data.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # 生成总结报告
    report_lines = []
    
    def add_line(text):
        print(text)
        report_lines.append(text)
    
    add_line("=" * 80)
    add_line(" " * 25 + "Fashion-MNIST 深度学习模型总结报告")
    add_line("=" * 80)
    
    add_line(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_line(f"分析的模型数量: {len(models_data)}")
    
    # 1. 模型性能对比表
    add_line("\n" + "=" * 80)
    add_line("【一、模型性能对比】")
    add_line("=" * 80)
    
    add_line("\n排名 | 模型名称 | 测试准确率 | 训练准确率 | 训练时间(秒)")
    add_line("-" * 80)
    
    for i, model in enumerate(models_data, 1):
        add_line(f"{i:2d}   | {model['model_name']:30s} | {model['test_acc']:6.2%}     | "
                f"{model['train_acc']:6.2%}     | {model['training_time']:8.2f}")
    
    # 2. 最佳模型分析
    add_line("\n" + "=" * 80)
    add_line("【二、最佳模型分析】")
    add_line("=" * 80)
    
    best_model = models_data[0]
    add_line(f"\n🏆 最佳模型: {best_model['model_name']}")
    add_line(f"   测试准确率: {best_model['test_acc']:.4f} ({best_model['test_acc']:.2%})")
    add_line(f"   训练准确率: {best_model['train_acc']:.4f} ({best_model['train_acc']:.2%})")
    add_line(f"   训练时间: {best_model['training_time']:.2f} 秒 ({best_model['training_time']/60:.2f} 分钟)")
    add_line(f"   网络结构: {best_model['structure']}")
    add_line(f"   学习率: {best_model['learning_rate']}")
    add_line(f"   训练轮数: {best_model['epochs']}")
    
    # 3. 各类别最优模型
    add_line("\n" + "=" * 80)
    add_line("【三、各类别最优模型】")
    add_line("=" * 80)
    
    # 最快训练速度
    fastest_model = min(models_data, key=lambda x: x['training_time'])
    add_line(f"\n⚡ 最快训练速度: {fastest_model['model_name']}")
    add_line(f"   训练时间: {fastest_model['training_time']:.2f} 秒")
    add_line(f"   测试准确率: {fastest_model['test_acc']:.2%}")
    
    # 最高准确率
    add_line(f"\n🎯 最高准确率: {best_model['model_name']}")
    add_line(f"   测试准确率: {best_model['test_acc']:.2%}")
    
    # 最佳性价比 (准确率/时间)
    efficiency = [(m['test_acc'] / (m['training_time'] / 60), m) for m in models_data if m['training_time'] > 0]
    if efficiency:
        best_efficiency = max(efficiency, key=lambda x: x[0])
        eff_score, eff_model = best_efficiency
        add_line(f"\n💎 最佳性价比: {eff_model['model_name']}")
        add_line(f"   准确率: {eff_model['test_acc']:.2%}")
        add_line(f"   训练时间: {eff_model['training_time']/60:.2f} 分钟")
        add_line(f"   性价比得分: {eff_score:.4f} (准确率/分钟)")
    
    # 4. 新增模型与基准对比
    add_line("\n" + "=" * 80)
    add_line("【四、Fashion-MNIST Benchmark 对比】")
    add_line("=" * 80)
    
    add_line("\n根据 GitHub Fashion-MNIST 官方 Benchmark:")
    add_line("  - Wide ResNet-28-10 + Random Erasing: 96.3% (官方benchmark)")
    add_line("  - DenseNet-BC: 95.4% (官方benchmark)")
    add_line("  - Capsule Network: 93.6% (官方benchmark)")
    
    add_line("\n本次实验结果:")
    for model in models_data:
        if 'Wide' in model['model_name'] or 'WRN' in model['model_name']:
            add_line(f"  - {model['model_name']}: {model['test_acc']:.1%} (本实验)")
        elif 'Dense' in model['model_name']:
            add_line(f"  - {model['model_name']}: {model['test_acc']:.1%} (本实验)")
        elif 'Capsule' in model['model_name'] or 'CapsNet' in model['model_name']:
            add_line(f"  - {model['model_name']}: {model['test_acc']:.1%} (本实验)")
    
    # 5. 详细统计分析
    add_line("\n" + "=" * 80)
    add_line("【五、统计分析】")
    add_line("=" * 80)
    
    test_accs = [m['test_acc'] for m in models_data]
    train_accs = [m['train_acc'] for m in models_data]
    times = [m['training_time'] for m in models_data]
    
    add_line(f"\n测试准确率统计:")
    add_line(f"  - 平均值: {sum(test_accs)/len(test_accs):.2%}")
    add_line(f"  - 最高值: {max(test_accs):.2%}")
    add_line(f"  - 最低值: {min(test_accs):.2%}")
    add_line(f"  - 标准差: {(sum([(x-sum(test_accs)/len(test_accs))**2 for x in test_accs])/len(test_accs))**0.5:.4f}")
    
    add_line(f"\n训练时间统计:")
    add_line(f"  - 平均值: {sum(times)/len(times):.2f} 秒 ({sum(times)/len(times)/60:.2f} 分钟)")
    add_line(f"  - 最快: {min(times):.2f} 秒")
    add_line(f"  - 最慢: {max(times):.2f} 秒")
    
    # 6. 模型架构对比
    add_line("\n" + "=" * 80)
    add_line("【六、模型架构特点】")
    add_line("=" * 80)
    
    architecture_analysis = {
        'Wide ResNet': '增加网络宽度而非深度，使用Random Erasing数据增强，适合复杂图像分类',
        'DenseNet': 'Dense连接促进特征重用，减少梯度消失，参数效率高',
        'Capsule': '使用向量表示特征，动态路由算法，更好保留空间层次关系',
        'ResNet': '残差连接解决梯度消失，可以训练很深的网络',
        'LeNet': '经典CNN架构，结构简单，训练快速',
        'MLP': '全连接网络，基准模型',
        'CNN': '标准卷积神经网络'
    }
    
    for model in models_data:
        model_type = None
        for key in architecture_analysis.keys():
            if key in model['model_name']:
                model_type = key
                break
        
        if model_type:
            add_line(f"\n{model['model_name']}:")
            add_line(f"  特点: {architecture_analysis[model_type]}")
            add_line(f"  性能: 测试准确率 {model['test_acc']:.2%}, 训练时间 {model['training_time']/60:.2f} 分钟")
    
    # 7. 结论与建议
    add_line("\n" + "=" * 80)
    add_line("【七、结论与建议】")
    add_line("=" * 80)
    
    add_line("\n📊 实验结论:")
    add_line(f"  1. 在本次实验中，{best_model['model_name']} 取得了最佳性能")
    add_line(f"  2. 所有模型的平均测试准确率为 {sum(test_accs)/len(test_accs):.2%}")
    add_line(f"  3. 训练时间范围从 {min(times):.2f} 秒到 {max(times):.2f} 秒")
    
    add_line("\n💡 应用建议:")
    add_line("  - 追求最高准确率: 推荐使用 Wide ResNet 或 DenseNet")
    add_line("  - 快速原型开发: 推荐使用 LeNet 或标准CNN")
    add_line("  - 资源受限环境: 推荐使用 MLP 或简化版CNN")
    add_line("  - 研究目的: 推荐尝试 Capsule Network 等新颖架构")
    
    add_line("\n🔬 进一步优化方向:")
    add_line("  1. 数据增强: 使用更多增强技术(Random Erasing, Cutout, Mixup)")
    add_line("  2. 正则化: 调整Dropout比率，使用L2正则化")
    add_line("  3. 学习率策略: 使用学习率衰减、warmup等技术")
    add_line("  4. 模型集成: 结合多个模型的预测结果")
    add_line("  5. 超参数优化: 使用网格搜索或贝叶斯优化")
    
    # 8. 参考文献
    add_line("\n" + "=" * 80)
    add_line("【八、参考文献】")
    add_line("=" * 80)
    
    add_line("\n1. Fashion-MNIST官方仓库:")
    add_line("   https://github.com/zalandoresearch/fashion-mnist")
    
    add_line("\n2. 相关论文:")
    add_line("   - Wide Residual Networks (Zagoruyko & Komodakis, 2016)")
    add_line("   - Densely Connected Convolutional Networks (Huang et al., 2017)")
    add_line("   - Dynamic Routing Between Capsules (Sabour et al., 2017)")
    add_line("   - Deep Residual Learning (He et al., 2015)")
    add_line("   - Random Erasing Data Augmentation (Zhong et al., 2017)")
    
    add_line("\n" + "=" * 80)
    add_line("报告生成完成！")
    add_line("=" * 80)
    
    # 保存报告
    output_filename = f"总结报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(reports_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ 总结报告已保存到: {output_path}")
    
    return output_path


def main():
    """主函数"""
    print("=" * 70)
    print("生成Fashion-MNIST深度学习模型总结报告")
    print("=" * 70)
    
    # 确定报告目录
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        print(f"错误: 报告目录 {reports_dir} 不存在")
        print("请先运行各个模型训练脚本生成报告")
        return
    
    # 生成总结报告
    output_path = generate_summary_report(reports_dir)
    
    if output_path:
        print(f"\n✓ 成功生成总结报告")
        print(f"✓ 报告位置: {output_path}")


if __name__ == '__main__':
    main()

