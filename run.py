"""
快速启动脚本
让用户选择要训练的神经网络模型
支持自动运行所有模型模式
"""
import os
import sys
import argparse


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print(" " * 15 + "Fashion-MNIST 神经网络训练")
    print("=" * 60)
    print("\n请选择要训练的模型：\n")
    print("  1. MLP      - 多层感知机 (最快, 推荐入门)")
    print("  2. CNN      - 卷积神经网络 (较慢)")
    print("  3. LeNet-5  - 经典LeNet架构 (中等)")
    print("  4. ResNet   - 残差网络 (较慢)")
    print("  5. Wide ResNet-28-10 + Random Erasing (96.3% benchmark, 最慢)")
    print("  6. DenseNet-BC - Dense连接网络 (95.4% benchmark, 很慢)")
    print("  7. Capsule Network - 胶囊网络 (93.6% benchmark, 很慢)")
    print("  8. 测试数据加载")
    print("  9. 运行所有新模型 (5-7)")
    print("  0. 退出")
    print("\n" + "=" * 60)


def run_model(choice):
    """运行选择的模型"""
    models = {
        '1': ('mlp.py', '多层感知机 (MLP)'),
        '2': ('cnn.py', '卷积神经网络 (CNN)'),
        '3': ('lenet.py', 'LeNet-5'),
        '4': ('resnet.py', '残差网络 (ResNet)'),
        '5': ('wide_resnet.py', 'Wide ResNet-28-10 + Random Erasing'),
        '6': ('densenet.py', 'DenseNet-BC'),
        '7': ('capsule_network.py', 'Capsule Network'),
        '8': ('test_data.py', '数据加载测试')
    }
    
    if choice in models:
        script, name = models[choice]
        print(f"\n正在启动 {name}...")
        print("-" * 60)
        
        # 运行对应的脚本
        if choice == '1':
            import mlp
            mlp.main()
        elif choice == '2':
            import cnn
            cnn.main()
        elif choice == '3':
            import lenet
            lenet.main()
        elif choice == '4':
            import resnet
            resnet.main()
        elif choice == '5':
            import wide_resnet
            wide_resnet.main()
        elif choice == '6':
            import densenet
            densenet.main()
        elif choice == '7':
            import capsule_network
            capsule_network.main()
        elif choice == '8':
            import test_data
            test_data.test_data_loading()
        
        print("\n" + "=" * 60)
        print("任务完成！")
        print("=" * 60)
        return True
    else:
        print("\n无效的选择，请重试。")
        return False


def run_all_models(include_test=False):
    """自动运行所有模型"""
    from utils import generate_summary_report
    
    print("\n" + "=" * 60)
    print(" " * 15 + "自动运行所有模型")
    print("=" * 60)
    
    # 要运行的模型列表（不包括测试数据加载，除非指定）
    models_to_run = ['1', '2', '3', '4', '5', '6', '7']
    if include_test:
        models_to_run.append('8')
    
    model_names = {
        '1': '多层感知机 (MLP)',
        '2': '卷积神经网络 (CNN)',
        '3': 'LeNet-5',
        '4': '残差网络 (ResNet)',
        '5': 'Wide ResNet-28-10 + Random Erasing',
        '6': 'DenseNet-BC',
        '7': 'Capsule Network',
        '8': '数据加载测试'
    }
    
    total = len(models_to_run)
    success_count = 0
    failed_models = []
    model_results = []  # 用于存储所有模型的结果
    
    for idx, choice in enumerate(models_to_run, 1):
        model_name = model_names[choice]
        print(f"\n[{idx}/{total}] 正在运行: {model_name}")
        print("=" * 60)
        
        result = {
            'model_name': model_name,
            'status': 'failed',
            'train_acc': 0,
            'test_acc': 0,
            'training_time': 0,
            'error': None
        }
        
        try:
            # 记录开始时间
            import time
            start_time = time.time()
            
            if run_model(choice):
                success_count += 1
                result['status'] = 'success'
                result['training_time'] = time.time() - start_time
                
                # 尝试从报告中提取准确率（如果模型训练成功）
                # 注意：这里我们无法直接获取准确率，因为run_model不返回这些信息
                # 但我们可以标记为成功，汇总报告会显示成功状态
                print(f"\n✓ {model_name} 运行成功")
            else:
                failed_models.append(model_name)
                result['error'] = '运行失败'
                print(f"\n✗ {model_name} 运行失败")
        except Exception as e:
            failed_models.append(model_name)
            result['error'] = str(e)
            print(f"\n✗ {model_name} 运行出错: {e}")
        
        model_results.append(result)
        
        # 在模型之间添加分隔
        if idx < total:
            print("\n" + "-" * 60)
            print("准备运行下一个模型...")
            print("-" * 60)
    
    # 显示总结
    print("\n" + "=" * 60)
    print(" " * 15 + "所有模型运行完成")
    print("=" * 60)
    print(f"\n总计: {total} 个模型")
    print(f"成功: {success_count} 个")
    print(f"失败: {len(failed_models)} 个")
    
    if failed_models:
        print(f"\n失败的模型:")
        for model in failed_models:
            print(f"  - {model}")
    
    print("=" * 60)
    
    # 生成汇总报告
    print("\n正在生成汇总报告...")
    try:
        summary_file = generate_summary_report(model_results)
        print(f"\n✓ 所有报告已生成完成！")
        print(f"  汇总报告: {summary_file}")
    except Exception as e:
        print(f"\n⚠ 生成汇总报告时出错: {e}")
        print("   但各模型的单独报告应该已经生成在 reports/ 目录中")


def show_info():
    """显示项目信息"""
    print("\n" + "=" * 60)
    print("项目信息")
    print("=" * 60)
    print("\n数据集: Fashion-MNIST")
    print("  - 训练集: 60,000 张图像")
    print("  - 测试集: 10,000 张图像")
    print("  - 图像尺寸: 28x28 灰度图")
    print("  - 类别数: 10 (衣服和鞋子)")
    print("\n实现的模型:")
    print("  1. MLP (多层感知机) - 全连接网络")
    print("  2. CNN (卷积神经网络) - 带卷积层")
    print("  3. LeNet-5 - 经典CNN架构")
    print("  4. ResNet - 带残差连接")
    print("  5. Wide ResNet-28-10 + Random Erasing - 96.3% benchmark")
    print("  6. DenseNet-BC - Dense连接, 95.4% benchmark")
    print("  7. Capsule Network - 胶囊网络, 93.6% benchmark")
    print("\n特点:")
    print("  ✓ 纯NumPy实现，无需深度学习框架")
    print("  ✓ 详细的代码注释")
    print("  ✓ 完整的训练报告")
    print("  ✓ 各类别性能分析")
    print("\n更多信息请查看 README.md")
    print("=" * 60)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Fashion-MNIST 神经网络训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py              # 交互模式
  python run.py --auto       # 自动运行所有模型
  python run.py -a           # 自动运行所有模型（简写）
  python run.py --auto --test # 自动运行所有模型（包括数据测试）
        """
    )
    parser.add_argument(
        '-a', '--auto',
        action='store_true',
        help='自动运行所有模型（不包含数据测试）'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='在自动模式下包含数据加载测试'
    )
    
    args = parser.parse_args()
    
    # 如果是自动模式
    if args.auto:
        show_info()
        run_all_models(include_test=args.test)
        return
    
    # 交互模式
    show_info()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n请输入选项 (0-9): ").strip()
            
            if choice == '0':
                print("\n再见！")
                break
            elif choice == '9':
                # 运行所有新模型
                print("\n将依次运行: Wide ResNet, DenseNet, Capsule Network")
                confirm = input(f"\n确认运行？这将需要较长时间 (y/n): ").strip().lower()
                if confirm == 'y':
                    for new_choice in ['5', '6', '7']:
                        run_model(new_choice)
                        print("\n" + "-" * 60)
                    
                    # 生成总结报告
                    print("\n正在生成总结报告...")
                    import generate_summary_report
                    generate_summary_report.main()
                    
                    cont = input("\n是否继续训练其他模型？(y/n): ").strip().lower()
                    if cont != 'y':
                        print("\n再见！")
                        break
                else:
                    print("已取消。")
            elif choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                confirm = input(f"\n确认运行？(y/n): ").strip().lower()
                if confirm == 'y':
                    run_model(choice)
                    
                    # 询问是否继续
                    cont = input("\n是否继续训练其他模型？(y/n): ").strip().lower()
                    if cont != 'y':
                        print("\n再见！")
                        break
                else:
                    print("已取消。")
            else:
                print("\n无效的选项，请输入 0-9 之间的数字。")
        
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试。")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n程序错误: {e}")
        sys.exit(1)

