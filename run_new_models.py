"""
运行Fashion-MNIST Benchmark中的高准确率模型
包括：
1. Wide ResNet-28-10 + Random Erasing (96.3% benchmark)
2. DenseNet-BC (95.4% benchmark)
3. Capsule Network (93.6% benchmark)

每个模型训练完成后会生成单独的报告
最后生成一个总结报告对比所有模型
"""
import sys
import time


def run_new_models():
    """运行所有新实现的高准确率模型"""
    
    print("=" * 80)
    print(" " * 20 + "Fashion-MNIST 高准确率模型训练")
    print("=" * 80)
    print("\n基于 Fashion-MNIST GitHub Benchmark:")
    print("  https://github.com/zalandoresearch/fashion-mnist")
    print("\n本次将训练以下3个模型:")
    print("  1. Wide ResNet-28-10 + Random Erasing  (目标: 96.3%)")
    print("  2. DenseNet-BC                         (目标: 95.4%)")
    print("  3. Capsule Network                     (目标: 93.6%)")
    print("\n⚠️  注意: 这些模型训练时间较长，请耐心等待")
    print("=" * 80)
    
    # 询问确认
    confirm = input("\n是否开始训练？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消训练。")
        return
    
    models = [
        ('wide_resnet', 'Wide ResNet-28-10 + Random Erasing', '96.3%'),
        ('densenet', 'DenseNet-BC', '95.4%'),
        ('capsule_network', 'Capsule Network', '93.6%')
    ]
    
    total = len(models)
    success_count = 0
    failed_models = []
    results = []
    
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    for idx, (module_name, model_name, benchmark_acc) in enumerate(models, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{total}] 训练 {model_name}")
        print(f"Benchmark准确率: {benchmark_acc}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 动态导入并运行模型
            module = __import__(module_name)
            print(f"\n✓ 模块 {module_name} 导入成功")
            
            print(f"\n开始训练 {model_name}...")
            print("-" * 80)
            
            module.main()
            
            training_time = time.time() - start_time
            
            print(f"\n✓ {model_name} 训练完成")
            print(f"  训练耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
            
            success_count += 1
            results.append({
                'model_name': model_name,
                'status': 'success',
                'time': training_time,
                'benchmark': benchmark_acc
            })
            
        except ImportError as e:
            error_msg = f"模块导入失败: {e}"
            print(f"\n✗ {model_name} 失败: {error_msg}")
            failed_models.append((model_name, error_msg))
            results.append({
                'model_name': model_name,
                'status': 'failed',
                'error': error_msg,
                'benchmark': benchmark_acc
            })
            
        except Exception as e:
            error_msg = f"训练出错: {e}"
            print(f"\n✗ {model_name} 失败: {error_msg}")
            failed_models.append((model_name, error_msg))
            results.append({
                'model_name': model_name,
                'status': 'failed',
                'error': error_msg,
                'benchmark': benchmark_acc
            })
        
        # 模型间分隔
        if idx < total:
            print("\n" + "-" * 80)
            print("准备训练下一个模型...")
            print("-" * 80)
            time.sleep(2)  # 短暂暂停
    
    # 显示总结
    print("\n" + "=" * 80)
    print(" " * 25 + "训练完成总结")
    print("=" * 80)
    
    print(f"\n总计: {total} 个模型")
    print(f"成功: {success_count} 个")
    print(f"失败: {len(failed_models)} 个")
    
    if results:
        print("\n模型训练结果:")
        for result in results:
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"  {status_icon} {result['model_name']:40s} - {result['status']:7s}", end="")
            if result['status'] == 'success':
                print(f" ({result['time']/60:.2f} 分钟)")
            else:
                print()
    
    if failed_models:
        print(f"\n失败的模型详情:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
    
    print("\n" + "=" * 80)
    
    # 生成总结报告
    if success_count > 0:
        print("\n正在生成总结报告...")
        print("-" * 80)
        
        try:
            import generate_summary_report
            generate_summary_report.main()
            print("\n✓ 总结报告生成成功！")
            print("  报告保存在 reports/ 目录")
        except Exception as e:
            print(f"\n⚠ 生成总结报告时出错: {e}")
            print("  但各模型的单独报告应该已经在 reports/ 目录中")
    else:
        print("\n⚠ 没有模型训练成功，跳过总结报告生成")
    
    print("\n" + "=" * 80)
    print("所有任务完成！")
    print("=" * 80)
    
    # 显示报告位置
    print("\n查看结果:")
    print("  - 单独报告: reports/ 目录下的各个 .txt 文件")
    print("  - 总结报告: reports/总结报告_*.txt")
    print("\n" + "=" * 80)


def main():
    """主函数"""
    try:
        run_new_models()
    except KeyboardInterrupt:
        print("\n\n用户中断训练。")
        print("已完成的模型报告已保存在 reports/ 目录。")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

