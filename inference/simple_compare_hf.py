"""
SAM2模型对比脚本 - 使用HuggingFace加载方式

该脚本使用huggingface的预训练模型加载方式，对比原始SAM2模型和微调模型
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

# 尝试导入SAM2相关库
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("错误: 无法导入SAM2库，请确保正确安装了sam2包")
    exit(1)

# 设置默认路径
DATA_DIR = "data"
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查CUDA可用性
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
if DEVICE == "cpu":
    print("警告: 未检测到CUDA，使用CPU进行推理将会非常慢")

# 加载模型
def load_models(fine_tuned_path=None):
    """加载原始预训练模型和微调模型"""
    models = {}
    
    # 1. 加载原始预训练模型
    try:
        print("正在加载原始预训练模型 (facebook/sam2-hiera-small)...")
        predictor_orig = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        predictor_orig.to(DEVICE)
        models["original"] = predictor_orig
        print("成功加载原始预训练模型")
    except Exception as e:
        print(f"加载原始预训练模型失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 加载微调模型（如果提供了路径）
    if fine_tuned_path and os.path.exists(fine_tuned_path):
        try:
            print(f"正在加载微调模型: {fine_tuned_path}...")
            
            # 从预训练模型复制一个实例
            predictor_ft = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
            
            # 加载微调权重
            state_dict = torch.load(fine_tuned_path, map_location="cpu")
            # 检查是否有嵌套的'model'键
            if isinstance(state_dict, dict) and "model" in state_dict:
                print("从嵌套字典中提取'model'键")
                state_dict = state_dict["model"]
                
            # 加载权重
            predictor_ft.model.load_state_dict(state_dict, strict=False)
            predictor_ft.to(DEVICE)
            models["fine_tuned"] = predictor_ft
            print("成功加载微调模型")
        except Exception as e:
            print(f"加载微调模型失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("未提供微调模型路径或文件不存在，将创建虚拟微调模型")
        # 创建虚拟微调模型
        models["fine_tuned"] = models.get("original", None)
    
    return models

# 加载测试图片
def load_test_images(num_samples=3):
    """从数据目录加载测试图片"""
    image_dir = os.path.join(DATA_DIR, "images/images")
    mask_dir = os.path.join(DATA_DIR, "masks/masks")
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"错误: 找不到图像目录 {image_dir} 或掩码目录 {mask_dir}")
        return [], []
    
    # 获取图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if not image_files:
        print(f"警告: 在 {image_dir} 中没有找到JPEG图像")
        return [], []
    
    # 选择样本
    if num_samples > 0 and num_samples < len(image_files):
        import random
        selected_files = random.sample(image_files, num_samples)
    else:
        selected_files = image_files[:min(num_samples, len(image_files))]
    
    test_images = []
    test_masks = []
    image_paths = []
    
    for img_file in selected_files:
        # 构建图像和掩码的完整路径
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '_mask_combined.jpg'))
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # 读取图像和掩码
            img = cv2.imread(img_path)
            if img is not None:
                img = img[..., ::-1]  # BGR转RGB
                
                mask = cv2.imread(mask_path, 0)  # 读取为灰度
                
                # 调整大小保持比例
                r = min(1024.0 / img.shape[1], 1024.0 / img.shape[0])
                img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
                mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                                 interpolation=cv2.INTER_NEAREST)
                
                test_images.append(img)
                test_masks.append(mask)
                image_paths.append(img_path)
                print(f"已加载测试图像: {img_path}")
    
    return test_images, test_masks, image_paths

# 提取掩码中的点
def sample_points_from_mask(mask, num_points=10):
    """从掩码中采样点坐标"""
    points = []
    coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        print("警告: 掩码为空，无法采样点")
        return np.array([])
    
    indices = np.random.choice(len(coords), min(num_points, len(coords)), replace=False)
    for idx in indices:
        yx = coords[idx]
        points.append([[yx[1], yx[0]]])  # 注意xy坐标顺序
    
    return np.array(points)

# 使用SAM2模型进行预测
def run_inference(predictor, image, input_points):
    """使用SAM2模型进行分割预测"""
    try:
        # 转换为PyTorch兼容格式
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )
        
        # 处理预测掩码
        mask = masks[0, 0].astype(np.uint8) * 255
        return mask, scores[0,0]
    except Exception as e:
        print(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空掩码
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0

# 计算评估指标
def calculate_metrics(gt_mask, pred_mask):
    """计算分割评估指标"""
    # 确保掩码为布尔类型
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    # 计算真阳性、假阳性、假阴性
    true_pos = np.logical_and(gt_mask, pred_mask).sum()
    false_pos = np.logical_and(~gt_mask, pred_mask).sum()
    false_neg = np.logical_and(gt_mask, ~pred_mask).sum()
    
    # 避免除零
    epsilon = 1e-8
    
    # 计算指标
    precision = true_pos / (true_pos + false_pos + epsilon)
    recall = true_pos / (true_pos + false_neg + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = true_pos / (true_pos + false_pos + false_neg + epsilon)
    dice = 2 * true_pos / (2 * true_pos + false_pos + false_neg + epsilon)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice)
    }

# 创建可视化比较
def visualize_comparison(image, mask, pred_orig, pred_ft, metrics_orig, metrics_ft, index, image_name):
    """创建原始模型和微调模型对比的可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 设置标题
    fig.suptitle('SAM2原始模型 vs 微调模型比较', fontsize=16)
    
    # 绘制输入图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('输入图像')
    axes[0, 0].axis('off')
    
    # 绘制真实掩码
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('真实掩码')
    axes[0, 1].axis('off')
    
    # 原始模型预测
    axes[0, 2].imshow(pred_orig, cmap='jet')
    axes[0, 2].set_title(f'原始模型 IoU: {metrics_orig["iou"]:.4f}, Dice: {metrics_orig["dice"]:.4f}')
    axes[0, 2].axis('off')
    
    # 微调模型预测
    axes[1, 0].imshow(pred_ft, cmap='jet')
    axes[1, 0].set_title(f'微调模型 IoU: {metrics_ft["iou"]:.4f}, Dice: {metrics_ft["dice"]:.4f}')
    axes[1, 0].axis('off')
    
    # 创建差异可视化
    diff_map = np.zeros((*mask.shape, 3), dtype=np.uint8)
    gt_binary = mask > 0
    pred_orig_binary = pred_orig > 0
    pred_ft_binary = pred_ft > 0
    
    # 标记差异区域：
    # 绿色: 两个模型都正确的区域
    # 红色: 只有微调模型正确的区域
    # 蓝色: 只有原始模型正确的区域
    # 黄色: 两个模型都错误的区域
    
    # 两个模型都正确（与真实掩码一致的部分）
    both_correct = np.logical_and(
        np.logical_and(gt_binary, pred_orig_binary),
        np.logical_and(gt_binary, pred_ft_binary)
    )
    diff_map[both_correct] = [0, 255, 0]  # 绿色
    
    # 只有微调模型正确
    only_ft_correct = np.logical_and(
        np.logical_and(gt_binary, pred_ft_binary),
        np.logical_not(np.logical_and(gt_binary, pred_orig_binary))
    )
    diff_map[only_ft_correct] = [255, 0, 0]  # 红色
    
    # 只有原始模型正确
    only_orig_correct = np.logical_and(
        np.logical_and(gt_binary, pred_orig_binary),
        np.logical_not(np.logical_and(gt_binary, pred_ft_binary))
    )
    diff_map[only_orig_correct] = [0, 0, 255]  # 蓝色
    
    # 两个模型都有假阳性（预测为前景但实际是背景）
    both_false_positive = np.logical_and(
        np.logical_and(np.logical_not(gt_binary), pred_orig_binary),
        np.logical_and(np.logical_not(gt_binary), pred_ft_binary)
    )
    diff_map[both_false_positive] = [255, 255, 0]  # 黄色
    
    # 绘制差异图
    axes[1, 1].imshow(diff_map)
    axes[1, 1].set_title('预测差异图')
    axes[1, 1].axis('off')
    
    # 绘制指标比较柱状图
    metrics_names = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # 准备数据
    metrics_values_orig = [
        metrics_orig['iou'], 
        metrics_orig['dice'], 
        metrics_orig['precision'], 
        metrics_orig['recall'], 
        metrics_orig['f1']
    ]
    
    metrics_values_ft = [
        metrics_ft['iou'], 
        metrics_ft['dice'], 
        metrics_ft['precision'], 
        metrics_ft['recall'], 
        metrics_ft['f1']
    ]
    
    # 绘制柱状图
    axes[1, 2].bar(x - width/2, metrics_values_orig, width, label='原始模型')
    axes[1, 2].bar(x + width/2, metrics_values_ft, width, label='微调模型')
    
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics_names)
    axes[1, 2].set_ylim(0, 1.1)
    axes[1, 2].set_title('指标比较')
    axes[1, 2].legend()
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 生成文件名 (使用图像名而不仅是索引)
    base_name = os.path.basename(image_name)
    filename = os.path.join(OUTPUT_DIR, f'comparison_{base_name}_{index}.png')
    plt.savefig(filename, dpi=200)
    plt.close()
    
    print(f"已保存比较可视化: {filename}")
    return filename

# 保存评估结果到CSV
def save_metrics_to_csv(all_metrics):
    """保存评估指标到CSV文件"""
    df = pd.DataFrame(all_metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'metrics_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"已保存评估指标到: {csv_path}")
    return csv_path

# 创建汇总对比可视化
def create_summary_visualization(all_metrics, output_dir=OUTPUT_DIR):
    """创建模型对比的汇总可视化"""
    
    if not all_metrics:
        print("没有指标数据，无法创建汇总可视化")
        return
    
    # 计算平均指标
    metrics_keys = ["iou", "dice", "precision", "recall", "f1"]
    
    # 提取数据
    orig_metrics = {}
    ft_metrics = {}
    diff_metrics = {}
    
    for key in metrics_keys:
        orig_values = [m[f"orig_{key}"] for m in all_metrics]
        ft_values = [m[f"ft_{key}"] for m in all_metrics]
        
        orig_metrics[key] = sum(orig_values) / len(orig_values)
        ft_metrics[key] = sum(ft_values) / len(ft_values)
        diff_metrics[key] = ft_metrics[key] - orig_metrics[key]
    
    # 创建汇总图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # 平均指标对比柱状图
    metrics_names = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # 原始和微调模型的平均指标
    axes[0].bar(x - width/2, [orig_metrics[k] for k in metrics_keys], width, label='原始模型')
    axes[0].bar(x + width/2, [ft_metrics[k] for k in metrics_keys], width, label='微调模型')
    
    # 添加改进百分比标签
    for i, key in enumerate(metrics_keys):
        improvement = diff_metrics[key]
        pct_improvement = (improvement / max(orig_metrics[key], 1e-5)) * 100
        
        if improvement > 0:
            label = f"+{improvement:.4f}\n({pct_improvement:.1f}%)"
            color = 'green'
        else:
            label = f"{improvement:.4f}\n({pct_improvement:.1f}%)"
            color = 'red'
        
        axes[0].text(i, max(orig_metrics[key], ft_metrics[key]) + 0.05, 
                     label, ha='center', va='bottom', color=color, fontweight='bold')
    
    axes[0].set_ylabel('平均指标值')
    axes[0].set_title('原始模型 vs 微调模型 - 平均指标对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.2)
    
    # 样本级别的指标差异箱线图
    diff_data = {key: [m[f"ft_{key}"] - m[f"orig_{key}"] for m in all_metrics] for key in metrics_keys}
    axes[1].boxplot([diff_data[key] for key in metrics_keys], labels=metrics_names)
    axes[1].set_ylabel('指标改进程度')
    axes[1].set_title('微调模型相对于原始模型的指标改进分布')
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # 添加改进值
    for i, key in enumerate(metrics_keys):
        avg_diff = diff_metrics[key]
        color = 'green' if avg_diff > 0 else 'red'
        axes[1].text(i+1, avg_diff, f"{avg_diff:.4f}", ha='center', va='bottom', 
                     color=color, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f'summary_comparison_{timestamp}.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"已保存汇总对比可视化到: {out_path}")
    return out_path

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SAM2模型对比脚本")
    parser.add_argument("--samples", type=int, default=3, help="要处理的样本数量")
    parser.add_argument("--finetuned", type=str, default="fine_tuned_sam2_3000.torch", 
                        help="微调模型的路径")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="数据目录路径")
    parser.add_argument("--output-dir", type=str, default="comparison_results", 
                        help="输出目录路径")
    args = parser.parse_args()
    
    # 更新全局变量
    global DATA_DIR, OUTPUT_DIR
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("SAM2 原始模型与微调模型对比")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {DEVICE}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)
    
    # 步骤1: 加载模型
    models = load_models(args.finetuned)
    
    if "original" not in models:
        print("错误: 无法加载原始预训练模型，退出")
        return
        
    if "fine_tuned" not in models:
        print("错误: 无法加载微调模型，退出")
        return
    
    # 步骤2: 加载测试图像
    test_images, test_masks, image_paths = load_test_images(num_samples=args.samples)
    
    if len(test_images) == 0:
        print("错误: 无法加载测试图像")
        return
    
    # 步骤3: 对每个测试样本进行评估
    all_metrics = []
    
    for i, (image, mask, img_path) in enumerate(zip(test_images, test_masks, image_paths)):
        print(f"\n处理测试样本 {i+1}/{len(test_images)}")
        
        # 从掩码中提取点
        input_points = sample_points_from_mask(mask, num_points=5)
        
        if len(input_points) == 0:
            print(f"警告: 样本 {i+1} 无法生成输入点，跳过")
            continue
        
        # 使用原始模型进行预测
        print("使用原始预训练模型进行推理...")
        pred_orig, score_orig = run_inference(models["original"], image, input_points)
        
        # 使用微调模型进行预测
        print("使用微调模型进行推理...")
        pred_ft, score_ft = run_inference(models["fine_tuned"], image, input_points)
        
        # 计算评估指标
        metrics_orig = calculate_metrics(mask > 0, pred_orig > 0)
        metrics_ft = calculate_metrics(mask > 0, pred_ft > 0)
        
        # 创建比较可视化
        vis_path = visualize_comparison(
            image, mask, pred_orig, pred_ft,
            metrics_orig, metrics_ft, i, img_path
        )
        
        # 保存指标
        sample_metrics = {
            "sample_id": i+1,
            "image_path": img_path,
            "orig_score": float(score_orig),
            "ft_score": float(score_ft),
            "score_diff": float(score_ft) - float(score_orig),
            "orig_iou": metrics_orig["iou"],
            "ft_iou": metrics_ft["iou"],
            "iou_diff": metrics_ft["iou"] - metrics_orig["iou"],
            "orig_dice": metrics_orig["dice"],
            "ft_dice": metrics_ft["dice"],
            "dice_diff": metrics_ft["dice"] - metrics_orig["dice"],
            "orig_precision": metrics_orig["precision"],
            "ft_precision": metrics_ft["precision"],
            "precision_diff": metrics_ft["precision"] - metrics_orig["precision"],
            "orig_recall": metrics_orig["recall"],
            "ft_recall": metrics_ft["recall"],
            "recall_diff": metrics_ft["recall"] - metrics_orig["recall"],
            "orig_f1": metrics_orig["f1"],
            "ft_f1": metrics_ft["f1"],
            "f1_diff": metrics_ft["f1"] - metrics_orig["f1"],
            "vis_path": vis_path
        }
        all_metrics.append(sample_metrics)
    
    # 步骤4: 保存评估结果
    if all_metrics:
        metrics_csv = save_metrics_to_csv(all_metrics)
        
        # 创建汇总对比可视化
        summary_vis = create_summary_visualization(all_metrics)
        
        # 计算平均指标差异
        avg_iou_diff = sum(m["iou_diff"] for m in all_metrics) / len(all_metrics)
        avg_dice_diff = sum(m["dice_diff"] for m in all_metrics) / len(all_metrics)
        
        print("\n评估摘要:")
        print(f"样本数量: {len(all_metrics)}")
        print(f"平均IoU提升: {avg_iou_diff:.4f}")
        print(f"平均Dice提升: {avg_dice_diff:.4f}")
        print(f"详细评估结果已保存到: {metrics_csv}")
        if summary_vis:
            print(f"汇总可视化已保存到: {summary_vis}")
    else:
        print("警告: 没有完成任何样本的评估")

if __name__ == "__main__":
    main()