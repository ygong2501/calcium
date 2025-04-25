"""
SAM2 Model Comparison Tool

This script compares the segmentation performance of the fine-tuned SAM2 model
with the original SAM2 model from HuggingFace (zero-shot). It generates side-by-side
visualizations of the segmentation results and calculates evaluation metrics.

Usage:
    python model_compare.py --samples 3 --output results
"""

import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Import necessary SAM2 modules
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load parameters from YAML and ensure correct types
def load_parameters_from_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config['model']
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        raise e

# Helper function for ensuring numeric types in config
def ensure_numeric_types(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                ensure_numeric_types(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        ensure_numeric_types(item)
                    elif isinstance(item, str) and item.isdigit():
                        # Convert string numbers to integers
                        value[i] = int(item)
            elif isinstance(value, str):
                # Try to convert strings to numeric values
                if value.isdigit():
                    config[key] = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                    # Handle floating point numbers
                    config[key] = float(value)

# Calculate evaluation metrics
def calculate_metrics(gt_mask, pred_mask):
    # Ensure masks are boolean type
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    # Flatten masks for pixel-wise metrics
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    # Calculate true positives, false positives, false negatives
    true_positive = np.logical_and(gt_mask, pred_mask).sum()
    false_positive = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    false_negative = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
    true_negative = np.logical_and(np.logical_not(gt_mask), np.logical_not(pred_mask)).sum()

    # Avoid division by zero
    epsilon = 1e-6

    # Calculate metrics
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative + epsilon)
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou
    }

# Load and prepare image and mask
def read_image(image_path, mask_path=None):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to read image {image_path}")
            return None, None

        img = img[..., ::-1]  # Convert BGR to RGB
        
        # Check if mask path is provided
        if mask_path:
            mask = cv2.imread(mask_path, 0)  # Read mask as grayscale
            if mask is None:
                print(f"Error: Unable to read mask {mask_path}")
                return None, None
        else:
            # Create a dummy mask (all zeros)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Resize to maintain aspect ratio with max dimension of 1024
        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                          interpolation=cv2.INTER_NEAREST)

        return img, mask
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Sample points inside the mask
def get_points(mask, num_points):
    try:
        points = []
        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            print("Warning: Mask is empty, cannot sample points")
            return np.array([])

        for i in range(num_points):
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])  # Note xy coordinate order

        return np.array(points)
    except Exception as e:
        print(f"Error generating points: {e}")
        return np.array([])

# Load the fine-tuned model
def load_fine_tuned_model(model_weights, model_cfg):
    try:
        print(f"Loading fine-tuned model: {model_weights}")
        
        # First load the configuration
        config = load_parameters_from_yaml(model_cfg)
        
        # Import the necessary functions
        from load_sam2_direct import build_sam2_model_direct
        
        # Build the model
        model = build_sam2_model_direct(config)
        
        # Load the weights
        state_dict = torch.load(model_weights, map_location="cpu")
        
        # Check if state dict has 'model' key
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Create a predictor
        predictor = SAM2ImagePredictor(model)
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor.model = predictor.model.to(device)
        
        print(f"Fine-tuned model loaded successfully and moved to {device}")
        return predictor
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load the original model from HuggingFace
def load_original_model():
    try:
        print("Loading original SAM2 model from HuggingFace...")
        
        # Create original model predictor from HuggingFace
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-small",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("Original model loaded successfully")
        return predictor
    except Exception as e:
        print(f"Error loading original model from HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            # Fallback to direct loading
            print("Falling back to direct model creation...")
            
            # Load configuration and build model manually
            from load_sam2_direct import build_sam2_model_direct, load_parameters_from_yaml
            
            # Fix the YAML configuration file if needed
            from load_sam2_direct import fix_yaml
            fixed_yaml_path = fix_yaml("sam2_hiera_s.yaml")
            print(f"Using fixed configuration: {fixed_yaml_path}")
            
            config = load_parameters_from_yaml(fixed_yaml_path)
            model = build_sam2_model_direct(config)
            
            # Create predictor
            predictor = SAM2ImagePredictor(model)
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor.model = predictor.model.to(device)
            
            print(f"Original model built manually and moved to {device}")
            return predictor
        except Exception as e2:
            print(f"Error with fallback loading: {e2}")
            traceback.print_exc()
            return None

# Process image with model and return masks and metrics
def process_image(predictor, image, mask, input_points, model_name="Model"):
    try:
        # Perform inference
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )
        
        print(f"{model_name} prediction completed with {len(masks)} masks")
        
        # Process predicted masks
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0]
        sorted_indices = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_indices]
        
        # Create binary ground truth mask
        gt_binary_mask = (mask > 0).astype(np.uint8)
        
        # Initialize segmentation map
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        
        # Combine masks for final segmentation
        all_metrics = []
        for i in range(sorted_masks.shape[0]):
            mask_i = sorted_masks[i]
            
            # Skip if overlapping too much with existing masks
            if (mask_i * occupancy_mask).sum() / (mask_i.sum() + 1e-10) > 0.15:
                continue
            
            mask_bool = mask_i.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = i + 1
            occupancy_mask[mask_bool] = True
            
            # Calculate metrics for this mask
            metrics = calculate_metrics(gt_binary_mask, mask_i > 0.5)
            metrics["mask_id"] = i + 1
            metrics["score"] = float(np_scores[sorted_indices[i]])
            metrics["model"] = model_name
            all_metrics.append(metrics)
        
        # Calculate metrics for final segmentation
        final_seg_binary = (seg_map > 0).astype(np.uint8)
        final_metrics = calculate_metrics(gt_binary_mask, final_seg_binary)
        
        # Add combined metrics
        final_metrics["mask_id"] = "combined"
        final_metrics["score"] = float('nan')
        final_metrics["model"] = model_name
        all_metrics.append(final_metrics)
        
        print(f"{model_name} Combined mask metrics: IoU={final_metrics['iou']:.4f}, Dice={final_metrics['dice']:.4f}")
        
        return all_metrics, seg_map
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], None

# Create side-by-side visualization
def create_comparison_visualization(image, mask, pred_ft, pred_orig, metrics_ft, metrics_orig, 
                                    output_path, sample_name):
    try:
        # Extract combined metrics
        combined_metrics_ft = None
        combined_metrics_orig = None
        
        for m in metrics_ft:
            if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                combined_metrics_ft = m
                break
        
        for m in metrics_orig:
            if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                combined_metrics_orig = m
                break
        
        # If metrics not found, use defaults
        if not combined_metrics_ft:
            combined_metrics_ft = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        if not combined_metrics_orig:
            combined_metrics_orig = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Calculate improvement percentages
        improvements = {}
        for metric in ['iou', 'dice', 'precision', 'recall', 'f1']:
            orig_val = combined_metrics_orig.get(metric, 0)
            ft_val = combined_metrics_ft.get(metric, 0)
            
            # Avoid division by zero
            if abs(orig_val) < 0.0001:
                if ft_val > 0:
                    improvements[metric] = "∞"  # Infinite improvement
                else:
                    improvements[metric] = "0"  # No improvement
            else:
                improvements[metric] = f"{(ft_val - orig_val) / orig_val * 100:.1f}%"
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SAM2 Model Comparison: Fine-tuned vs Original - {sample_name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # Fine-tuned model prediction
        axes[0, 2].imshow(pred_ft, cmap='jet')
        axes[0, 2].set_title(f'Fine-tuned Prediction\nIoU: {combined_metrics_ft.get("iou", 0):.4f}, Dice: {combined_metrics_ft.get("dice", 0):.4f}')
        axes[0, 2].axis('off')
        
        # Original model prediction
        axes[1, 0].imshow(pred_orig, cmap='jet')
        axes[1, 0].set_title(f'Original Prediction\nIoU: {combined_metrics_orig.get("iou", 0):.4f}, Dice: {combined_metrics_orig.get("dice", 0):.4f}')
        axes[1, 0].axis('off')
        
        # Overlap visualizations
        gt_binary = (mask > 0).astype(bool)
        pred_binary_ft = (pred_ft > 0).astype(bool)
        pred_binary_orig = (pred_orig > 0).astype(bool)
        
        # Fine-tuned overlap
        overlap_ft = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        overlap_ft[np.logical_and(gt_binary, pred_binary_ft)] = [0, 255, 0]  # True positive (green)
        overlap_ft[np.logical_and(~gt_binary, pred_binary_ft)] = [255, 0, 0]  # False positive (red)
        overlap_ft[np.logical_and(gt_binary, ~pred_binary_ft)] = [0, 0, 255]  # False negative (blue)
        
        axes[1, 1].imshow(overlap_ft)
        axes[1, 1].set_title('Fine-tuned Overlap')
        axes[1, 1].axis('off')
        
        # Original overlap
        overlap_orig = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        overlap_orig[np.logical_and(gt_binary, pred_binary_orig)] = [0, 255, 0]  # True positive (green)
        overlap_orig[np.logical_and(~gt_binary, pred_binary_orig)] = [255, 0, 0]  # False positive (red)
        overlap_orig[np.logical_and(gt_binary, ~pred_binary_orig)] = [0, 0, 255]  # False negative (blue)
        
        axes[1, 2].imshow(overlap_orig)
        axes[1, 2].set_title('Original Overlap')
        axes[1, 2].axis('off')
        
        # Add metrics table as text
        metrics_text = (
            f"Metric       Fine-tuned     Original      Improvement\n"
            f"----------------------------------------------\n"
            f"IoU          {combined_metrics_ft.get('iou', 0):.4f}        {combined_metrics_orig.get('iou', 0):.4f}        {improvements['iou']}\n"
            f"Dice         {combined_metrics_ft.get('dice', 0):.4f}        {combined_metrics_orig.get('dice', 0):.4f}        {improvements['dice']}\n"
            f"Precision    {combined_metrics_ft.get('precision', 0):.4f}        {combined_metrics_orig.get('precision', 0):.4f}        {improvements['precision']}\n"
            f"Recall       {combined_metrics_ft.get('recall', 0):.4f}        {combined_metrics_orig.get('recall', 0):.4f}        {improvements['recall']}\n"
            f"F1           {combined_metrics_ft.get('f1', 0):.4f}        {combined_metrics_orig.get('f1', 0):.4f}        {improvements['f1']}"
        )
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='g', label='True Positive (Correct)'),
            plt.Rectangle((0, 0), 1, 1, color='r', label='False Positive (Over-segmentation)'),
            plt.Rectangle((0, 0), 1, 1, color='b', label='False Negative (Under-segmentation)')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=3)
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save figure
        plt.savefig(output_path, dpi=200)
        plt.close()
        
        print(f"Comparison visualization saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

# Create a summary visualization with multiple samples
def create_summary_visualization(sample_results, output_dir):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Count number of samples
        num_samples = len(sample_results)
        if num_samples == 0:
            return None
        
        # Calculate average metrics
        avg_metrics_ft = {metric: 0.0 for metric in ['iou', 'dice', 'precision', 'recall', 'f1']}
        avg_metrics_orig = {metric: 0.0 for metric in ['iou', 'dice', 'precision', 'recall', 'f1']}
        
        for sample in sample_results:
            for metric in avg_metrics_ft:
                avg_metrics_ft[metric] += sample['metrics_ft'].get(metric, 0)
                avg_metrics_orig[metric] += sample['metrics_orig'].get(metric, 0)
        
        # Calculate averages
        for metric in avg_metrics_ft:
            avg_metrics_ft[metric] /= num_samples
            avg_metrics_orig[metric] /= num_samples
        
        # Calculate improvements
        improvements = {}
        for metric in avg_metrics_ft:
            orig_val = avg_metrics_orig[metric]
            ft_val = avg_metrics_ft[metric]
            
            if abs(orig_val) < 0.0001:
                if ft_val > 0:
                    improvements[metric] = float('inf')  # Infinite improvement
                else:
                    improvements[metric] = 0.0  # No improvement
            else:
                improvements[metric] = (ft_val - orig_val) / orig_val * 100
        
        # Create the figure
        fig = plt.figure(figsize=(12, 10 + num_samples * 4))
        grid = plt.GridSpec(num_samples + 2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Add title
        fig.suptitle('SAM2 Model Comparison: Fine-tuned vs Original Model', fontsize=16, y=0.98)
        
        # Add sample visualizations
        for i, sample in enumerate(sample_results):
            # Display original image
            ax_img = fig.add_subplot(grid[i, 0])
            ax_img.imshow(sample['image'])
            ax_img.set_title(f'Sample {i+1}')
            ax_img.axis('off')
            
            # Display fine-tuned prediction
            ax_ft = fig.add_subplot(grid[i, 1])
            ax_ft.imshow(sample['pred_ft'], cmap='jet')
            ax_ft.set_title(f'Fine-tuned (IoU: {sample["metrics_ft"].get("iou", 0):.4f})')
            ax_ft.axis('off')
            
            # Display original prediction
            ax_orig = fig.add_subplot(grid[i, 2])
            ax_orig.imshow(sample['pred_orig'], cmap='jet')
            ax_orig.set_title(f'Original (IoU: {sample["metrics_orig"].get("iou", 0):.4f})')
            ax_orig.axis('off')
        
        # Add metrics bar chart
        ax_metrics = fig.add_subplot(grid[num_samples:num_samples+2, :])
        metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create grouped bar chart
        rects1 = ax_metrics.bar(x - width/2, 
                               [avg_metrics_ft['iou'], avg_metrics_ft['dice'], avg_metrics_ft['precision'], 
                                avg_metrics_ft['recall'], avg_metrics_ft['f1']], 
                               width, label='Fine-tuned')
        
        rects2 = ax_metrics.bar(x + width/2, 
                               [avg_metrics_orig['iou'], avg_metrics_orig['dice'], avg_metrics_orig['precision'], 
                                avg_metrics_orig['recall'], avg_metrics_orig['f1']], 
                               width, label='Original')
        
        # Add labels and title
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Average Performance Metrics')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(metrics)
        ax_metrics.legend()
        
        # Add improvement percentages above bars
        for i, (metric, rect1, rect2) in enumerate(zip(metrics, rects1, rects2)):
            height1 = rect1.get_height()
            height2 = rect2.get_height()
            
            metric_lower = metric.lower()
            if improvements[metric_lower] == float('inf'):
                impr_text = '∞%'
            else:
                impr_text = f"{improvements[metric_lower]:+.1f}%"
            
            # Add annotation above the taller bar
            if height1 >= height2:
                ax_metrics.annotate(impr_text,
                                   xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                                   xytext=(0, 5),  # 5 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
            else:
                ax_metrics.annotate(impr_text,
                                   xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                                   xytext=(0, 5),  # 5 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save summary visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.png")
        plt.savefig(summary_path, dpi=200)
        plt.close()
        
        print(f"Summary visualization saved to {summary_path}")
        
        # Create summary CSV
        summary_data = {
            'Metric': metrics,
            'Fine-tuned': [avg_metrics_ft['iou'], avg_metrics_ft['dice'], avg_metrics_ft['precision'], 
                          avg_metrics_ft['recall'], avg_metrics_ft['f1']],
            'Original': [avg_metrics_orig['iou'], avg_metrics_orig['dice'], avg_metrics_orig['precision'], 
                        avg_metrics_orig['recall'], avg_metrics_orig['f1']],
            'Difference': [(avg_metrics_ft[m.lower()] - avg_metrics_orig[m.lower()]) for m in metrics],
            'Improvement (%)': [improvements[m.lower()] if improvements[m.lower()] != float('inf') else float('nan') 
                              for m in metrics]
        }
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, f"model_comparison_metrics_{timestamp}.csv")
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        print(f"Summary metrics saved to {csv_path}")
        return summary_path, csv_path
    except Exception as e:
        print(f"Error creating summary visualization: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Main function
def main(args):
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=== SAM2 Model Comparison ===")
    print(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Samples: {args.samples}")
    print(f"Output directory: {args.output}")
    print(f"Fine-tuned model: {args.finetuned}")
    print("=" * 30)
    
    # Load the models
    predictor_ft = load_fine_tuned_model(args.finetuned, args.config)
    predictor_orig = load_original_model()
    
    if predictor_ft is None:
        print("Error: Failed to load fine-tuned model. Exiting.")
        return
    
    if predictor_orig is None:
        print("Error: Failed to load original model. Exiting.")
        return
    
    # Verify models are different (quick test)
    print("Verifying models are different...")
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_point = np.array([[[50, 50]]])
    
    with torch.no_grad():
        predictor_ft.set_image(test_img)
        predictor_orig.set_image(test_img)
        
        masks_ft, _, _ = predictor_ft.predict(
            point_coords=test_point,
            point_labels=np.ones([test_point.shape[0], 1])
        )
        
        masks_orig, _, _ = predictor_orig.predict(
            point_coords=test_point,
            point_labels=np.ones([test_point.shape[0], 1])
        )
    
    if np.array_equal(masks_ft[0,0], masks_orig[0,0]):
        print("WARNING: Models produce identical outputs on test image!")
    else:
        print("Verification successful: Models produce different outputs")
    
    # Load synthetic data
    print("Loading test data...")
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images/images")
    masks_dir = os.path.join(data_dir, "masks/masks")
    
    # Find all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Randomly select samples
    if args.samples > 0 and args.samples < len(image_files):
        selected_files = random.sample(image_files, args.samples)
    else:
        selected_files = image_files
    
    print(f"Selected {len(selected_files)} samples for evaluation")
    
    # Process each sample
    sample_results = []
    progress_bar = tqdm(selected_files)
    
    for i, image_file in enumerate(progress_bar):
        progress_bar.set_description(f"Processing sample {i+1}/{len(selected_files)}")
        
        # Determine mask file (use the image filename as a guide)
        mask_file = image_file.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(masks_dir, mask_file)):
            # Try CSV lookup if needed here
            print(f"Warning: Mask not found for {image_file}")
            continue
        
        # Load image and mask
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        image, mask = read_image(image_path, mask_path)
        
        if image is None or mask is None:
            print(f"Error: Could not load image or mask for {image_file}")
            continue
        
        # Sample points from mask (using 20 points for reliable segmentation)
        input_points = get_points(mask, 20)
        if len(input_points) == 0:
            print(f"Error: Could not generate points for {image_file}")
            continue
        
        # Process with fine-tuned model
        print(f"Processing with fine-tuned model: {image_file}")
        metrics_ft, pred_ft = process_image(
            predictor_ft, 
            image, 
            mask, 
            input_points, 
            model_name="Fine-tuned"
        )
        
        # Clear CUDA cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process with original model
        print(f"Processing with original model: {image_file}")
        metrics_orig, pred_orig = process_image(
            predictor_orig, 
            image, 
            mask, 
            input_points.copy(),  # Use a copy to avoid any cross-contamination
            model_name="Original"
        )
        
        if pred_ft is None or pred_orig is None:
            print(f"Error: Failed to process {image_file} with one or both models")
            continue
        
        # Create visualization for this sample
        os.makedirs(os.path.join(args.output, "samples"), exist_ok=True)
        vis_path = os.path.join(args.output, "samples", f"comparison_{i+1}_{os.path.basename(image_file)}.png")
        
        create_comparison_visualization(
            image=image,
            mask=mask,
            pred_ft=pred_ft,
            pred_orig=pred_orig,
            metrics_ft=metrics_ft,
            metrics_orig=metrics_orig,
            output_path=vis_path,
            sample_name=f"Sample {i+1}: {os.path.basename(image_file)}"
        )
        
        # Extract combined metrics for summary
        combined_metrics_ft = None
        combined_metrics_orig = None
        
        for m in metrics_ft:
            if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                combined_metrics_ft = m
                break
        
        for m in metrics_orig:
            if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                combined_metrics_orig = m
                break
        
        # Store results for summary
        sample_results.append({
            'image': image,
            'mask': mask,
            'pred_ft': pred_ft,
            'pred_orig': pred_orig,
            'metrics_ft': combined_metrics_ft,
            'metrics_orig': combined_metrics_orig,
            'vis_path': vis_path
        })
    
    # Create summary visualization
    if sample_results:
        summary_path, csv_path = create_summary_visualization(sample_results, args.output)
        print("\nModel comparison completed successfully!")
        print(f"Summary visualization: {summary_path}")
        print(f"Summary metrics: {csv_path}")
    else:
        print("Error: No samples were successfully processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM2 Model Comparison')
    parser.add_argument('--samples', type=int, default=3,
                      help='Number of samples to process (default: 3, use 0 for all)')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--finetuned', type=str, default='fine_tuned_sam2_3000.torch',
                      help='Path to fine-tuned SAM2 model weights')
    parser.add_argument('--config', type=str, default='sam2_hiera_s.yaml',
                      help='Path to SAM2 model configuration file')
    
    args = parser.parse_args()
    main(args)