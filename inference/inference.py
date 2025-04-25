import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime

# Import necessary SAM2 model classes
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Data paths
data_dir = "data"
images_dir = os.path.join(data_dir, "images/images")
masks_dir = os.path.join(data_dir, "masks/masks")
real_data_dir = "denoised_realsample"

# Load test dataset from synthetic data
def load_test_data():
    try:
        # Load train.csv file
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

        # Split dataset
        _, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

        # Prepare test data list
        test_data = []
        for index, row in test_df.iterrows():
            image_name = row['ImageId']
            mask_name = row['MaskId']

            # Add image and corresponding mask paths
            test_data.append({
                "image": os.path.join(images_dir, image_name),
                "annotation": os.path.join(masks_dir, mask_name)
            })

        print(f"Loaded {len(test_data)} synthetic test samples")
        return test_data
    except Exception as e:
        print(f"Error loading synthetic test data: {e}")
        return []

# Load test data from real examples
def load_real_test_data():
    try:
        # Get all image files in the real data directory
        image_files = glob.glob(os.path.join(real_data_dir, "*.jpg"))
        
        # Sort files to ensure consistent processing
        image_files.sort()
        
        # Prepare test data list (no masks for real data)
        test_data = []
        for image_path in image_files:
            # Add image path only
            test_data.append({
                "image": image_path,
                "annotation": None  # No mask for real data
            })

        print(f"Loaded {len(test_data)} real test samples")
        return test_data
    except Exception as e:
        print(f"Error loading real test data: {e}")
        return []

# Read and resize image and mask
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
            # Create a dummy mask for real data (all zeros)
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

# Generate random points for real images (where we don't have ground truth masks)
def generate_grid_points(image, num_points=10):
    try:
        points = []
        height, width = image.shape[:2]
        
        # Calculate grid size for sampling points
        grid_h = int(np.sqrt(num_points))
        grid_w = int(np.ceil(num_points / grid_h))
        
        # Calculate step size
        step_h = height / (grid_h + 1)
        step_w = width / (grid_w + 1)
        
        # Generate grid points
        for i in range(1, grid_h + 1):
            for j in range(1, grid_w + 1):
                if len(points) < num_points:
                    y = int(i * step_h)
                    x = int(j * step_w)
                    points.append([[x, y]])
        
        return np.array(points)
    except Exception as e:
        print(f"Error generating grid points: {e}")
        return np.array([])

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

# Load model parameters from YAML and ensure correct types
def load_parameters_from_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        # Ensure numeric configurations have the correct type
        ensure_numeric_types(config['model'])
        return config['model']
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        raise e

# Ensure numeric values in the configuration have the correct types
def ensure_numeric_types(config):
    """Recursively process the configuration dictionary to ensure numeric parameters have the correct type"""
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

# Helper function to get configuration values
def get_config_value(config, key, default, value_type=None):
    """Get configuration value and ensure its type is correct"""
    value = config.get(key, default)
    if value_type is not None:
        try:
            if value_type == int and isinstance(value, str):
                return int(value)
            if value_type == float and isinstance(value, str):
                return float(value)
            return value_type(value)
        except (ValueError, TypeError):
            print(
                f"Warning: Cannot convert {key}={value} to {value_type.__name__}, using default value {default}")
            return default
    return value

# Build position encoding
def build_position_encoding(config):
    return PositionEmbeddingSine(
        num_pos_feats=get_config_value(config, 'num_pos_feats', 256, int),
        normalize=get_config_value(config, 'normalize', True, bool),
        scale=config.get('scale', None),
        temperature=get_config_value(config, 'temperature', 10000, float)
    )

# Build RoPE attention
def build_rope_attention(config):
    return RoPEAttention(
        rope_theta=get_config_value(config, 'rope_theta', 10000.0, float),
        feat_sizes=config.get('feat_sizes', [32, 32]),
        rope_k_repeat=get_config_value(config, 'rope_k_repeat', True, bool),
        embedding_dim=get_config_value(config, 'embedding_dim', 256, int),
        num_heads=get_config_value(config, 'num_heads', 1, int),
        downsample_rate=get_config_value(config, 'downsample_rate', 1, int),
        dropout=get_config_value(config, 'dropout', 0.1, float),
        kv_in_dim=get_config_value(config, 'kv_in_dim', 64, int)
    )

# Build memory attention layer
def build_memory_attention_layer(config):
    # For self_attention, ensure kv_in_dim is 256
    self_attn_config = config['self_attention'].copy(
    ) if 'self_attention' in config else {}
    # Force set self_attention's kv_in_dim to 256
    self_attn_config['kv_in_dim'] = 256

    self_attention = build_rope_attention(self_attn_config)
    cross_attention = build_rope_attention(config['cross_attention'])

    return MemoryAttentionLayer(
        d_model=get_config_value(config, 'd_model', 256, int),
        self_attention=self_attention,
        cross_attention=cross_attention,
        dim_feedforward=get_config_value(config, 'dim_feedforward', 2048, int),
        dropout=get_config_value(config, 'dropout', 0.1, float),
        activation=config.get('activation', 'relu'),
        pos_enc_at_attn=get_config_value(
            config, 'pos_enc_at_attn', False, bool),
        pos_enc_at_cross_attn_keys=get_config_value(
            config, 'pos_enc_at_cross_attn_keys', True, bool),
        pos_enc_at_cross_attn_queries=get_config_value(
            config, 'pos_enc_at_cross_attn_queries', False, bool)
    )

# Build memory attention
def build_memory_attention(config):
    layer = build_memory_attention_layer(config['layer'])

    return MemoryAttention(
        d_model=get_config_value(config, 'd_model', 256, int),
        layer=layer,
        num_layers=get_config_value(config, 'num_layers', 4, int),
        pos_enc_at_input=get_config_value(
            config, 'pos_enc_at_input', True, bool)
    )

# Build memory encoder
def build_memory_encoder(config):
    position_encoding = build_position_encoding(config['position_encoding'])

    mask_downsampler = MaskDownSampler(
        kernel_size=get_config_value(
            config['mask_downsampler'], 'kernel_size', 3, int),
        stride=get_config_value(config['mask_downsampler'], 'stride', 2, int),
        padding=get_config_value(config['mask_downsampler'], 'padding', 1, int)
    )

    cx_block = CXBlock(
        dim=get_config_value(config['fuser']['layer'], 'dim', 256, int),
        kernel_size=get_config_value(
            config['fuser']['layer'], 'kernel_size', 7, int),
        padding=get_config_value(config['fuser']['layer'], 'padding', 3, int),
        layer_scale_init_value=get_config_value(
            config['fuser']['layer'], 'layer_scale_init_value', 1e-6, float),
        use_dwconv=get_config_value(
            config['fuser']['layer'], 'use_dwconv', True, bool)
    )

    fuser = Fuser(
        layer=cx_block,
        num_layers=get_config_value(config['fuser'], 'num_layers', 2, int)
    )

    return MemoryEncoder(
        out_dim=get_config_value(config, 'out_dim', 64, int),
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser
    )

# Build image encoder
def build_image_encoder(config):
    # Process trunk parameters
    trunk_config = config['trunk']
    embed_dim = get_config_value(trunk_config, 'embed_dim', 96, int)
    num_heads = get_config_value(trunk_config, 'num_heads', 1, int)

    # Ensure stages is a list of integers
    stages = trunk_config.get('stages', [1, 2, 11, 2])
    if isinstance(stages, list):
        stages = [int(s) if isinstance(s, str) else s for s in stages]

    # Ensure global_att_blocks is a list of integers
    global_att_blocks = trunk_config.get('global_att_blocks', [7, 10, 13])
    if isinstance(global_att_blocks, list):
        global_att_blocks = [int(g) if isinstance(
            g, str) else g for g in global_att_blocks]

    # Ensure window_pos_embed_bkg_spatial_size is a list of integers
    window_size = trunk_config.get('window_pos_embed_bkg_spatial_size', [7, 7])
    if isinstance(window_size, list):
        window_size = [int(w) if isinstance(
            w, str) else w for w in window_size]

    trunk = Hiera(
        embed_dim=embed_dim,
        num_heads=num_heads,
        stages=stages,
        global_att_blocks=global_att_blocks,
        window_pos_embed_bkg_spatial_size=window_size
    )

    # Process neck parameters
    neck_config = config['neck']
    position_encoding = build_position_encoding(
        neck_config['position_encoding'])

    d_model = get_config_value(neck_config, 'd_model', 256, int)

    # Ensure backbone_channel_list is a list of integers
    backbone_channels = neck_config.get(
        'backbone_channel_list', [768, 384, 192, 96])
    if isinstance(backbone_channels, list):
        backbone_channels = [int(c) if isinstance(
            c, str) else c for c in backbone_channels]

    # Ensure fpn_top_down_levels is a list of integers
    fpn_levels = neck_config.get('fpn_top_down_levels', [2, 3])
    if isinstance(fpn_levels, list):
        fpn_levels = [int(l) if isinstance(l, str) else l for l in fpn_levels]

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=d_model,
        backbone_channel_list=backbone_channels,
        fpn_top_down_levels=fpn_levels,
        fpn_interp_model=neck_config.get('fpn_interp_model', 'nearest')
    )

    return ImageEncoder(
        trunk=trunk,
        neck=neck,
        scalp=get_config_value(config, 'scalp', 1, int)
    )

# Build SAM2 model directly from configuration
def build_sam2_model_direct(config):
    # Build components
    image_encoder = build_image_encoder(config['image_encoder'])
    memory_attention = build_memory_attention(config['memory_attention'])
    memory_encoder = build_memory_encoder(config['memory_encoder'])

    # Create SAM2Base model
    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        image_size=get_config_value(config, 'image_size', 1024, int),
        num_maskmem=get_config_value(config, 'num_maskmem', 7, int),
        sigmoid_scale_for_mem_enc=get_config_value(
            config, 'sigmoid_scale_for_mem_enc', 20.0, float),
        sigmoid_bias_for_mem_enc=get_config_value(
            config, 'sigmoid_bias_for_mem_enc', -10.0, float),
        use_mask_input_as_output_without_sam=get_config_value(
            config, 'use_mask_input_as_output_without_sam', True, bool),
        directly_add_no_mem_embed=get_config_value(
            config, 'directly_add_no_mem_embed', True, bool),
        use_high_res_features_in_sam=get_config_value(
            config, 'use_high_res_features_in_sam', True, bool),
        multimask_output_in_sam=get_config_value(
            config, 'multimask_output_in_sam', True, bool),
        iou_prediction_use_sigmoid=get_config_value(
            config, 'iou_prediction_use_sigmoid', True, bool),
        use_obj_ptrs_in_encoder=get_config_value(
            config, 'use_obj_ptrs_in_encoder', True, bool),
        add_tpos_enc_to_obj_ptrs=get_config_value(
            config, 'add_tpos_enc_to_obj_ptrs', False, bool),
        only_obj_ptrs_in_the_past_for_eval=get_config_value(
            config, 'only_obj_ptrs_in_the_past_for_eval', True, bool),
        pred_obj_scores=get_config_value(
            config, 'pred_obj_scores', True, bool),
        pred_obj_scores_mlp=get_config_value(
            config, 'pred_obj_scores_mlp', True, bool),
        fixed_no_obj_ptr=get_config_value(
            config, 'fixed_no_obj_ptr', True, bool),
        multimask_output_for_tracking=get_config_value(
            config, 'multimask_output_for_tracking', True, bool),
        use_multimask_token_for_obj_ptr=get_config_value(
            config, 'use_multimask_token_for_obj_ptr', True, bool),
        multimask_min_pt_num=get_config_value(
            config, 'multimask_min_pt_num', 0, int),
        multimask_max_pt_num=get_config_value(
            config, 'multimask_max_pt_num', 1, int),
        use_mlp_for_obj_ptr_proj=get_config_value(
            config, 'use_mlp_for_obj_ptr_proj', True, bool),
        compile_image_encoder=get_config_value(
            config, 'compile_image_encoder', False, bool)
    )

    return model

# Add or update the calculate_metrics function to compute all required metrics
def calculate_metrics(gt_mask, pred_mask):
    """
    Calculate segmentation evaluation metrics between ground truth and predicted mask

    Args:
        gt_mask: Ground truth mask (boolean or binary)
        pred_mask: Predicted mask (boolean or binary)

    Returns:
        Dictionary containing all evaluation metrics
    """
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
    true_negative = np.logical_and(np.logical_not(
        gt_mask), np.logical_not(pred_mask)).sum()

    # Avoid division by zero
    epsilon = 1e-6

    # Calculate metrics
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    dice = 2 * true_positive / \
        (2 * true_positive + false_positive + false_negative + epsilon)
    iou = true_positive / \
        (true_positive + false_positive + false_negative + epsilon)

    # Calculate binary cross-entropy loss
    # Clip prediction to avoid log(0)
    pred_flat_clipped = np.clip(
        pred_flat.astype(float), epsilon, 1.0 - epsilon)
    bce = -(gt_flat * np.log(pred_flat_clipped) + (1 - gt_flat)
            * np.log(1 - pred_flat_clipped)).mean()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "bce": bce
    }

# Load model and initialize predictor
def load_model(model_weights, model_cfg, model_name="Model", strict=True):
    try:
        print(f"Loading SAM2 {model_name} configuration...")
        config = load_parameters_from_yaml(model_cfg)

        print(f"Building SAM2 {model_name} directly from configuration...")
        sam2_model = build_sam2_model_direct(config)

        print(f"Creating SAM2 {model_name} image predictor...")
        predictor = SAM2ImagePredictor(sam2_model)

        print(f"Loading {model_name} weights: {model_weights}")
        
        # Attempt to load weights with proper error handling
        state_dict = torch.load(model_weights, map_location="cpu")
        
        # Check if the state dict contains a 'model' key (common format for pretrained models)
        if isinstance(state_dict, dict) and "model" in state_dict:
            print(f"Found 'model' key in state_dict - extracting for {model_name}")
            state_dict = state_dict["model"]
        
        # Load the state dict with specified strictness
        predictor.model.load_state_dict(state_dict, strict=strict)
        predictor.model.eval()  # Set to evaluation mode

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor.model = predictor.model.to(device)

        print(f"{model_name} loaded and moved to {device}")
        return predictor
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error stack
        return None

# Process a single image and return metrics and visualization
def process_image(predictor, image, mask, input_points, has_ground_truth=True, output_dir="results", model_name="Model"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform inference and predict masks
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )

        print(f"{model_name} prediction completed, obtained {len(masks)} masks")
        
        # Process predicted masks and sort by scores
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0]
        sorted_indices = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_indices]

        # Create binary ground truth mask (1 where mask > 0)
        gt_binary_mask = (mask > 0).astype(np.uint8)

        # Initialize segmentation map and occupancy mask
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

        # Dictionary to store metrics for each mask
        all_metrics = []

        # Combine masks to create final segmentation map
        for i in range(sorted_masks.shape[0]):
            mask_i = sorted_masks[i]
            # Skip if overlapping too much with existing masks
            if (mask_i * occupancy_mask).sum() / (mask_i.sum() + 1e-10) > 0.15:
                continue

            mask_bool = mask_i.astype(bool)
            # Set overlapping areas to False in the mask
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
            occupancy_mask[mask_bool] = True  # Update occupancy mask

            # Calculate metrics for this mask if ground truth is available
            if has_ground_truth:
                metrics = calculate_metrics(gt_binary_mask, mask_i > 0.5)
                metrics["mask_id"] = i + 1
                metrics["score"] = float(np_scores[sorted_indices[i]])
                metrics["model"] = model_name  # Add model name for comparison
                all_metrics.append(metrics)
                print(f"{model_name} Mask {i+1} metrics: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # Calculate metrics for the final segmentation if ground truth is available
        final_metrics = {}
        if has_ground_truth:
            final_seg_binary = (seg_map > 0).astype(np.uint8)
            final_metrics = calculate_metrics(gt_binary_mask, final_seg_binary)
            
            # Convert numpy values to Python native types for better JSON serialization
            for key in final_metrics:
                if isinstance(final_metrics[key], (np.float32, np.float64)):
                    final_metrics[key] = float(final_metrics[key])
            
            # Add final metrics as a row
            final_metrics["mask_id"] = "combined"
            final_metrics["score"] = float('nan')
            final_metrics["model"] = model_name  # Add model name for comparison
            all_metrics.append(final_metrics)
            
            print(f"{model_name} Combined mask metrics: IoU={final_metrics['iou']:.4f}, Dice={final_metrics['dice']:.4f}")

        # Generate individual visualization only if requested
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{model_name}_segmentation_result_{timestamp}.png"
        
        # Visualization: save the results
        plt.figure(figsize=(8, 18 if has_ground_truth else 12))
        
        plt.subplot(3 if has_ground_truth else 2, 1, 1)
        plt.title('Input Image')
        plt.imshow(image)
        plt.axis('on')

        if has_ground_truth:
            plt.subplot(3, 1, 2)
            plt.title('Ground Truth Mask')
            plt.imshow(mask, cmap='gray')
            plt.axis('on')
            
            plt.subplot(3, 1, 3)
            title_text = f'{model_name} Segmentation (IoU: {final_metrics.get("iou", 0):.4f}, DICE: {final_metrics.get("dice", 0):.4f})'
        else:
            plt.subplot(2, 1, 2)
            title_text = f'{model_name} Segmentation'
            
        plt.title(title_text)
        plt.imshow(seg_map, cmap='jet')
        plt.axis('on')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, img_filename))
        plt.close()
        
        # If no metrics were calculated (but should have been), add default metrics
        if has_ground_truth and not all_metrics:
            print(f"WARNING: No metrics calculated for {model_name}, adding default combined metrics")
            default_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "dice": 0.0,
                "iou": 0.0,
                "bce": 0.0,
                "mask_id": "combined",
                "score": float('nan'),
                "model": model_name
            }
            all_metrics.append(default_metrics)
        
        return all_metrics, seg_map, img_filename
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None

# Save metrics to CSV file
def save_metrics_to_csv(all_metrics, image_name, data_mode, output_dir="results"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Add image name and data mode columns
        metrics_df["image_name"] = image_name
        metrics_df["data_mode"] = data_mode
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{data_mode}_metrics_{os.path.basename(image_name)}_{timestamp}.csv"
        
        # Save to CSV
        metrics_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
        print(f"Metrics saved to {os.path.join(output_dir, csv_filename)}")
        
        return csv_filename
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")
        return None

# Combine metrics from multiple runs for summary
def combine_metrics(metric_files, output_dir="results"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize empty list to hold all dataframes
        all_dfs = []
        
        # Read each CSV file and append to list
        for file in metric_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                all_dfs.append(df)
        
        if not all_dfs:
            print("No valid metric files found to combine")
            return None
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Calculate overall statistics (mean, std) grouped by data_mode
        summary = combined_df.groupby(["data_mode", "mask_id"]).agg({
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "dice": ["mean", "std"],
            "iou": ["mean", "std"],
            "bce": ["mean", "std"]
        }).reset_index()
        
        # Save summary to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"metrics_summary_{timestamp}.csv"
        summary.to_csv(os.path.join(output_dir, summary_filename), index=False)
        
        # Also save the combined raw data
        combined_filename = f"metrics_combined_{timestamp}.csv"
        combined_df.to_csv(os.path.join(output_dir, combined_filename), index=False)
        
        print(f"Summary metrics saved to {os.path.join(output_dir, summary_filename)}")
        print(f"Combined metrics saved to {os.path.join(output_dir, combined_filename)}")
        
        return summary_filename, combined_filename
    except Exception as e:
        print(f"Error combining metrics: {e}")
        return None, None

# Create comprehensive visualization of test and real samples with evaluation metrics
def create_comprehensive_visualization(test_images, test_masks, test_predictions_ft, test_predictions_orig, 
                                      test_metrics_ft, test_metrics_orig, test_avg_metrics_ft, test_avg_metrics_orig,
                                      real_images, real_predictions_ft, real_predictions_orig,
                                      output_dir="results", num_samples=3):
    """
    Create a comprehensive visualization showing test and real samples with metrics.
    
    Args:
        test_images: List of test images (numpy arrays)
        test_masks: List of ground truth masks (numpy arrays)
        test_predictions_ft: List of predicted segmentation maps from fine-tuned model (numpy arrays)
        test_predictions_orig: List of predicted segmentation maps from original model (numpy arrays)
        test_metrics_ft: List of dictionaries containing metrics for fine-tuned model
        test_metrics_orig: List of dictionaries containing metrics for original model
        test_avg_metrics_ft: Dictionary containing average metrics for fine-tuned model
        test_avg_metrics_orig: Dictionary containing average metrics for original model
        real_images: List of real sample images (numpy arrays)
        real_predictions_ft: List of predicted segmentation maps from fine-tuned model (numpy arrays)
        real_predictions_orig: List of predicted segmentation maps from original model (numpy arrays)
        output_dir: Directory to save the output visualization
        num_samples: Number of samples to display (default: 3)
    
    Returns:
        Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure we have at most num_samples for each category
        test_images = test_images[:num_samples]
        test_masks = test_masks[:num_samples]
        test_predictions_ft = test_predictions_ft[:num_samples]
        test_predictions_orig = test_predictions_orig[:num_samples]
        test_metrics_ft = test_metrics_ft[:num_samples]
        test_metrics_orig = test_metrics_orig[:num_samples]
        real_images = real_images[:num_samples]
        real_predictions_ft = real_predictions_ft[:num_samples]
        real_predictions_orig = real_predictions_orig[:num_samples]
        
        # Create figure with appropriate size
        fig = plt.figure(figsize=(24, 30))
        
        # Calculate dynamic height ratios based on number of samples
        # Include a small row after the images for the color legend
        height_ratios = [4] * num_samples + [0.8, 3, 3]
        
        # Define the grid layout with additional column for overlap visualization
        grid_spec = fig.add_gridspec(num_samples+3, 8, height_ratios=height_ratios)
        
        # Add title to the entire figure
        fig.suptitle('SAM2 Segmentation Results: Fine-tuned vs Original Model Comparison', fontsize=20, y=0.98)
        
        # Create a list to store the color patches for the legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Ground Truth Mask'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Fine-tuned Model Prediction'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Original Model Prediction'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=12, label='Overlap with Ground Truth')
        ]
        
        # Create color maps for visualization
        cmap_gt = plt.cm.Blues      # Ground truth mask (blue)
        cmap_pred_ft = plt.cm.Reds   # Fine-tuned model prediction (red)
        cmap_pred_orig = plt.cm.Greens  # Original model prediction (green)
        
        # Plot test samples with comparative visualizations
        for i in range(num_samples):
            # Test image
            ax_test_img = fig.add_subplot(grid_spec[i, 0])
            ax_test_img.imshow(test_images[i])
            ax_test_img.set_title(f'Test Image {i+1}')
            ax_test_img.axis('off')
            
            # Test ground truth mask
            ax_test_mask = fig.add_subplot(grid_spec[i, 1])
            ax_test_mask.imshow(test_masks[i], cmap='Blues')
            ax_test_mask.set_title(f'Ground Truth {i+1}')
            ax_test_mask.axis('off')
            
            # Fine-tuned model prediction
            ax_test_pred_ft = fig.add_subplot(grid_spec[i, 2])
            ax_test_pred_ft.imshow(test_predictions_ft[i], cmap='Reds')
            ax_test_pred_ft.set_title(f'Fine-tuned Pred {i+1}')
            ax_test_pred_ft.axis('off')
            
            # Original model prediction
            ax_test_pred_orig = fig.add_subplot(grid_spec[i, 3])
            ax_test_pred_orig.imshow(test_predictions_orig[i], cmap='Greens')
            ax_test_pred_orig.set_title(f'Original Pred {i+1}')
            ax_test_pred_orig.axis('off')
            
            # Overlap visualization for fine-tuned model
            ax_test_overlap_ft = fig.add_subplot(grid_spec[i, 4])
            
            # Create RGB overlap image for fine-tuned model
            overlap_img_ft = np.zeros((*test_masks[i].shape, 3), dtype=np.uint8)
            
            # Convert masks to binary
            gt_binary = (test_masks[i] > 0).astype(bool)
            pred_binary_ft = (test_predictions_ft[i] > 0).astype(bool)
            
            # Blue channel for ground truth
            overlap_img_ft[gt_binary, 2] = 255
            # Red channel for fine-tuned prediction
            overlap_img_ft[pred_binary_ft, 0] = 255
            
            ax_test_overlap_ft.imshow(overlap_img_ft)
            ax_test_overlap_ft.set_title(f'FT Overlap {i+1}')
            ax_test_overlap_ft.axis('off')
            
            # Overlap visualization for original model
            ax_test_overlap_orig = fig.add_subplot(grid_spec[i, 5])
            
            # Create RGB overlap image for original model
            overlap_img_orig = np.zeros((*test_masks[i].shape, 3), dtype=np.uint8)
            
            # Convert masks to binary
            pred_binary_orig = (test_predictions_orig[i] > 0).astype(bool)
            
            # Blue channel for ground truth
            overlap_img_orig[gt_binary, 2] = 255
            # Green channel for original prediction
            overlap_img_orig[pred_binary_orig, 1] = 255
            
            ax_test_overlap_orig.imshow(overlap_img_orig)
            ax_test_overlap_orig.set_title(f'Orig Overlap {i+1}')
            ax_test_overlap_orig.axis('off')
            
            # Real sample image and predictions (if available)
            if i < len(real_images):
                ax_real_img = fig.add_subplot(grid_spec[i, 6])
                ax_real_img.imshow(real_images[i])
                ax_real_img.set_title(f'Real Image {i+1}')
                ax_real_img.axis('off')
                
                # Real sample comparison (split plot)
                ax_real_comp = fig.add_subplot(grid_spec[i, 7])
                
                # Create a side-by-side comparison image
                comp_height = real_predictions_ft[i].shape[0]
                comp_width = real_predictions_ft[i].shape[1]
                comparison = np.zeros((comp_height, comp_width, 3), dtype=np.uint8)
                
                # Left half: Fine-tuned model (red)
                half_width = comp_width // 2
                pred_binary_ft_real = (real_predictions_ft[i] > 0).astype(bool)
                comparison[:, :half_width, 0][pred_binary_ft_real[:, :half_width]] = 255
                
                # Right half: Original model (green)
                pred_binary_orig_real = (real_predictions_orig[i] > 0).astype(bool)
                comparison[:, half_width:, 1][pred_binary_orig_real[:, half_width:]] = 255
                
                # Add a vertical line to separate the two halves
                comparison[:, half_width-1:half_width+1, :] = 255
                
                ax_real_comp.imshow(comparison)
                ax_real_comp.set_title(f'FT | Orig {i+1}')
                ax_real_comp.axis('off')
        
        # Add legend directly below images
        ax_legend = fig.add_subplot(grid_spec[num_samples, :])
        ax_legend.axis('off')
        ax_legend.legend(handles=legend_elements, loc='center', ncol=4, fontsize=12, title="Color Legend")
        
        # Create comparative metrics table
        ax_metrics = fig.add_subplot(grid_spec[num_samples+1, :])
        ax_metrics.axis('tight')
        ax_metrics.axis('off')
        
        # Prepare metrics data for the table - side by side comparison
        metrics_data = []
        metrics_columns = ['Sample', 
                          'FT IoU', 'Orig IoU', 'Δ IoU',
                          'FT Dice', 'Orig Dice', 'Δ Dice', 
                          'FT Prec', 'Orig Prec', 'Δ Prec',
                          'FT Recall', 'Orig Recall', 'Δ Recall']
        
        for i in range(min(num_samples, len(test_metrics_ft), len(test_metrics_orig))):
            # Extract the "combined" metrics if available for both models
            combined_metrics_ft = None
            combined_metrics_orig = None
            
            # Find combined metrics for fine-tuned model
            for m in test_metrics_ft[i]:
                if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                    combined_metrics_ft = m
                    break
            
            # If not found, use default values
            if not combined_metrics_ft:
                combined_metrics_ft = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
            # Find combined metrics for original model
            for m in test_metrics_orig[i]:
                if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                    combined_metrics_orig = m
                    break
            
            # If not found, use default values
            if not combined_metrics_orig:
                combined_metrics_orig = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
            # Calculate differences
            iou_diff = combined_metrics_ft.get('iou', 0) - combined_metrics_orig.get('iou', 0)
            dice_diff = combined_metrics_ft.get('dice', 0) - combined_metrics_orig.get('dice', 0)
            prec_diff = combined_metrics_ft.get('precision', 0) - combined_metrics_orig.get('precision', 0)
            recall_diff = combined_metrics_ft.get('recall', 0) - combined_metrics_orig.get('recall', 0)
            
            # Format differences with arrows
            iou_diff_str = f"↑{iou_diff:.4f}" if iou_diff > 0 else f"↓{abs(iou_diff):.4f}" if iou_diff < 0 else f"{iou_diff:.4f}"
            dice_diff_str = f"↑{dice_diff:.4f}" if dice_diff > 0 else f"↓{abs(dice_diff):.4f}" if dice_diff < 0 else f"{dice_diff:.4f}"
            prec_diff_str = f"↑{prec_diff:.4f}" if prec_diff > 0 else f"↓{abs(prec_diff):.4f}" if prec_diff < 0 else f"{prec_diff:.4f}"
            recall_diff_str = f"↑{recall_diff:.4f}" if recall_diff > 0 else f"↓{abs(recall_diff):.4f}" if recall_diff < 0 else f"{recall_diff:.4f}"
            
            # Add row to metrics data
            metrics_data.append([
                f'Test {i+1}',
                f'{combined_metrics_ft.get("iou", 0):.4f}',
                f'{combined_metrics_orig.get("iou", 0):.4f}',
                iou_diff_str,
                f'{combined_metrics_ft.get("dice", 0):.4f}',
                f'{combined_metrics_orig.get("dice", 0):.4f}',
                dice_diff_str,
                f'{combined_metrics_ft.get("precision", 0):.4f}',
                f'{combined_metrics_orig.get("precision", 0):.4f}',
                prec_diff_str,
                f'{combined_metrics_ft.get("recall", 0):.4f}',
                f'{combined_metrics_orig.get("recall", 0):.4f}',
                recall_diff_str
            ])
        
        metrics_table = ax_metrics.table(
            cellText=metrics_data,
            colLabels=metrics_columns,
            loc='center',
            cellLoc='center'
        )
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(9)
        metrics_table.scale(1, 1.5)
        ax_metrics.set_title('Comparative Metrics: Fine-tuned vs Original Model', fontsize=14, pad=20)
        
        # Add average metrics table
        ax_avg = fig.add_subplot(grid_spec[num_samples+2, :])
        ax_avg.axis('tight')
        ax_avg.axis('off')
        
        # Calculate differences for averages
        avg_iou_diff = test_avg_metrics_ft.get('iou', 0) - test_avg_metrics_orig.get('iou', 0)
        avg_dice_diff = test_avg_metrics_ft.get('dice', 0) - test_avg_metrics_orig.get('dice', 0)
        avg_prec_diff = test_avg_metrics_ft.get('precision', 0) - test_avg_metrics_orig.get('precision', 0)
        avg_recall_diff = test_avg_metrics_ft.get('recall', 0) - test_avg_metrics_orig.get('recall', 0)
        
        # Format average differences with arrows
        avg_iou_diff_str = f"↑{avg_iou_diff:.4f}" if avg_iou_diff > 0 else f"↓{abs(avg_iou_diff):.4f}" if avg_iou_diff < 0 else f"{avg_iou_diff:.4f}"
        avg_dice_diff_str = f"↑{avg_dice_diff:.4f}" if avg_dice_diff > 0 else f"↓{abs(avg_dice_diff):.4f}" if avg_dice_diff < 0 else f"{avg_dice_diff:.4f}"
        avg_prec_diff_str = f"↑{avg_prec_diff:.4f}" if avg_prec_diff > 0 else f"↓{abs(avg_prec_diff):.4f}" if avg_prec_diff < 0 else f"{avg_prec_diff:.4f}"
        avg_recall_diff_str = f"↑{avg_recall_diff:.4f}" if avg_recall_diff > 0 else f"↓{abs(avg_recall_diff):.4f}" if avg_recall_diff < 0 else f"{avg_recall_diff:.4f}"
        
        # Calculate percentage improvements with safeguards
        def calculate_improvement(new_val, old_val):
            if old_val == 0 or abs(old_val) < 0.0001:
                # Avoid division by zero
                if new_val == 0:
                    return 0.0  # No change
                elif new_val > 0:
                    return float('inf')  # Infinite improvement (from 0 to something positive)
                else:
                    return float('-inf')  # Infinite negative (shouldn't happen with metrics)
            else:
                return (new_val - old_val) / abs(old_val) * 100
        
        pct_iou_impr = calculate_improvement(test_avg_metrics_ft.get('iou', 0), test_avg_metrics_orig.get('iou', 0))
        pct_dice_impr = calculate_improvement(test_avg_metrics_ft.get('dice', 0), test_avg_metrics_orig.get('dice', 0))
        pct_prec_impr = calculate_improvement(test_avg_metrics_ft.get('precision', 0), test_avg_metrics_orig.get('precision', 0))
        pct_recall_impr = calculate_improvement(test_avg_metrics_ft.get('recall', 0), test_avg_metrics_orig.get('recall', 0))
        
        # Include percentage improvement in the diff strings
        def format_with_percentage(diff_str, percentage):
            if percentage == 0:
                return diff_str
            elif percentage == float('inf'):
                return f"{diff_str} (∞%)"
            elif percentage == float('-inf'):
                return f"{diff_str} (-∞%)"
            else:
                return f"{diff_str} ({percentage:.1f}%)"
                
        if avg_iou_diff != 0:
            avg_iou_diff_str = format_with_percentage(avg_iou_diff_str, pct_iou_impr)
        if avg_dice_diff != 0:
            avg_dice_diff_str = format_with_percentage(avg_dice_diff_str, pct_dice_impr)
        if avg_prec_diff != 0:
            avg_prec_diff_str = format_with_percentage(avg_prec_diff_str, pct_prec_impr)
        if avg_recall_diff != 0:
            avg_recall_diff_str = format_with_percentage(avg_recall_diff_str, pct_recall_impr)
        
        # Prepare average metrics data
        avg_data = [
            ['All Samples',
             f'{test_avg_metrics_ft.get("iou", 0):.4f}',
             f'{test_avg_metrics_orig.get("iou", 0):.4f}',
             avg_iou_diff_str,
             f'{test_avg_metrics_ft.get("dice", 0):.4f}',
             f'{test_avg_metrics_orig.get("dice", 0):.4f}',
             avg_dice_diff_str,
             f'{test_avg_metrics_ft.get("precision", 0):.4f}',
             f'{test_avg_metrics_orig.get("precision", 0):.4f}',
             avg_prec_diff_str,
             f'{test_avg_metrics_ft.get("recall", 0):.4f}',
             f'{test_avg_metrics_orig.get("recall", 0):.4f}',
             avg_recall_diff_str]
        ]
        
        avg_table = ax_avg.table(
            cellText=avg_data,
            colLabels=metrics_columns,
            loc='center',
            cellLoc='center',
            cellColours=None
        )
        avg_table.auto_set_font_size(False)
        avg_table.set_fontsize(9)
        avg_table.scale(1, 1.5)
        
        # Highlight cells with improvements
        if avg_iou_diff > 0:
            avg_table[(1, 3)].set_facecolor('#d8f3dc')  # Light green for positive change
        elif avg_iou_diff < 0:
            avg_table[(1, 3)].set_facecolor('#ffccd5')  # Light red for negative change
            
        if avg_dice_diff > 0:
            avg_table[(1, 6)].set_facecolor('#d8f3dc')
        elif avg_dice_diff < 0:
            avg_table[(1, 6)].set_facecolor('#ffccd5')
            
        if avg_prec_diff > 0:
            avg_table[(1, 9)].set_facecolor('#d8f3dc')
        elif avg_prec_diff < 0:
            avg_table[(1, 9)].set_facecolor('#ffccd5')
            
        if avg_recall_diff > 0:
            avg_table[(1, 12)].set_facecolor('#d8f3dc')
        elif avg_recall_diff < 0:
            avg_table[(1, 12)].set_facecolor('#ffccd5')
        
        ax_avg.set_title('Average Metrics & Performance Improvement', fontsize=14, pad=20)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_path = os.path.join(output_dir, f"comprehensive_visualization_{timestamp}.png")
        plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Comprehensive visualization saved to {visualization_path}")
        return visualization_path
    
    except Exception as e:
        print(f"Error creating comprehensive visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

# Create individual side-by-side comparison for a sample
def create_side_by_side_comparison(image, mask, pred_ft, pred_orig, metrics_ft, metrics_orig, sample_id, output_dir="results"):
    """
    Create a side-by-side comparison visualization for a single sample.
    
    Args:
        image: Input image
        mask: Ground truth mask
        pred_ft: Fine-tuned model prediction
        pred_orig: Original model prediction
        metrics_ft: Fine-tuned model metrics
        metrics_orig: Original model metrics
        sample_id: Sample identifier
        output_dir: Output directory
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get combined metrics
        combined_metrics_ft = None
        combined_metrics_orig = None
        
        # Find combined metrics for fine-tuned model
        if metrics_ft:
            for m in metrics_ft:
                if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                    combined_metrics_ft = m
                    break
        
        # Find combined metrics for original model
        if metrics_orig:
            for m in metrics_orig:
                if isinstance(m.get("mask_id"), str) and m["mask_id"] == "combined":
                    combined_metrics_orig = m
                    break
        
        # Use default values if metrics not found
        if not combined_metrics_ft:
            combined_metrics_ft = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        if not combined_metrics_orig:
            combined_metrics_orig = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Add title
        fig.suptitle(f'Sample {sample_id}: Fine-tuned vs Original Model Comparison', fontsize=16)
        
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
        
        # Fine-tuned overlap
        gt_binary = (mask > 0).astype(bool)
        pred_binary_ft = (pred_ft > 0).astype(bool)
        
        # Create RGB overlap image for fine-tuned
        overlap_ft = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        
        # True positive (green), False positive (red), False negative (blue)
        overlap_ft[np.logical_and(gt_binary, pred_binary_ft)] = [0, 255, 0]  # True positive
        overlap_ft[np.logical_and(~gt_binary, pred_binary_ft)] = [255, 0, 0]  # False positive
        overlap_ft[np.logical_and(gt_binary, ~pred_binary_ft)] = [0, 0, 255]  # False negative
        
        axes[1, 1].imshow(overlap_ft)
        axes[1, 1].set_title('Fine-tuned Overlap')
        axes[1, 1].axis('off')
        
        # Original overlap
        pred_binary_orig = (pred_orig > 0).astype(bool)
        
        # Create RGB overlap image for original
        overlap_orig = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        
        # True positive (green), False positive (red), False negative (blue)
        overlap_orig[np.logical_and(gt_binary, pred_binary_orig)] = [0, 255, 0]  # True positive
        overlap_orig[np.logical_and(~gt_binary, pred_binary_orig)] = [255, 0, 0]  # False positive
        overlap_orig[np.logical_and(gt_binary, ~pred_binary_orig)] = [0, 0, 255]  # False negative
        
        axes[1, 2].imshow(overlap_orig)
        axes[1, 2].set_title('Original Overlap')
        axes[1, 2].axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='g', label='True Positive'),
            plt.Rectangle((0, 0), 1, 1, color='r', label='False Positive'),
            plt.Rectangle((0, 0), 1, 1, color='b', label='False Negative')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=3)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(output_dir, f"comparison_sample_{sample_id}_{timestamp}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)
        
        print(f"Saved comparison for sample {sample_id} to {fig_path}")
    except Exception as e:
        print(f"Error creating side-by-side comparison: {e}")
        import traceback
        traceback.print_exc()


# Inference function with mode selection and model comparison
def inference(mode="synthetic", num_samples=5, output_dir="results", visualize_together=False, compare_models=True):
    print(f"Starting inference process in {mode} mode...")
    os.makedirs(output_dir, exist_ok=True)

    # Load data based on mode
    if mode == "real":
        test_data = load_real_test_data()
        has_ground_truth = False
    else:  # synthetic mode
        test_data = load_test_data()
        has_ground_truth = True

    if not test_data:
        print(f"Error: Unable to load {mode} test data")
        return

    # Limit the number of samples to process
    if num_samples > 0 and num_samples < len(test_data):
        # Randomly select samples
        sampled_data = random.sample(test_data, num_samples)
    else:
        sampled_data = test_data

    # Load models with error handling
    try:
        # Define model paths - allow overriding from command line args if provided
        FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_3000.torch"
        ORIGINAL_MODEL_WEIGHTS = "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        
        # Import needed SAM2 modules
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Verify fine-tuned model file exists
        if not os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
            print(f"WARNING: Fine-tuned model file {FINE_TUNED_MODEL_WEIGHTS} not found!")
        
        # Load fine-tuned model using our custom loader
        print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_WEIGHTS}...")
        predictor_ft = load_model(FINE_TUNED_MODEL_WEIGHTS, model_cfg, model_name="Fine-tuned")
        
        # Load original model from HuggingFace if comparison is enabled
        predictor_orig = None
        if compare_models:
            try:
                print(f"Loading original SAM2 model from HuggingFace...")
                
                # Try to load original model using the HuggingFace API
                try:
                    # Create original model predictor from HuggingFace
                    predictor_orig = SAM2ImagePredictor.from_pretrained(
                        "facebook/sam2-hiera-small",
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    print("Successfully loaded original model from HuggingFace")
                except Exception as e:
                    print(f"Error loading original model from HuggingFace: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fall back to trying our direct model loading approach
                    print("Falling back to custom direct loader...")
                    try:
                        from load_sam2_direct import build_sam2_model_direct, load_parameters_from_yaml
                        
                        # First, fix the YAML configuration file if needed
                        from load_sam2_direct import fix_yaml
                        fixed_yaml_path = fix_yaml(model_cfg)
                        print(f"Using fixed YAML configuration: {fixed_yaml_path}")
                        
                        # Load configuration and build model directly
                        config = load_parameters_from_yaml(fixed_yaml_path)
                        sam2_model = build_sam2_model_direct(config)
                        
                        # Create predictor with the directly built model
                        predictor_orig = SAM2ImagePredictor(sam2_model)
                        print("Successfully built original model using direct loading")
                        
                        # Move to appropriate device
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        predictor_orig.model = predictor_orig.model.to(device)
                        print(f"Moved original model to {device}")
                    except Exception as e2:
                        print(f"Error with fallback direct loader: {e2}")
                        traceback.print_exc()
                        predictor_orig = None
            except Exception as e:
                print(f"Failed to load original model: {e}")
                traceback.print_exc()
                predictor_orig = None
            
            # Verify models are different
            if predictor_ft is not None and predictor_orig is not None:
                # Instead of comparing state dictionaries which might have different structures,
                # we'll run a quick inference test to see if they produce different outputs
                print("Running comparison test between models...")
                
                # Create a test image
                import numpy as np
                test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                test_point = np.array([[[50, 50]]])
                
                # Run both models on the same input and compare
                try:
                    with torch.no_grad():
                        # Set same image
                        predictor_ft.set_image(test_img)
                        predictor_orig.set_image(test_img)
                        
                        # Predict with both models
                        ft_masks, _, _ = predictor_ft.predict(
                            point_coords=test_point,
                            point_labels=np.ones([test_point.shape[0], 1])
                        )
                        
                        orig_masks, _, _ = predictor_orig.predict(
                            point_coords=test_point,
                            point_labels=np.ones([test_point.shape[0], 1])
                        )
                        
                        # Check if masks are different
                        if np.array_equal(ft_masks[0,0], orig_masks[0,0]):
                            print("WARNING: Models produce identical results on test image!")
                        else:
                            print("Models produce different results - verification passed.")
                except Exception as comp_e:
                    print(f"Error during model comparison: {comp_e}")
                    traceback.print_exc()
                
        if predictor_ft is None:
            print("Error: Unable to load fine-tuned model")
            return
        
        if compare_models and predictor_orig is None:
            print("Warning: Unable to load original model, continuing with fine-tuned model only")
            compare_models = False
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Process each sample
    all_csv_files_ft = []
    all_csv_files_orig = []
    
    # Store data for comprehensive visualization
    all_images = []
    all_masks = []
    all_predictions_ft = []
    all_metrics_list_ft = []
    all_predictions_orig = []
    all_metrics_list_orig = []
    
    for i, entry in enumerate(sampled_data):
        print(f"\nProcessing sample {i+1}/{len(sampled_data)}")
        
        image_path = entry['image']
        mask_path = entry['annotation']
        
        print(f"Image path: {image_path}")
        if mask_path:
            print(f"Mask path: {mask_path}")

        # Load the image and mask
        image, mask = read_image(image_path, mask_path)
        if image is None:
            print("Unable to load image")
            continue

        # Generate points for input based on mode
        if mode == "real" or mask_path is None:
            # For real data, generate grid points
            input_points = generate_grid_points(image, num_points=30)
        else:
            # For synthetic data, sample points from the mask
            input_points = get_points(mask, 30)
            
        if len(input_points) == 0:
            print("Unable to generate input points")
            continue

        # Process the image with fine-tuned model
        metrics_ft, seg_map_ft, img_filename_ft = process_image(
            predictor_ft, 
            image, 
            mask, 
            input_points, 
            has_ground_truth=has_ground_truth,
            output_dir=output_dir,
            model_name="Fine-tuned"
        )
        
        # Process the image with original model if comparison is enabled
        metrics_orig, seg_map_orig, img_filename_orig = None, None, None
        if compare_models and predictor_orig:
            try:
                print(f"Running prediction with original model on sample {i+1}...")
                
                # Create a deep copy of the input points to ensure no cross-contamination
                orig_input_points = input_points.copy()
                
                # First, clear the CUDA cache to avoid any memory sharing between models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process with original model separately
                metrics_orig, seg_map_orig, img_filename_orig = process_image(
                    predictor_orig, 
                    image.copy(), # Use a copy of the image to avoid any shared processing
                    mask.copy(),  # Use a copy of the mask as well
                    orig_input_points, 
                    has_ground_truth=has_ground_truth,
                    output_dir=output_dir,
                    model_name="Original"
                )
                
                # Verify results are different
                if seg_map_ft is not None and seg_map_orig is not None:
                    if np.array_equal(seg_map_ft, seg_map_orig):
                        print("WARNING: Fine-tuned and original model produced IDENTICAL segmentation maps!")
                        print("This strongly suggests there's an issue with model loading or inference.")
                    else:
                        # Calculate overlap percentage to quantify difference
                        ft_positive = (seg_map_ft > 0).astype(np.uint8)
                        orig_positive = (seg_map_orig > 0).astype(np.uint8)
                        
                        intersection = np.logical_and(ft_positive, orig_positive).sum()
                        union = np.logical_or(ft_positive, orig_positive).sum()
                        
                        if union > 0:
                            overlap = intersection / union
                            print(f"Models produced different results: {overlap*100:.2f}% overlap")
                        else:
                            print("Both models produced empty segmentation maps")
                
                # Generate side-by-side comparison for this sample
                if has_ground_truth and seg_map_ft is not None and seg_map_orig is not None:
                    create_side_by_side_comparison(
                        image=image,
                        mask=mask,
                        pred_ft=seg_map_ft,
                        pred_orig=seg_map_orig,
                        metrics_ft=metrics_ft,
                        metrics_orig=metrics_orig,
                        sample_id=i+1,
                        output_dir=output_dir
                    )
            except Exception as e:
                print(f"Error processing image with original model: {e}")
                import traceback
                traceback.print_exc()
                # Continue with just the fine-tuned model results
                pass
        
        # Store data for visualization
        all_images.append(image)
        all_masks.append(mask)
        all_predictions_ft.append(seg_map_ft)
        all_metrics_list_ft.append(metrics_ft)
        
        if compare_models and seg_map_orig is not None:
            all_predictions_orig.append(seg_map_orig)
            all_metrics_list_orig.append(metrics_orig)
        
        # Save metrics to CSV if we have metrics (ground truth available)
        if metrics_ft:
            csv_filename_ft = save_metrics_to_csv(
                metrics_ft, 
                os.path.basename(image_path), 
                f"{mode}_fine_tuned", 
                output_dir=output_dir
            )
            if csv_filename_ft:
                all_csv_files_ft.append(os.path.join(output_dir, csv_filename_ft))
        
        if compare_models and metrics_orig:
            csv_filename_orig = save_metrics_to_csv(
                metrics_orig, 
                os.path.basename(image_path), 
                f"{mode}_original", 
                output_dir=output_dir
            )
            if csv_filename_orig:
                all_csv_files_orig.append(os.path.join(output_dir, csv_filename_orig))

    # Combine all metrics into summaries
    summary_filename_ft = None
    summary_filename_orig = None
    
    if all_csv_files_ft:
        summary_filename_ft, _ = combine_metrics(all_csv_files_ft, output_dir=output_dir)
    
    if compare_models and all_csv_files_orig:
        summary_filename_orig, _ = combine_metrics(all_csv_files_orig, output_dir=output_dir)

    # Get average metrics for fine-tuned model
    avg_metrics_ft = {}
    if all_csv_files_ft:
        # Read all metric files and calculate averages directly
        all_metrics_data = []
        for csv_file in all_csv_files_ft:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # Filter for "combined" metrics which represent overall segmentation
                combined_rows = df[df["mask_id"] == "combined"] if "mask_id" in df.columns else df
                if not combined_rows.empty:
                    all_metrics_data.append(combined_rows)
        
        # Combine all metrics data
        if all_metrics_data:
            combined_df = pd.concat(all_metrics_data, ignore_index=True)
            # Calculate average for each metric
            for metric in ["precision", "recall", "f1", "dice", "iou"]:
                if metric in combined_df.columns:
                    avg_metrics_ft[metric] = combined_df[metric].mean()
                    print(f"Fine-tuned Average {metric}: {avg_metrics_ft[metric]}")
    
    # Get average metrics for original model
    avg_metrics_orig = {}
    if compare_models and all_csv_files_orig:
        # Read all metric files and calculate averages directly
        all_metrics_data = []
        for csv_file in all_csv_files_orig:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # Filter for "combined" metrics which represent overall segmentation
                combined_rows = df[df["mask_id"] == "combined"] if "mask_id" in df.columns else df
                if not combined_rows.empty:
                    all_metrics_data.append(combined_rows)
        
        # Combine all metrics data
        if all_metrics_data:
            combined_df = pd.concat(all_metrics_data, ignore_index=True)
            # Calculate average for each metric
            for metric in ["precision", "recall", "f1", "dice", "iou"]:
                if metric in combined_df.columns:
                    avg_metrics_orig[metric] = combined_df[metric].mean()
                    print(f"Original Average {metric}: {avg_metrics_orig[metric]}")
    
    # If we still don't have avg_metrics (empty), create placeholder values
    if not avg_metrics_ft:
        print("Warning: Could not calculate average metrics for fine-tuned model, using placeholder values")
        avg_metrics_ft = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "dice": 0.0,
            "iou": 0.0
        }
    
    if compare_models and not avg_metrics_orig:
        print("Warning: Could not calculate average metrics for original model, using placeholder values")
        avg_metrics_orig = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "dice": 0.0,
            "iou": 0.0
        }

    print(f"Finished processing {len(sampled_data)} samples in {mode} mode")
    
    # Generate comparison visualization if both models are available
    if compare_models and has_ground_truth:
        # Create a detailed comparative visualization
        create_model_comparison_visualization(
            all_images[:min(3, len(all_images))],
            all_masks[:min(3, len(all_masks))],
            all_predictions_ft[:min(3, len(all_predictions_ft))],
            all_predictions_orig[:min(3, len(all_predictions_orig))],
            avg_metrics_ft,
            avg_metrics_orig,
            output_dir=output_dir
        )
    
    # Return data for comprehensive visualization
    if compare_models:
        return {
            "mode": mode,
            "images": all_images,
            "masks": all_masks,
            "predictions_ft": all_predictions_ft,
            "predictions_orig": all_predictions_orig,
            "metrics_ft": all_metrics_list_ft,
            "metrics_orig": all_metrics_list_orig,
            "avg_metrics_ft": avg_metrics_ft,
            "avg_metrics_orig": avg_metrics_orig
        }
    else:
        # For backward compatibility, return both the old and new format
        return {
            "mode": mode,
            "images": all_images,
            "masks": all_masks,
            "predictions": all_predictions_ft,
            "predictions_ft": all_predictions_ft,  # Add this for compatibility with the new function signature
            "predictions_orig": all_predictions_ft,  # Use the same for both when not comparing
            "metrics": all_metrics_list_ft,
            "metrics_ft": all_metrics_list_ft,  # Add this for compatibility with the new function signature
            "metrics_orig": all_metrics_list_ft,  # Use the same for both when not comparing
            "avg_metrics": avg_metrics_ft,
            "avg_metrics_ft": avg_metrics_ft,  # Add this for compatibility with the new function signature
            "avg_metrics_orig": avg_metrics_ft  # Use the same for both when not comparing
        }


# Create a separate visualization specifically focused on model comparison
def create_model_comparison_visualization(images, masks, predictions_ft, predictions_orig, 
                                          avg_metrics_ft, avg_metrics_orig, 
                                          output_dir="results"):
    """
    Create a dedicated visualization to compare fine-tuned and original model performance.
    
    Args:
        images: List of input images
        masks: List of ground truth masks
        predictions_ft: List of predictions from fine-tuned model
        predictions_orig: List of predictions from original model
        avg_metrics_ft: Dictionary of average metrics for fine-tuned model
        avg_metrics_orig: Dictionary of average metrics for original model
        output_dir: Directory to save the output visualization
    
    Returns:
        Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        num_samples = len(images)
        if num_samples == 0:
            print("No samples to visualize")
            return None
        
        # Create figure with appropriate size
        fig = plt.figure(figsize=(15, 5 * num_samples + 8))
        
        # Create a grid layout with 3 columns and variable rows
        grid = plt.GridSpec(num_samples + 2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Add title to the figure
        fig.suptitle('SAM2 Model Comparison: Fine-tuned vs Original', fontsize=16, y=0.98)
        
        # Plot each sample row by row
        for i in range(num_samples):
            # Original image
            ax_img = fig.add_subplot(grid[i, 0])
            ax_img.imshow(images[i])
            ax_img.set_title(f'Sample {i+1}')
            ax_img.axis('off')
            
            # Fine-tuned model prediction
            ax_ft = fig.add_subplot(grid[i, 1])
            # Create overlay visualization
            gt_binary = (masks[i] > 0).astype(bool)
            pred_binary_ft = (predictions_ft[i] > 0).astype(bool)
            
            # Compute overlap
            true_pos_ft = np.logical_and(gt_binary, pred_binary_ft)
            false_pos_ft = np.logical_and(~gt_binary, pred_binary_ft)
            false_neg_ft = np.logical_and(gt_binary, ~pred_binary_ft)
            
            # Create RGB image
            overlay_ft = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
            overlay_ft[true_pos_ft] = [0, 255, 0]  # Green for true positives
            overlay_ft[false_pos_ft] = [255, 0, 0]  # Red for false positives
            overlay_ft[false_neg_ft] = [0, 0, 255]  # Blue for false negatives
            
            ax_ft.imshow(overlay_ft)
            ax_ft.set_title(f'Fine-tuned Model (IoU: {avg_metrics_ft.get("iou", 0):.4f})')
            ax_ft.axis('off')
            
            # Original model prediction
            ax_orig = fig.add_subplot(grid[i, 2])
            # Create overlay visualization
            pred_binary_orig = (predictions_orig[i] > 0).astype(bool)
            
            # Compute overlap
            true_pos_orig = np.logical_and(gt_binary, pred_binary_orig)
            false_pos_orig = np.logical_and(~gt_binary, pred_binary_orig)
            false_neg_orig = np.logical_and(gt_binary, ~pred_binary_orig)
            
            # Create RGB image
            overlay_orig = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
            overlay_orig[true_pos_orig] = [0, 255, 0]  # Green for true positives
            overlay_orig[false_pos_orig] = [255, 0, 0]  # Red for false positives
            overlay_orig[false_neg_orig] = [0, 0, 255]  # Blue for false negatives
            
            ax_orig.imshow(overlay_orig)
            ax_orig.set_title(f'Original Model (IoU: {avg_metrics_orig.get("iou", 0):.4f})')
            ax_orig.axis('off')
        
        # Add legend
        ax_legend = fig.add_subplot(grid[num_samples, :])
        ax_legend.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='g', label='True Positive (Correct Segmentation)'),
            plt.Rectangle((0, 0), 1, 1, color='r', label='False Positive (Over-segmentation)'),
            plt.Rectangle((0, 0), 1, 1, color='b', label='False Negative (Under-segmentation)'),
        ]
        ax_legend.legend(handles=legend_elements, loc='center', ncol=3)
        
        # Add metrics comparison table
        ax_metrics = fig.add_subplot(grid[num_samples+1, :])
        ax_metrics.axis('off')
        
        # Calculate differences and improvements
        metric_diff = {}
        metric_impr = {}
        for metric in ["iou", "dice", "precision", "recall", "f1"]:
            orig_val = avg_metrics_orig.get(metric, 0)
            ft_val = avg_metrics_ft.get(metric, 0)
            metric_diff[metric] = ft_val - orig_val
            # Avoid division by zero
            metric_impr[metric] = (metric_diff[metric] / max(orig_val, 0.0001)) * 100
        
        # Prepare table data
        table_data = [
            ['Model', 'IoU', 'Dice', 'Precision', 'Recall', 'F1'],
            ['Original', f'{avg_metrics_orig.get("iou", 0):.4f}', 
                        f'{avg_metrics_orig.get("dice", 0):.4f}', 
                        f'{avg_metrics_orig.get("precision", 0):.4f}', 
                        f'{avg_metrics_orig.get("recall", 0):.4f}', 
                        f'{avg_metrics_orig.get("f1", 0):.4f}'],
            ['Fine-tuned', f'{avg_metrics_ft.get("iou", 0):.4f}', 
                          f'{avg_metrics_ft.get("dice", 0):.4f}', 
                          f'{avg_metrics_ft.get("precision", 0):.4f}', 
                          f'{avg_metrics_ft.get("recall", 0):.4f}', 
                          f'{avg_metrics_ft.get("f1", 0):.4f}'],
            ['Difference', f'{metric_diff["iou"]:.4f}', 
                          f'{metric_diff["dice"]:.4f}', 
                          f'{metric_diff["precision"]:.4f}', 
                          f'{metric_diff["recall"]:.4f}', 
                          f'{metric_diff["f1"]:.4f}'],
            ['Improvement', f'{metric_impr["iou"]:.2f}%', 
                           f'{metric_impr["dice"]:.2f}%', 
                           f'{metric_impr["precision"]:.2f}%', 
                           f'{metric_impr["recall"]:.2f}%', 
                           f'{metric_impr["f1"]:.2f}%']
        ]
        
        # Create table
        table = ax_metrics.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code improvement cells
        for j, metric in enumerate(["iou", "dice", "precision", "recall", "f1"]):
            cell = table[(3, j+1)]  # Difference row
            if metric_diff[metric] > 0:
                cell.set_facecolor('#d8f3dc')  # Light green
            elif metric_diff[metric] < 0:
                cell.set_facecolor('#ffccd5')  # Light red
                
            cell = table[(4, j+1)]  # Improvement row
            if metric_impr[metric] > 0:
                cell.set_facecolor('#d8f3dc')  # Light green
            elif metric_impr[metric] < 0:
                cell.set_facecolor('#ffccd5')  # Light red
        
        # Add a title above the table
        ax_metrics.set_title('Performance Metrics Comparison', fontsize=14, pad=5)
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Model comparison visualization saved to {fig_path}")
        return fig_path
        
    except Exception as e:
        print(f"Error creating model comparison visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SAM2 Model Inference')
    parser.add_argument('--mode', choices=['synthetic', 'real', 'both'], default='synthetic',
                      help='Inference mode: synthetic, real, or both')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of samples to process (default: 5, use 0 for all)')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                      help='Create comprehensive visualization with test and real samples')
    parser.add_argument('--compare', action='store_true', 
                      help='Compare fine-tuned model with original zero-shot model')
    parser.add_argument('--original-weights', type=str, default='sam2_hiera_small.pt',
                      help='Path to original SAM2 model weights')
    parser.add_argument('--fine-tuned-weights', type=str, default='fine_tuned_sam2_3000.torch',
                      help='Path to fine-tuned SAM2 model weights')
    parser.add_argument('--use-official-loader', action='store_true',
                      help='Use the official SAM2 loader for the original model')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with additional information')
    
    args = parser.parse_args()
    
    # Print important information about the execution
    print(f"SAM2 Model Inference")
    print(f"=" * 50)
    print(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Comparison mode: {'Enabled' if args.compare else 'Disabled'}")
    print(f"Fine-tuned model: {args.fine_tuned_weights}")
    if args.compare:
        print(f"Original model: {args.original_weights}")
    print("=" * 50)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run inference based on mode
    if args.mode == 'both' or args.visualize:
        if args.compare:
            print("Running inference on synthetic data with model comparison...")
            try:
                # Only run synthetic data with comparison when specifically requested
                synthetic_results = inference(
                    mode="synthetic", 
                    num_samples=args.samples, 
                    output_dir=args.output,
                    compare_models=True
                )
                
                # Skip real data when focusing on synthetic comparison
                real_results = None
                
                # Create only model comparison visualization for synthetic data
                if synthetic_results and "predictions_ft" in synthetic_results:
                    print("\nCreating model comparison visualization for synthetic data...")
                    try:
                        # Create a summary file with model comparison statistics
                        with open(os.path.join(args.output, "model_comparison_summary.txt"), "w") as f:
                            f.write("SAM2 Model Comparison: Fine-tuned vs Original\n")
                            f.write("===========================================\n\n")
                            
                            # Write metrics comparison
                            f.write("Average Metric Comparison:\n")
                            f.write(f"{'Metric':<12} {'Fine-tuned':<12} {'Original':<12} {'Difference':<12} {'Improvement':<12}\n")
                            f.write("-" * 60 + "\n")
                            
                            for metric in ["iou", "dice", "precision", "recall", "f1"]:
                                ft_val = synthetic_results["avg_metrics_ft"].get(metric, 0)
                                orig_val = synthetic_results["avg_metrics_orig"].get(metric, 0)
                                diff = ft_val - orig_val
                                
                                # Calculate improvement percentage
                                if orig_val > 0.0001:
                                    impr = (diff / orig_val) * 100
                                    impr_str = f"{impr:+.2f}%"
                                else:
                                    impr_str = "∞%" if ft_val > 0 else "0%"
                                
                                f.write(f"{metric.capitalize():<12} {ft_val:.4f}{' '*8} {orig_val:.4f}{' '*8} {diff:+.4f}{' '*8} {impr_str}\n")
                            
                            f.write("\nNote: The improvement percentage indicates how much better the fine-tuned model performs\n")
                            f.write("compared to the original model. Positive values indicate improvement.\n\n")
                            
                            # Add timestamp
                            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            
                        print(f"Saved model comparison summary to {os.path.join(args.output, 'model_comparison_summary.txt')}")
                        
                        # Generate visualization
                        create_model_comparison_visualization(
                            images=synthetic_results["images"][:min(3, len(synthetic_results["images"]))],
                            masks=synthetic_results["masks"][:min(3, len(synthetic_results["masks"]))],
                            predictions_ft=synthetic_results["predictions_ft"][:min(3, len(synthetic_results["predictions_ft"]))],
                            predictions_orig=synthetic_results["predictions_orig"][:min(3, len(synthetic_results["predictions_orig"]))],
                            avg_metrics_ft=synthetic_results["avg_metrics_ft"],
                            avg_metrics_orig=synthetic_results["avg_metrics_orig"],
                            output_dir=args.output
                        )
                    except Exception as e:
                        print(f"Error creating model comparison visualization: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Error during model comparison: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Standard processing for both modes without comparison
            print("Running inference on synthetic data...")
            synthetic_results = inference(
                mode="synthetic", 
                num_samples=args.samples, 
                output_dir=args.output,
                compare_models=False
            )
            
            print("\nRunning inference on real data...")
            real_results = inference(
                mode="real", 
                num_samples=args.samples, 
                output_dir=args.output,
                compare_models=False
            )
            
            # Create comprehensive visualization if requested
            if args.visualize and synthetic_results and real_results:
                print("\nCreating comprehensive visualization...")
                try:
                    # Use standard visualization with only fine-tuned model
                    create_comprehensive_visualization(
                        test_images=synthetic_results["images"],
                        test_masks=synthetic_results["masks"],
                        test_predictions_ft=synthetic_results["predictions"],
                        test_predictions_orig=synthetic_results["predictions"],  # Same as FT since we're not comparing
                        test_metrics_ft=synthetic_results["metrics"],
                        test_metrics_orig=synthetic_results["metrics"],  # Same as FT since we're not comparing
                        test_avg_metrics_ft=synthetic_results["avg_metrics"],
                        test_avg_metrics_orig=synthetic_results["avg_metrics"],  # Same as FT since we're not comparing
                        real_images=real_results["images"],
                        real_predictions_ft=real_results["predictions"],
                        real_predictions_orig=real_results["predictions"],  # Same as FT since we're not comparing
                        output_dir=args.output,
                        num_samples=min(3, len(synthetic_results["images"]), len(real_results["images"]))
                    )
                except Exception as e:
                    print(f"Error creating visualization: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Single mode inference
        try:
            inference(
                mode=args.mode, 
                num_samples=args.samples, 
                output_dir=args.output,
                compare_models=args.compare
            )
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
