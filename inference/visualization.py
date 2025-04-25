import subprocess
import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random  # 添加random模块导入
from sklearn.model_selection import train_test_split
# 不使用build_sam2，直接导入必要的模型类
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Path to the chest-ct-segmentation dataset folder
data_dir = "data"
images_dir = os.path.join(data_dir, "images/images")
masks_dir = os.path.join(data_dir, "masks/masks")

# Load the train.csv file
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Split the data into two halves: one for training and one for testing
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Prepare the training data list
train_data = []
for index, row in train_df.iterrows():
    image_name = row['ImageId']
    mask_name = row['MaskId']

    # Append image and corresponding mask paths
    train_data.append({
        "image": os.path.join(images_dir, image_name),
        "annotation": os.path.join(masks_dir, mask_name)
    })

# Prepare the testing data list (if needed for inference or evaluation later)
test_data = []
for index, row in test_df.iterrows():
    image_name = row['ImageId']
    mask_name = row['MaskId']

    # Append image and corresponding mask paths
    test_data.append({
        "image": os.path.join(images_dir, image_name),
        "annotation": os.path.join(masks_dir, mask_name)
    })


def read_batch(data, visualize_data=False):
    # Select a random entry
    ent = data[np.random.randint(len(data))]

    # Get full paths
    Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    # Read annotation as grayscale
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)

    if Img is None or ann_map is None:
        print(
            f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
        return None, None, None, 0

    # Resize image and mask
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(
        ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    ### Continuation of read_batch() ###

    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        # Create binary mask for each unique index
        mask = (ann_map == ind).astype(np.uint8)
        # Combine with the existing binary mask
        binary_mask = np.maximum(binary_mask, mask)

    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones(
        (5, 5), np.uint8), iterations=1)

    # Get all coordinates inside the eroded mask and choose a random point
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        for _ in inds:  # Select as many points as there are unique labels
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])

    points = np.array(points)
    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(Img)
        plt.axis('on')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('on')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(
                # Corrected to plot y, x order
                colors)], s=100, label=f'Point {i+1}')

        # plt.legend()
        plt.axis('on')

        plt.tight_layout()
        plt.show()

    # Now shape is (1024, 1024, 1)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)
    # Return the image, binarized mask, points, and number of masks
    return Img, binary_mask, points, len(inds)


if __name__ == "__main__":
    # Make sure the train_data is not empty
    if len(train_data) == 0:
        print("No training data found. Check your data paths.")
    else:
        print(f"Found {len(train_data)} training samples.")
        # Try visualizing a few random samples
        for i in range(3):  # Visualize 3 random samples
            print(f"Visualizing sample {i+1}/3...")
            img, mask, points, num_masks = read_batch(
                train_data, visualize_data=True)
            if img is None:
                print("Failed to read image or mask.")
            else:
                print(
                    f"Successfully visualized sample with {num_masks} masks.")
