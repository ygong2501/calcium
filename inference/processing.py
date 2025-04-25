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
                colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order

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

# Visualize the data
# Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)


# model loading and training
# @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
# @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
sam2_checkpoint = "sam2_hiera_small.pt"

# 加载预训练权重
checkpoint = torch.load(sam2_checkpoint, map_location="cpu")
if "model" in checkpoint:
    state_dict = checkpoint["model"]
    print("找到了预训练权重!")
else:
    print("警告: 检查点文件中没有'model'键!")
    print("检查点包含的键:", checkpoint.keys())
    state_dict = checkpoint  # 尝试直接使用检查点作为状态字典

# 修改模型加载部分
print("开始加载SAM2模型...")

# 首先尝试安装huggingface_hub
try:
    print("尝试安装huggingface_hub...")
    subprocess.check_call(["pip", "install", "huggingface_hub"])
    print("安装huggingface_hub成功!")
except Exception as e:
    print(f"安装huggingface_hub失败: {e}")

try:
    # 尝试直接从预训练模型加载
    print("尝试从Hugging Face加载预训练模型...")
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-small", device="cuda")
    print("从Hugging Face成功加载模型!")
except Exception as e:
    print(f"从Hugging Face加载失败: {e}")

    # 如果从HF加载失败，尝试从本地文件加载
    try:
        print("尝试从本地文件加载模型...")
        sam2_checkpoint = "sam2_hiera_small.pt"
        checkpoint = torch.load(sam2_checkpoint, map_location="cpu")

        # 分析检查点结构
        if "model" in checkpoint:
            print("找到model键，尝试获取ModelConfig...")
            from sam2.build_sam import build_sam2

            # 使用from_config方法
            try:
                # 尝试从yaml加载配置
                model_cfg = "sam2_hiera_s.yaml"
                # 使用subprocess调用示例代码加载模型，避免OmegaConf错误
                print("尝试使用示例代码加载模型...")

                # 创建一个临时Python文件
                with open("load_model_temp.py", "w") as f:
                    f.write('''
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# 加载模型而不加载权重
model = build_sam2("sam2_hiera_s.yaml", ckpt_path=None, device="cuda", load_pretrained=False)

# 手动加载权重
checkpoint = torch.load("sam2_hiera_small.pt", map_location="cpu")
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
    torch.save(model, "sam2_model_loaded.pt")
    print("成功保存模型!")
else:
    print("检查点中没有model键")
''')

                # 执行临时Python文件
                subprocess.check_call(["python", "load_model_temp.py"])

                # 加载保存的模型
                print("加载保存的模型...")
                sam2_model = torch.load("sam2_model_loaded.pt")
                predictor = SAM2ImagePredictor(sam2_model)
                print("成功加载模型!")

            except Exception as e2:
                print(f"加载模型配置失败: {e2}")
                print("无法加载SAM2模型，请检查模型文件或配置")
                raise e2
        else:
            print(f"检查点不包含'model'键，包含的键: {checkpoint.keys()}")
            raise ValueError("检查点格式不正确")
    except Exception as e3:
        print(f"所有加载尝试都失败: {e3}")
        raise e3

# Train mask decoder
print("设置模型组件为训练模式...")
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer = torch.optim.AdamW(params=predictor.model.parameters(
), lr=0.0001, weight_decay=1e-4)  # 1e-5, weight_decay = 4e-5

# Mix precision.
scaler = torch.cuda.amp.GradScaler()

# No. of steps to train the model.
NO_OF_STEPS = 3000  # @param

# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"

# Initialize scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=500, gamma=0.2)  # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

for step in range(1, NO_OF_STEPS + 1):
    with torch.cuda.amp.autocast():
        image, mask, input_point, num_masks = read_batch(
            train_data, visualize_data=False)
        if image is None or mask is None or num_masks == 0:
            continue

        input_label = np.ones((num_masks, 1))
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            continue

        if input_point.size == 0 or input_label.size == 0:
            continue

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(
                0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1])

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) -
                    (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) +
                       (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            predictor.model.parameters(), max_norm=1.0)

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()

        # Update scheduler
        scheduler.step()

        if step % 500 == 0:
            FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + \
                "_" + str(step) + ".torch"
            torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

        if step == 1:
            mean_iou = 0

        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

        if step % 100 == 0:
            print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)

# model inferance
def read_image(image_path, mask_path):  # read and resize image and mask
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(
        mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask


def get_points(mask, num_points):  # Sample points inside the input mask
    points = []
    coords = np.argwhere(mask > 0)
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)


# 测试代码部分
def test_fine_tuned_model():
    # 随机选择一个测试图像
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']

    print(f"测试图像路径: {image_path}")
    print(f"测试掩码路径: {mask_path}")

    # 加载选定的图像和掩码
    image, mask = read_image(image_path, mask_path)
    if image is None or mask is None:
        print("无法加载图像或掩码")
        return

    # 为输入生成随机点
    num_samples = 30  # 每个分段采样的点数
    input_points = get_points(mask, num_samples)
    if len(input_points) == 0:
        print("无法生成输入点")
        return

    # 加载微调后的模型
    FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_3000.torch"  # 使用3000步权重
    model_cfg = "sam2_hiera_s.yaml"
    sam2_checkpoint = "sam2_hiera_small.pt"

    try:
        print("正在加载SAM2模型配置...")
        config = load_parameters_from_yaml(model_cfg)

        print("正在从配置直接构建SAM2模型...")
        sam2_model = build_sam2_model_direct(config)

        print("构建SAM2图像预测器...")
        predictor = SAM2ImagePredictor(sam2_model)

        print(f"加载微调权重: {FINE_TUNED_MODEL_WEIGHTS}")
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
        predictor.model.eval()  # 设置为评估模式

        # 将模型移至GPU（如果可用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor.model = predictor.model.to(device)

        print(f"模型已加载并移至 {device}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 执行推理和预测掩码
    try:
        print("设置图像并执行预测...")
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )

        print(f"预测完成，获得 {len(masks)} 个掩码")
    except Exception as e:
        print(f"预测过程中出错: {e}")
        return

    # 处理预测的掩码并按分数排序
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_indices = np.argsort(np_scores)[::-1]
    sorted_masks = np_masks[sorted_indices]

    # 初始化分割图和占用掩码
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    # 合并掩码创建最终分割图
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        # 如果与已有掩码重叠过多，则跳过
        if (mask * occupancy_mask).sum() / (mask.sum() + 1e-10) > 0.15:
            continue

        mask_bool = mask.astype(bool)
        # 将重叠区域在掩码中设置为False
        mask_bool[occupancy_mask] = False
        seg_map[mask_bool] = i + 1  # 使用布尔掩码索引seg_map
        occupancy_mask[mask_bool] = True  # 更新占用掩码

    # 可视化：并排显示原始图像、掩码和最终分割
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('测试图像')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('原始掩码')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('最终分割')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    print("分割结果已保存为 segmentation_result.png")
    plt.show()


# 执行测试
if __name__ == "__main__":
    test_fine_tuned_model()
