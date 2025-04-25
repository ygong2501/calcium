"""
直接SAM2模型加载脚本

该脚本会从源码直接构建SAM2模型，绕过build_sam2函数和OmegaConf配置系统，
以解决"Missing key to"和"Missing key load_state_dict"等错误。
"""

import torch
import sys
import os
import yaml

# 导入必要的SAM2模型相关类
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import CXBlock
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_parameters_from_yaml(yaml_file):
    """从YAML文件加载参数，但不使用OmegaConf"""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['model']


def build_position_encoding(config):
    """构建位置编码模块"""
    return PositionEmbeddingSine(
        num_pos_feats=config.get('num_pos_feats', 256),
        normalize=config.get('normalize', True),
        scale=config.get('scale', None),
        temperature=config.get('temperature', 10000)
    )


def build_rope_attention(config):
    """构建RoPE注意力模块"""
    return RoPEAttention(
        rope_theta=config.get('rope_theta', 10000.0),
        feat_sizes=config.get('feat_sizes', [32, 32]),
        rope_k_repeat=config.get('rope_k_repeat', True),
        embedding_dim=config.get('embedding_dim', 256),
        num_heads=config.get('num_heads', 1),
        downsample_rate=config.get('downsample_rate', 1),
        dropout=config.get('dropout', 0.1),
        kv_in_dim=config.get('kv_in_dim', 64)
    )


def build_memory_attention_layer(config):
    """构建内存注意力层"""
    self_attention = build_rope_attention(config['self_attention'])
    cross_attention = build_rope_attention(config['cross_attention'])

    return MemoryAttentionLayer(
        d_model=config.get('d_model', 256),
        self_attention=self_attention,
        cross_attention=cross_attention,
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        pos_enc_at_attn=config.get('pos_enc_at_attn', False),
        pos_enc_at_cross_attn_keys=config.get(
            'pos_enc_at_cross_attn_keys', True),
        pos_enc_at_cross_attn_queries=config.get(
            'pos_enc_at_cross_attn_queries', False)
    )


def build_memory_attention(config):
    """构建内存注意力模块"""
    layer = build_memory_attention_layer(config['layer'])

    return MemoryAttention(
        d_model=config.get('d_model', 256),
        layer=layer,
        num_layers=config.get('num_layers', 4),
        pos_enc_at_input=config.get('pos_enc_at_input', True)
    )


def build_memory_encoder(config):
    """构建内存编码器"""
    position_encoding = build_position_encoding(config['position_encoding'])

    mask_downsampler = MaskDownSampler(
        kernel_size=config['mask_downsampler'].get('kernel_size', 3),
        stride=config['mask_downsampler'].get('stride', 2),
        padding=config['mask_downsampler'].get('padding', 1)
    )

    cx_block = CXBlock(
        dim=config['fuser']['layer'].get('dim', 256),
        kernel_size=config['fuser']['layer'].get('kernel_size', 7),
        padding=config['fuser']['layer'].get('padding', 3),
        layer_scale_init_value=config['fuser']['layer'].get(
            'layer_scale_init_value', 1e-6),
        use_dwconv=config['fuser']['layer'].get('use_dwconv', True)
    )

    fuser = Fuser(
        layer=cx_block,
        num_layers=config['fuser'].get('num_layers', 2)
    )

    return MemoryEncoder(
        out_dim=config.get('out_dim', 64),
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser
    )


def build_image_encoder(config):
    """构建图像编码器"""
    trunk = Hiera(
        embed_dim=config['trunk'].get('embed_dim', 96),
        num_heads=config['trunk'].get('num_heads', 1),
        stages=config['trunk'].get('stages', [1, 2, 11, 2]),
        global_att_blocks=config['trunk'].get(
            'global_att_blocks', [7, 10, 13]),
        window_pos_embed_bkg_spatial_size=config['trunk'].get(
            'window_pos_embed_bkg_spatial_size', [7, 7])
    )

    position_encoding = build_position_encoding(
        config['neck']['position_encoding'])

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=config['neck'].get('d_model', 256),
        backbone_channel_list=config['neck'].get(
            'backbone_channel_list', [768, 384, 192, 96]),
        fpn_top_down_levels=config['neck'].get('fpn_top_down_levels', [2, 3]),
        fpn_interp_model=config['neck'].get('fpn_interp_model', 'nearest')
    )

    return ImageEncoder(
        trunk=trunk,
        neck=neck,
        scalp=config.get('scalp', 1)
    )


def build_sam2_model_direct(config):
    """直接从配置构建SAM2模型，避免OmegaConf"""
    # 构建各个组件
    image_encoder = build_image_encoder(config['image_encoder'])
    memory_attention = build_memory_attention(config['memory_attention'])
    memory_encoder = build_memory_encoder(config['memory_encoder'])

    # 创建SAM2Base模型
    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        image_size=config.get('image_size', 1024),
        num_maskmem=config.get('num_maskmem', 7),
        sigmoid_scale_for_mem_enc=config.get(
            'sigmoid_scale_for_mem_enc', 20.0),
        sigmoid_bias_for_mem_enc=config.get('sigmoid_bias_for_mem_enc', -10.0),
        use_mask_input_as_output_without_sam=config.get(
            'use_mask_input_as_output_without_sam', True),
        directly_add_no_mem_embed=config.get(
            'directly_add_no_mem_embed', True),
        use_high_res_features_in_sam=config.get(
            'use_high_res_features_in_sam', True),
        multimask_output_in_sam=config.get('multimask_output_in_sam', True),
        iou_prediction_use_sigmoid=config.get(
            'iou_prediction_use_sigmoid', True),
        use_obj_ptrs_in_encoder=config.get('use_obj_ptrs_in_encoder', True),
        add_tpos_enc_to_obj_ptrs=config.get('add_tpos_enc_to_obj_ptrs', False),
        only_obj_ptrs_in_the_past_for_eval=config.get(
            'only_obj_ptrs_in_the_past_for_eval', True),
        pred_obj_scores=config.get('pred_obj_scores', True),
        pred_obj_scores_mlp=config.get('pred_obj_scores_mlp', True),
        fixed_no_obj_ptr=config.get('fixed_no_obj_ptr', True),
        multimask_output_for_tracking=config.get(
            'multimask_output_for_tracking', True),
        use_multimask_token_for_obj_ptr=config.get(
            'use_multimask_token_for_obj_ptr', True),
        multimask_min_pt_num=config.get('multimask_min_pt_num', 0),
        multimask_max_pt_num=config.get('multimask_max_pt_num', 1),
        use_mlp_for_obj_ptr_proj=config.get('use_mlp_for_obj_ptr_proj', True),
        compile_image_encoder=config.get('compile_image_encoder', False)
    )

    return model


def fix_yaml(config_path):
    """修复YAML文件，移除load_state_dict字段"""
    with open(config_path, 'r') as f:
        content = f.read()

    if 'load_state_dict: {}' in content:
        fixed_content = content.replace('load_state_dict: {}', '')
        fixed_path = f"fixed_{os.path.basename(config_path)}"
        with open(fixed_path, 'w') as f:
            f.write(fixed_content)
        return fixed_path
    return config_path


if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) < 3:
        print("Usage: python load_sam2_direct.py <config_path> <output_model_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2]

    # 修复YAML文件
    fixed_config_path = fix_yaml(config_path)

    try:
        # 加载配置
        config = load_parameters_from_yaml(fixed_config_path)

        # 构建模型
        print("直接从源码构建SAM2模型...")
        model = build_sam2_model_direct(config)

        # 将模型移动到GPU（如果可用）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # 保存模型
        torch.save(model, output_path)
        print(f"模型已成功保存到 {output_path}")

    except Exception as e:
        print(f"构建模型时出错: {e}")
        raise e
