import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import torch.nn as nn


def load_Vit16():
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.eval()

    # # 获取丢弃概率值
    # dropout_p = model.encoder.dropout.p
    #
    # # 获取patch embedding层的正确方式
    # patch_embedding = nn.Sequential(
    #     model.conv_proj,  # 使用conv_proj替代patch_embed
    #     model.class_token,
    #     model.encoder.pos_embedding,  # 使用encoder.pos_embedding替代pos_embed
    #     nn.Dropout(dropout_p)  # 传递丢弃概率值而不是Dropout实例
    # )
    #
    # # 前6层编码器
    # encoder_layers = list(model.encoder.layers)[:6]
    # encoder = nn.Sequential(*encoder_layers)
    #
    # # 组合特征提取器
    # feature_extractor = nn.Sequential(
    #     patch_embedding,
    #     encoder
    # )

    return model