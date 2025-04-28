import torch.nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


def load_ResNet50():
    # 1.加载预训练的ResNet50模型
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    # 去掉最后的分类层
    # 获取CNN的输出特征向量（2048维）
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return feature_extractor
