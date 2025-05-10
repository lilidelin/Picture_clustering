from torchvision import models
from torchvision.models import ViT_B_16_Weights
model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
print(model)
