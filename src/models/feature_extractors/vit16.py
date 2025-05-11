
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

def load_Vit16():
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.eval()
    return model
