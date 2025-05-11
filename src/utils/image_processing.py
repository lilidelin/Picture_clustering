import torchvision.transforms as transforms


# 图像预处理适配ResNet
def get_image_transform_for_resnet():
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_image_transform_for_vit():
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform

def get_image_transform(model_name):
    if model_name in ["ResNet50","ResNet18"]:
        return get_image_transform_for_resnet()
    elif model_name == "Vit16":
        return get_image_transform_for_vit()
    else:
        raise ValueError(f"Unsupported model for image transform: {model_name}")
