import os
import torch
from PIL import Image
import numpy as np


def extract_features_from_folder(folder_path, transform, feature_extractor):
    feature_list = []
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")

            image_tensor = transform(image).unsqueeze(0)  # 增加batch维度

            with torch.no_grad():
                features = feature_extractor(image_tensor)  # 输出shape:[1,2048,1,1]
                features = features.squeeze().numpy()  # 变成shape:(2048,)
                feature_list.append(features)
                image_names.append(filename)
    return np.array(feature_list), image_names
