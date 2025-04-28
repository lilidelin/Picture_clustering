import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def cluster_and_show_results(features, image_names, image_folder, model):
    # 聚类
    labels = model.fit_predict(features)
    labels_class = np.unique(labels)
    labels_len = len(labels_class)

    # 为每个簇创建子图
    fig, axes = plt.subplots(labels_len, 5, figsize=(15, 3 * labels_len))
    if labels_len == 1:
        axes = np.expand_dims(axes, axis=0)

    # 每张图显示在对应类下
    for i in range(labels_len):
        cluster_indices = np.where(labels == i)[0]
        for j in range(min(5, len(cluster_indices))):  # 没类最多显示五张图
            img_index = cluster_indices[j]
            img_path = os.path.join(image_folder, image_names[img_index])
            image = Image.open(img_path).convert('RGB')
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
            axes[i, j].set_title(f"Cluster{i}")
        for j in range(len(cluster_indices), 5):
            axes[i, j].axis('off')  # 不带图片的格子也隐藏

    plt.tight_layout()
    plt.show()
