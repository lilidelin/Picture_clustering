import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.t_SNE import plot_tsne


# 使用matplotlib绘画结果
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


# 在GUI界面中显示结果
def cluster_and_return_image_groups(features, image_names, image_folder, model):
    """
    对图像特征进行聚类，并返回一个聚类标签到图像路径列表的映射。

    参数：
        features: 图像特征，是一个 numpy 数组。
        image_names: 图像文件名列表。
        image_folder: 图像所在文件夹路径。
        model: 聚类模型（如 KMeans）。

    返回：
        一个 dict，键是类别标签（整数），值是该类对应图像的完整路径列表。
    """
    labels = model.fit_predict(features)
    plot_tsne(features, labels)
    cluster_map = defaultdict(list)

    for file, label in zip(image_names, labels):
        path = os.path.join(image_folder, file)
        cluster_map[label].append(path)

    return dict(cluster_map)  # 转成普通 dict 方便 PyQt 使用
