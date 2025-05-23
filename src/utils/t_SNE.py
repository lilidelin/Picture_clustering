import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(features, labels, save_path=None):
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
    print("正在计算 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", len(np.unique(labels)))
    sns.scatterplot(
        x=reduced_features[:, 0],
        y=reduced_features[:, 1],
        hue=labels,
        legend='full',
        palette=palette
    )
    plt.title("t-SNE 聚类可视化")
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
