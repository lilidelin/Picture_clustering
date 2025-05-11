import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from src.models.model_factory import ModelFactory
import src.utils.image_processing as ImageProcessor
import src.utils.visualization as Visualizer
from PIL import Image
import torch


class ClusteringThread(QThread):
    progress_updated = pyqtSignal(int, str)  # 进度更新信号（进度值，状态信息）
    clustering_finished = pyqtSignal(dict)  # 聚类完成信号
    error_occurrred = pyqtSignal(str)  # 错误发生信号

    def __init__(self, folder, feature_model, cluster_model, params):
        super().__init__()
        self.folder = folder
        self.feature_model = feature_model
        self.cluster_model = cluster_model
        self.params = params

    def run(self):
        try:
            # 创建模型工厂和图像处理程序
            model_factory = ModelFactory()
            feature_extractor = model_factory.get_feature_extractor(self.feature_model)
            cluster_model = model_factory.get_cluster_model(self.cluster_model, self.params)
            transform = ImageProcessor.get_image_transform(self.feature_model)

            # 获取图片列表
            image_files = []
            for filename in os.listdir(self.folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(filename)

            total_images = len(image_files)
            if total_images == 0:
                raise ValueError("所选文件夹中没有找到图片")

            self.progress_updated.emit(5, f"找到 {total_images} 张图片")

            # 提取特征
            features = []
            processed = 0
            for filename in image_files:
                image_path = os.path.join(self.folder, filename)
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        feature = feature_extractor(image_tensor)
                        feature = feature.squeeze().numpy()
                        features.append(feature)

                    processed += 1
                    progress = 5 + int((processed / total_images) * 70)  # 5%~75%
                    self.progress_updated.emit(progress, f"已提取 {processed}/{total_images} 张图片的特征")
                except Exception as e:
                    self.progress_updated.emit(progress, f"无法处理图片 {filename}: {str(e)}")

            self.progress_updated.emit(75, "特征提取完成，开始聚类...")

            # 聚类
            features = np.array(features)
            cluster_map = Visualizer.cluster_and_return_image_groups(features, image_files, self.folder, cluster_model)
            self.progress_updated.emit(90, "聚类完成，正在准备显示结果...")

            # 返回结果
            self.progress_updated.emit(100, "完成")
            self.clustering_finished.emit(cluster_map)

        except Exception as e:
            self.error_occurrred.emit(str(e))
