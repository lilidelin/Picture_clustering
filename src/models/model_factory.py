from src.models.feature_extractors.resnet import load_ResNet50, load_ResNet18
from src.models.feature_extractors.vit16 import load_Vit16
from src.models.clustering.KMeans import load_KMeans
from src.models.clustering.dbscan import load_DBSCAN


class ModelFactory:
    @staticmethod
    def get_feature_extractor(model_name):
        if model_name == "ResNet50":
            return load_ResNet50()
        elif model_name == "ResNet18":
            return load_ResNet18()
        elif model_name == "Vit16":
            return load_Vit16()
        else:
            raise ValueError(f"Unsupported feature extractor: {model_name}")

    @staticmethod
    def get_cluster_model(model_name, params):
        if model_name == "KMeans":
            return load_KMeans(params["k"])
        elif model_name == "DBSCAN":
            return load_DBSCAN(params['eps'], params["min_samples"])
        else:
            raise ValueError(f"Unsupported clustering model: {model_name}")
