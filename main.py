from features import features_extractor
from models import KMeans
from models import ResNet50
from clustering import clutering_and_show_result
from preprocess import image_preprocessor

if __name__ == '__main__':
    folder = "Pictures1"  # 图片的文件夹
    transform = image_preprocessor.get_image_transform()  # 获取图片转换器
    feature_extractor = ResNet50.load_ResNet50()  # 获取特征提取器
    features, file_name = features_extractor.extract_features_from_folder(
        folder, transform, feature_extractor
    )  # 提取图片特征
    clutering_and_show_result.cluster_and_show_results(features,
                                                       file_name, folder, KMeans.load_KMeans(3))  # 聚类及展示结果
