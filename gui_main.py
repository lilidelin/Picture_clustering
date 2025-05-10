import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QSpinBox, QHBoxLayout, QMessageBox, QScrollArea,
    QComboBox
)
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from features import features_extractor
from models import KMeans, ResNet50
from clustering import clutering_and_show_result
from preprocess import image_preprocessor
from models import Vit16
from models import DBSCAN
from clustering import clutering_and_show_result


class ImageClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片聚类程序")
        self.resize(1000, 700)
        self.folder = ""
        self.k = 3
        self.init_ui()

    # 界面设计函数
    def init_ui(self):
        main_layout = QVBoxLayout()

        title = QLabel("图片聚类系统")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # 文件夹选择部分
        self.label_folder = QLabel("未选择文件夹")
        self.btn_select_folder = QPushButton("选择图片文件夹")
        self.btn_select_folder.clicked.connect(self.select_folder)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.label_folder)
        folder_layout.addWidget(self.btn_select_folder)
        main_layout.addLayout(folder_layout)

        # KMeans聚类参数选择
        param_layout = QHBoxLayout()
        self.k_selector = QSpinBox()
        self.k_selector.setMinimum(2)
        self.k_selector.setMaximum(20)
        self.k_selector.setValue(3)
        param_layout.addWidget(QLabel("聚类数 K:"))
        param_layout.addWidget(self.k_selector)

        self.btn_cluster = QPushButton("开始聚类")
        self.btn_cluster.clicked.connect(self.run_clustering)
        param_layout.addWidget(self.btn_cluster)

        main_layout.addLayout(param_layout)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # 图片显示区域（带滚动）
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout()
        self.result_widget.setLayout(self.result_layout)
        self.result_area.setWidget(self.result_widget)

        main_layout.addWidget(self.result_area)

        self.setLayout(main_layout)

        # 模型选择区域
        model_layout = QHBoxLayout()

        # 特征模型选择
        self.feature_combobox = QComboBox()
        self.feature_combobox.addItems(["ResNet50", "ResNet18", "Vit16"])
        model_layout.addWidget(QLabel("特征提取模型"))
        model_layout.addWidget(self.feature_combobox)

        # 聚类算法选择
        self.cluster_combobox = QComboBox()
        self.cluster_combobox.addItems(['KMeans', "DBSCAN"])
        model_layout.addWidget(QLabel("聚类算法"))
        model_layout.addWidget(self.cluster_combobox)

        main_layout.addLayout(model_layout)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.folder = folder
            self.label_folder.setText(f"已选择: {folder}")

    def run_clustering(self):
        if not self.folder:
            QMessageBox.warning(self, "错误", "请先选择图片文件夹")
            return

        k = self.k_selector.value()
        # 获取用户选择的模型
        feature_model_name = self.feature_combobox.currentText()
        cluster_model_name = self.cluster_combobox.currentText()

        # 映射模型名称到类
        feature_models = {
            "ResNet50": ResNet50.load_ResNet50(),
            "ResNet18": ResNet50.load_ResNet18(),
            "Vit16": Vit16.load_Vit16()
        }
        cluster_models = {
            "KMeans": KMeans.load_KMeans(k),
            "DBSCAN": DBSCAN.load_DBSCAN(1.2, 5)
        }
        transform_models = {
            "ResNet50": image_preprocessor.get_image_transform_for_resnet(),
            "ResNet18": image_preprocessor.get_image_transform_for_resnet(),
            "Vit16": image_preprocessor.get_image_transform_for_vit()
        }
        try:
            # 获取相应的图片转换模型，特征提取模型，聚类模型
            transform = transform_models[feature_model_name]
            feature_extractor = feature_models[feature_model_name]

            cluster_model = cluster_models[cluster_model_name]

            features, file_names = features_extractor.extract_features_from_folder(
                self.folder, transform, feature_extractor
            )
            cluster_map = clutering_and_show_result.cluster_and_return_image_groups(
                features, file_names, self.folder, cluster_model
            )

            self.display_clusters(cluster_map)

        except Exception as e:
            QMessageBox.critical(self, "出错啦", str(e))

    def display_clusters(self, cluster_map):
        # 清空旧的内容
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # 逐个类别显示
        for label in sorted(cluster_map.keys()):
            group_box = QGroupBox(f"类别 {label}")
            grid = QGridLayout()

            images = cluster_map[label][:10]  # 每类最多显示 10 张
            for idx, image_path in enumerate(images):
                img_label = QLabel()
                pixmap = QPixmap(image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                grid.addWidget(img_label, idx // 5, idx % 5)

            group_box.setLayout(grid)
            self.result_layout.addWidget(group_box)

        self.result_layout.addStretch()


if __name__ == '__main__':
    print("main")
    app = QApplication(sys.argv)
    win = ImageClusteringApp()
    win.show()
    sys.exit(app.exec_())
