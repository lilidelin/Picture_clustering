import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QSpinBox, QHBoxLayout, QMessageBox, QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from features import features_extractor
from models import KMeans, ResNet50
from clustering import clutering_and_show_result
from preprocess import image_preprocessor


class ImageClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片聚类程序")
        self.folder = ""
        self.k = 3
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label_folder = QLabel("未选择文件夹")
        self.btn_select_folder = QPushButton("选择图片文件夹")
        self.btn_select_folder.clicked.connect(self.select_folder)

        h_layout = QHBoxLayout()
        self.k_selector = QSpinBox()
        self.k_selector.setMinimum(2)
        self.k_selector.setMaximum(20)
        self.k_selector.setValue(3)
        h_layout.addWidget(QLabel("聚类数 K:"))
        h_layout.addWidget(self.k_selector)

        self.btn_cluster = QPushButton("开始聚类")
        self.btn_cluster.clicked.connect(self.run_clustering)

        layout.addWidget(self.label_folder)
        layout.addWidget(self.btn_select_folder)
        layout.addLayout(h_layout)
        layout.addWidget(self.btn_cluster)

        self.setLayout(layout)

        self.result_area = QScrollArea()
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_area.setWidgetResizable(True)
        self.result_area.setWidget(self.result_label)
        layout.addWidget(self.result_area)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.folder = folder
            self.label_folder.setText(f"已选择: {folder}")

    def run_clustering(self):
        if not self.folder:
            QMessageBox.warning(self, "错误", "请先选择图片文件夹")
            return

        try:
            k = self.k_selector.value()
            transform = image_preprocessor.get_image_transform()
            feature_extractor = ResNet50.load_ResNet50()
            features, file_names = features_extractor.extract_features_from_folder(
                self.folder, transform, feature_extractor
            )
            model = KMeans.load_KMeans(k)

            from clustering import clutering_and_show_result
            cluster_map = clutering_and_show_result.cluster_and_return_image_groups(
                features, file_names, self.folder, model
            )

            self.display_clusters(cluster_map)

        except Exception as e:
            QMessageBox.critical(self, "出错啦", str(e))

    def display_clusters(self, cluster_map):
        html = ""
        for label in sorted(cluster_map):
            html += f"<h3>类别 {label}</h3><div>"
            for path in cluster_map[label][:10]:  # 每类只显示前10张
                html += f'<img src="{path}" width="100" style="margin:4px"/>'
            html += "</div><hr/>"
        self.result_label.setText(html)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageClusteringApp()
    win.show()
    sys.exit(app.exec_())
