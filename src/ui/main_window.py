from src.utils.clustering_thread import ClusteringThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QSpinBox, QHBoxLayout, QMessageBox, QScrollArea,
    QComboBox, QDoubleSpinBox, QProgressBar
)
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片聚类程序")
        self.resize(1000, 700)
        self.folder = ""
        self.clustering_thread = None  # 存储聚类线程
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
        self.cluster_combobox.currentTextChanged.connect(self.update_cluster_params)
        model_layout.addWidget(QLabel("聚类算法"))
        model_layout.addWidget(self.cluster_combobox)

        main_layout.addLayout(model_layout)

        # 聚类参数面板
        self.param_group = QGroupBox("聚类参数")
        self.param_layout = QGridLayout()
        self.param_group.setLayout(self.param_layout)
        main_layout.addWidget(self.param_group)

        # 初始化参数面板
        self.init_cluster_params()

        # 开始聚类按钮
        self.btn_cluster = QPushButton("开始聚类")
        self.btn_cluster.clicked.connect(self.run_clustering)
        main_layout.addWidget(self.btn_cluster)

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

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("准备就绪")
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)

        self.setLayout(main_layout)

    def init_cluster_params(self):
        # 清空参数面板
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # 默认显示Kmeans参数
        self.k_selector = QSpinBox()
        self.k_selector.setMinimum(2)
        self.k_selector.setMaximum(50)
        self.k_selector.setValue(4)

        self.param_layout.addWidget(QLabel("聚类数 K:"), 0, 0)
        self.param_layout.addWidget(self.k_selector, 0, 1)

        # DBSCAN参数控件（默认隐藏）
        self.eps_selector = QDoubleSpinBox()
        self.eps_selector.setMinimum(0.1)
        self.eps_selector.setMaximum(100)
        self.eps_selector.setSingleStep(0.1)
        self.eps_selector.setValue(1.2)

        self.min_samples_selector = QSpinBox()
        self.min_samples_selector.setMinimum(1)
        self.min_samples_selector.setMaximum(50)
        self.min_samples_selector.setValue(5)

        self.param_layout.addWidget(QLabel("邻域半径 eps:"), 1, 0)
        self.param_layout.addWidget(self.eps_selector, 1, 1)
        self.param_layout.addWidget(QLabel("最小样本数 min_samples:"), 2, 0)
        self.param_layout.addWidget(self.min_samples_selector, 2, 1)

        # 根据当前选择更新可见性
        self.update_cluster_params(self.cluster_combobox.currentText())

    def update_cluster_params(self, cluster_method):
        # 根据选择的聚类方法显示/隐藏相应的参数控件
        if cluster_method == "KMeans":
            self.k_selector.show()
            self.param_layout.itemAtPosition(0, 0).widget().show()  # 显示"聚类数 K:"标签

            self.eps_selector.hide()
            self.min_samples_selector.hide()
            self.param_layout.itemAtPosition(1, 0).widget().hide()  # 隐藏"邻域半径 eps:"标签
            self.param_layout.itemAtPosition(2, 0).widget().hide()  # 隐藏"最小样本数 min_samples:"标签
        elif cluster_method == "DBSCAN":
            self.k_selector.hide()
            self.param_layout.itemAtPosition(0, 0).widget().hide()  # 隐藏"聚类数 K:"标签

            self.eps_selector.show()
            self.min_samples_selector.show()
            self.param_layout.itemAtPosition(1, 0).widget().show()  # 显示"邻域半径 eps:"标签
            self.param_layout.itemAtPosition(2, 0).widget().show()  # 显示"最小样本数 min_samples:"标签

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.folder = folder
            self.label_folder.setText(f"已选择: {folder}")

    def run_clustering(self):
        if not self.folder:
            QMessageBox.warning(self, "错误", "请先选择图片文件夹")
            return

        if self.clustering_thread and self.clustering_thread.isRunning():
            QMessageBox.warning(self, "提示", "聚类正在进行中，请等待")
            return

            # 获取用户选择的模型
        feature_model_name = self.feature_combobox.currentText()
        cluster_model_name = self.cluster_combobox.currentText()

        # 获取聚类参数
        if cluster_model_name == "KMeans":
            k = self.k_selector.value()
            cluster_params = {"k": k}
        else:  # DBSCAN
            eps = self.eps_selector.value()
            min_samples = self.min_samples_selector.value()
            cluster_params = {"eps": eps, "min_samples": min_samples}

        # 禁用按钮，防止重复点击
        self.btn_cluster.setEnabled(False)

        # 创建并启动聚类线程
        self.clustering_thread = ClusteringThread(
            self.folder, feature_model_name, cluster_model_name, cluster_params
        )
        self.clustering_thread.progress_updated.connect(self.update_progress)
        self.clustering_thread.clustering_finished.connect(self.on_clustering_finished)
        self.clustering_thread.error_occurrred.connect(self.on_clustering_error)
        self.clustering_thread.start()

    def update_progress(self, value, message):
        """更新进度条和状态信息"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_clustering_finished(self, cluster_map):
        """聚类完成后的回调函数"""
        self.display_clusters(cluster_map)
        self.btn_cluster.setEnabled(True)
        self.clustering_thread = None
        QMessageBox.information(self, "提示", "聚类完成，PDF 报告已生成: clustering_report.pdf")

    def on_clustering_error(self, error_msg):
        """聚类出错时的回调函数"""
        QMessageBox.critical(self, "错误", error_msg)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备就绪")
        self.btn_cluster.setEnabled(True)
        self.clustering_thread = None

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
