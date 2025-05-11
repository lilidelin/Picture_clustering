# Picture_clustering

image-clustering-app/
├── src/                     # 源代码目录
│   ├── main.py              # 程序入口点
│   ├── ui/                  # 界面相关代码
│   │   ├── main_window.py   # 主窗口类
│   │   ├── widgets/         # 自定义组件
│   │   └── resources/       # 界面资源（图标、样式等）
│   ├── models/              # 模型相关代码
│   │   ├── feature_extractors/  # 特征提取模型
│   │   │   ├── resnet.py    # ResNet50和ResNet18特征提取
│   │   │   └── vit16.py     # Vit16特征提取
│   │   ├── clustering/      # 聚类算法
│   │   │   ├── kmeans.py    # KMeans聚类
│   │   │   └── dbscan.py    # DBSCAN聚类
│   │   └── model_factory.py # 模型工厂类（创建模型实例）
│   ├── utils/               # 工具函数
│   │   ├── file_io.py       # 文件操作
│   │   ├── image_processing.py # 图像处理
│   │   ├── visualization.py # 可视化工具
│   │   ├── t_SNE.py         # 可视化特征
│   │   └── clustering_thread.py # 程序线程
│   └── config/              # 配置文件
│       └── default_config.py # 默认配置
├── tests/                   # 测试代码
├── docs/                    # 文档
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明