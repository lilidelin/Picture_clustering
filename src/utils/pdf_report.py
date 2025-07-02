from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime
from PIL import Image as PILImage  # 避免与reportlab的Image冲突


def register_chinese_font():
    """注册中文字体，优先使用系统字体"""
    try:
        # 尝试注册常见中文字体
        pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))
        return 'SimHei'
    except:
        try:
            pdfmetrics.registerFont(TTFont('STHeiti', 'STHeiti Medium.ttc'))
            return 'STHeiti'
        except:
            try:
                pdfmetrics.registerFont(TTFont('WenQuanYi Micro Hei', 'wqy-microhei.ttc'))
                return 'WenQuanYi Micro Hei'
            except:
                # 如果没有中文字体，使用默认字体（中文可能显示异常）
                print("警告: 未找到中文字体，中文可能显示为黑框")
                return 'Helvetica'


def create_thumbnail(image_path, size=(100, 100)):
    """创建图片缩略图，保持原始宽高比"""
    try:
        img = PILImage.open(image_path)
        img.thumbnail(size, PILImage.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"创建缩略图失败: {image_path}, 错误: {e}")
        return None


def generate_pdf_report(cluster_map, image_folder, save_path, features, labels, cluster_model):
    """生成聚类分析PDF报告

    Args:
        cluster_map: 聚类结果字典 {标签: [图片文件名列表]}
        image_folder: 图片所在文件夹
        save_path: PDF保存路径
        features: 特征向量数组
        labels: 聚类标签数组
        cluster_model: 聚类模型实例
    """
    # 注册中文字体
    chinese_font = register_chinese_font()

    # 创建PDF文档
    doc = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4

    # 自定义样式
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Chinese', fontName=chinese_font, fontSize=12))

    # 封面
    doc.setFont(chinese_font, 28)
    doc.drawCentredString(width / 2, height - 120, '图片聚类分析报告')

    doc.setFont(chinese_font, 14)
    doc.drawCentredString(width / 2, height - 170, f'生成日期: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # 添加基本信息
    y = height - 240
    doc.setFont(chinese_font, 16)
    doc.drawString(100, y, '分析摘要')
    doc.setFont(chinese_font, 12)

    info_items = [
        f'总图片数量: {len(features)}',
        f'聚类算法: {type(cluster_model).__name__}',
        f'聚类数量: {len(cluster_map)}',
        f'特征提取模型: {getattr(cluster_model, "feature_model", "未知")}',
    ]

    for item in info_items:
        y -= 25
        doc.drawString(120, y, item)

    # 分页
    doc.showPage()

    # t-SNE可视化
    from src.utils.t_SNE import plot_tsne
    plot_tsne(features, labels, save_path='tsne_visualization.png')

    doc.setFont(chinese_font, 16)
    doc.drawString(100, height - 50, '聚类分布可视化 (t-SNE)')

    try:
        img = Image('tsne_visualization.png', width=450, height=350)
        img.drawOn(doc, 100, height - 400)
    except Exception as e:
        doc.drawString(100, height - 420, f"无法加载t-SNE图像: {str(e)}")

    # 添加聚类质量评估
    y = height - 450
    doc.setFont(chinese_font, 14)
    doc.drawString(100, y, '聚类质量评估')
    doc.setFont(chinese_font, 12)

    # 计算简单的聚类质量指标
    try:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(features, labels)
        doc.drawString(120, y - 25, f'轮廓系数: {silhouette_avg:.4f} (越接近1表示聚类效果越好)')
    except:
        doc.drawString(120, y - 25, '轮廓系数: 无法计算')

    doc.showPage()

    # 聚类详情
    doc.setFont(chinese_font, 16)
    doc.drawString(100, height - 50, '聚类结果详情')
    doc.showPage()

    # 每页显示的图片数量和布局参数
    images_per_row = 5  # 每行显示图片数
    rows_per_page = 6   # 每页显示行数
    thumbnail_size = 80  # 缩略图大小

    # 每个聚类占一页或多页
    for label, images in sorted(cluster_map.items()):
        # 计算当前聚类的图片总数和总页数
        total_images = len(images)
        images_per_page = images_per_row * rows_per_page
        total_pages = (total_images + images_per_page - 1) // images_per_page

        # 绘制聚类标题（第一页）
        doc.setFont(chinese_font, 18)
        doc.drawString(100, height - 80, f'聚类类别 {label} ({total_images}张图片)')
        current_y = height - 120  # 从标题下方开始绘制图片

        # 为每个聚类创建独立的页
        for page in range(total_pages):
            # 非第一页需要新建页并添加“续”标题
            if page > 0:
                doc.showPage()
                current_y = height - 50
                doc.setFont(chinese_font, 14)
                doc.drawString(100, current_y, f'聚类类别 {label} (续)')
                current_y -= 30  # 标题下方留出空间

            # 当前页显示的图片范围
            start_idx = page * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)

            # 创建图片网格
            for i in range(start_idx, end_idx):
                img_idx = i - start_idx
                row = img_idx // images_per_row
                col = img_idx % images_per_row

                # 计算图片位置
                x_pos = 50 + col * (thumbnail_size + 20)
                y_pos = current_y - (row + 1) * (thumbnail_size + 30)

                # 添加图片文件名
                img_name = images[i]
                doc.setFont(chinese_font, 9)
                doc.drawString(x_pos, y_pos + thumbnail_size + 5, img_name[:20] + ('...' if len(img_name) > 20 else ''))

                # 添加缩略图
                img_path = os.path.join(image_folder, img_name)
                try:
                    # 创建缩略图
                    thumbnail = create_thumbnail(img_path, size=(thumbnail_size, thumbnail_size))
                    if thumbnail:
                        # 保存临时缩略图文件（添加label作为前缀，确保唯一性）
                        temp_path = f"temp_thumb_{label}_{i}.jpg"
                        thumbnail.save(temp_path, "JPEG")

                        # 添加到PDF
                        img = Image(temp_path, width=thumbnail_size, height=thumbnail_size)
                        img.drawOn(doc, x_pos, y_pos)

                        # 删除临时文件
                        os.remove(temp_path)
                    else:
                        # 缩略图创建失败，显示占位符
                        doc.setFont(chinese_font, 9)
                        doc.drawString(x_pos, y_pos + thumbnail_size / 2, "图片加载失败")
                except Exception as e:
                    doc.setFont(chinese_font, 9)
                    doc.drawString(x_pos, y_pos + thumbnail_size / 2, f"错误: {str(e)[:15]}")

            # 更新当前y坐标，用于下一页的定位
            current_y = y_pos - 50

            # 页脚：页码信息（在每页底部添加）
            doc.setFont(chinese_font, 10)
            doc.drawCentredString(width / 2, 30, f"第 {page + 1}/{total_pages} 页")

        # 每个聚类结束后，确保下一个聚类从新页开始
        if label != list(cluster_map.keys())[-1]:
            doc.showPage()

    # 保存PDF
    doc.save()