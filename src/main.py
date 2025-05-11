import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import ImageClusteringApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageClusteringApp()
    win.show()
    sys.exit(app.exec_())
