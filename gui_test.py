import sys
from PySide6.QtWidgets import QApplication, QMainWindow


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.path = ""
        self.setWindowTitle('Drag and Drop')
        self.setGeometry(300, 300, 400, 300)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()

        # 打印文件路径
        # print(file_path)
        self.path = file_path
        # 关闭窗口
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    # sys.exit(app.exec())
    app.exec()
    print("1")