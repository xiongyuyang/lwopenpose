import cv2
from PySide6.QtWidgets import QWidget, QApplication, QLineEdit, QMainWindow, QTextBrowser, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap
import sys

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


def split_video(video_path, num_frames):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频总帧数和每帧的宽高信息
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算等距拆分后每个图片的间隔帧数
    interval = int(total_frames / num_frames)

    frames = []  # 存储拆分后的图片列表

    for i in range(num_frames):
        # 设置视频读取指针位置
        video.set(cv2.CAP_PROP_POS_FRAMES, i * interval)

        # 读取当前帧
        ret, frame = video.read()

        if ret:
            frames.append(frame)
        else:
            break

    # 关闭视频文件
    video.release()

    return frames


# 测试函数
app = QApplication(sys.argv)
window = Window()
window.show()
app.exec()
video_path =window.path
num_frames = 5

frames_list = split_video(video_path, num_frames)
print(len(frames_list))  # 输出拆分后的图片数量