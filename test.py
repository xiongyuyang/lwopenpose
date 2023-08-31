import cv2
import tkinter as tk


class CameraApp:
    def __init__(self, window):
        self.window = window
        self.camera = None

        # 创建开始按钮
        self.start_button = tk.Button(window, text="开始", command=self.start_camera)
        self.start_button.pack()

        # 创建结束按钮
        self.stop_button = tk.Button(window, text="结束", command=self.stop_camera)
        self.stop_button.pack()

    def start_camera(self):
        if not self.camera:
            # 打开摄像头并获取视频流
            self.camera = cv2.VideoCapture(0)

        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            # 在窗口中显示图像帧
            cv2.imshow("Camera", frame)

            # 监听键盘事件，如果按下q键则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop_camera(self):
        if self.camera:
            # 停止获取视频流并释放资源
            self.camera.release()
            cv2.destroyAllWindows()

        self.camera = None


window = tk.Tk()
app = CameraApp(window)
window.mainloop()