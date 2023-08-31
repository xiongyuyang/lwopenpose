import argparse
import sys
from PySide6.QtWidgets import QWidget, QApplication, QLineEdit, QMainWindow, QTextBrowser, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap
import cv2
import numpy as np
import torch
import tkinter as tk
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


def split_video(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频总帧数和每帧的宽高信息
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    frames = []  # 存储拆分后的图片列表

    for i in range(total_frames):
        # 设置视频读取指针位置
        video.set(cv2.CAP_PROP_POS_FRAMES, i)

        # 读取当前帧
        ret, frame = video.read()

        if ret:
            frames.append(frame)
        else:
            break

    # 关闭视频文件
    video.release()
    return frames

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



class CameraApp:
    def __init__(self, window ,img_list):
        self.window = window
        self.camera = None
        # self.pose_list = pose_list
        self.img_list = img_list
        # 创建开始按钮
        self.start_button = tk.Button(window, text="开始", command=self.start_camera)
        self.start_button.pack()
        self.num = 0
        # 创建结束按钮
        self.stop_button = tk.Button(window, text="结束", command=self.stop_camera)
        self.stop_button.pack()

    def start_camera(self):
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("D:\人工智能学习资料\lightweight-human-pose-estimation.pytorch-master\_checkpoints\checkpoint_iter_370000.pth", map_location='cpu')
        load_state(net, checkpoint)
        if not self.camera:
            # 打开摄像头并获取视频流
            self.camera = cv2.VideoCapture(0)

        while True:
            ret, frame = self.camera.read()

            if not ret:
                break
            realtime_pose = get_single_pose(frame,net,height_size=256,cpu=True)
            standard_pose = get_single_pose(self.img_list[self.num],net,height_size=256,cpu=True)
            realtime_pose.draw_with_score(frame,standard_pose)
            standard_pose.draw(self.img_list[self.num])
            # imgs = np.hstack([frame, self.img_list[self.num]])
            cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('realtime', 500, 250)
            cv2.imshow('realtime', frame)
            cv2.namedWindow("standard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('standard', 500, 250)
            cv2.imshow("standard",self.img_list[self.num])
            print(self.num)
            self.num += 1
            # 在窗口中显示图像帧
            # cv2.imshow("Camera", frame)
            # print(realtime_pose)
            # 监听键盘事件，如果按下q键则退出循环
            if self.num >= len(self.img_list):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop_camera(self):
        self.num=0
        if self.camera:
            # 停止获取视频流并释放资源
            self.camera.release()
            cv2.destroyAllWindows()

        self.camera = None

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cpu()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def get_single_pose(image, net, height_size, cpu):
    net = net.eval()
    if not cpu:
        net = net.cpu()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    orig_img = image.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, image, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    if len(pose_entries) > 0:
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[0][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[0][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[0][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[0][18])
        # print(pose)
        return pose

    return None



def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cpu()

    stride = 8
    upsample_ratio = 4
    # realtime_pose = None
    num_keypoints = Pose.num_kpts
    previous_poses = []
    pose_list=[]
    img_list =[]
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            # pose.draw(img)
            pose_list.append(pose)
            pose.draw(img)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        img_list.append(img)
        # key = cv2.waitKey(delay)
        # if key == 27:  # esc
        #     return
        # elif key == 112:  # 'p'
        #     if delay == 1:
        #         delay = 0
        #     else:
        #         delay = 1
    return pose_list,img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=False,
                        default = "D:\人工智能学习资料\lightweight-human-pose-estimation.pytorch-master\_checkpoints\checkpoint_iter_370000.pth",
                        help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--video', type=str, default= "C:/Users/29943/Pictures/Camera Roll/WIN_20230830_21_06_55_Pro.mp4" , help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
    # if args.video == '' and args.images == '':
    #     raise ValueError('Either --video or --image has to be provided')
    img_list =[]
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    # frame_provider = VideoReader(window.path)
    # frame_provider = ImageReader(args.images)
    # if args.video != '':
    #     frame_provider = VideoReader(args.video)
    # else:
    #     args.track = 0
    # standard_pose = get_single_pose(img,net,args.height_size,args.cpu)
    # standard_pose_img = standard_pose.draw_single_pose(img)
    # standard_pose_list, standard_img_list= run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
    # print(len(standard_pose_list))
    # print(len(img_list))
    img_list =split_video(window.path)
    window = tk.Tk()
    app = CameraApp(window, img_list)
    window.mainloop()

