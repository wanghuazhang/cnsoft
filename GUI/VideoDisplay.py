import cv2
import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from yolov3_deepsort.yolo_video import detectAPI

class Display:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd

        # 默认视频源为相机
        self.ui.radioButtonCam.setChecked(True)
        self.isCamera = True

        # 信号槽设置
        ui.Open.clicked.connect(self.Open)
        ui.Close.clicked.connect(self.Close)
        ui.radioButtonCam.clicked.connect(self.radioButtonCam)
        ui.radioButtonFile.clicked.connect(self.radioButtonFile)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def radioButtonCam(self):
        self.isCamera = True
        self.stopEvent.set()

    def radioButtonFile(self):
        self.isCamera = False
        self.stopEvent.set()

    def Open(self):
        self.stopEvent.clear()
        self.ui.textBrowser.setText("开始识别")
        #这里需要改为调用深度学习框架
        if not self.isCamera:
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, 'Choose file', '', '*.*')

            input_path = 'yolov3_deepsort/img/video-02.mp4'
            output_path = 'yolov3_deepsort/output'
            # detectAPI(input_path, output_path)
            self.fileName='yolov3_deepsort/output/output_video-02.mp4'
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = cv2.VideoCapture(0)


        # 创建新线程显示视频信息
        self.th = threading.Thread(target=self.Display)
        self.th.start()

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def Display(self):
        self.ui.Open.setEnabled(False)
        self.ui.Close.setEnabled(True)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            if frame is None:
                self.stopEvent.clear()
                self.ui.DisplayLabel.clear()
                self.ui.Close.setEnabled(False)
                self.ui.Open.setEnabled(True)
            if success is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                break

            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            img = img.scaled(576, 324)
            self.ui.DisplayLabel.setPixmap(QPixmap.fromImage(img))

            # 调整速率
            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.DisplayLabel.clear()
                self.ui.Close.setEnabled(False)
                self.ui.Open.setEnabled(True)
                break