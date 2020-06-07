import cv2
import threading
import json
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from win32api import Sleep

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


            if self.fileName:
                # input_path = 'yolov3_deepsort/img/video-02.mp4'
                print("开始目标检测：")
                output_path = 'yolov3_deepsort/output'
                detectAPI(self.fileName, output_path)
                print("目标检测完毕！")
                # self.fileName='yolov3_deepsort/output/output_video-02.mp4'
                filename = (self.fileName.split('/')[-1]).split('.')[0]
                #记录输出的track、plate对应json的文件路径
                self.output_track_path = 'yolov3_deepsort/output/' + 'output_' + filename + '/' + 'output_track_' + filename + '.json'
                self.output_plate_path = 'yolov3_deepsort/output/' + 'output_' + filename + '/' + 'output_plate_' + filename + '.json'
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

        with open(self.output_track_path, 'r', encoding='utf-8') as f:
            jsondatas = json.load(f)
            frame_index = -1

            while self.cap.isOpened():
                frame_index = frame_index + 1
                success, frame = self.cap.read()
                # RGB转BGR
                if frame is None:
                    self.stopEvent.clear()
                    self.ui.DisplayLabel.clear()
                    self.ui.Close.setEnabled(False)
                    self.ui.Open.setEnabled(True)
                    break
                if success is True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    break

                jsondata = jsondatas[frame_index]

                #Todo 在右边栏输入推理信息

                self.ui.textBrowser.append(jsondata['inference'])

                for item in jsondata['body']:
                    bbox = item['box']
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    content = str(item['trackerID']) + ": " + item['class'] + ' ' + item['confidence'] + ' ' + item['speed']
                    cv2.putText(frame, content, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                img = img.scaled(self.ui.DisplayLabel.width(), self.ui.DisplayLabel.height())
                self.ui.DisplayLabel.setPixmap(QPixmap.fromImage(img))

                #cv2.waitKey(40)
                #调整速率
                if self.isCamera:
                    cv2.waitKey(1)
                else:
                    Sleep(int(1000 / self.frameRate))
                    #cv2.waitKey(int(1000 / self.frameRate))

                # 判断关闭事件是否已触发
                if True == self.stopEvent.is_set():
                    # 关闭事件置为未触发，清空显示label
                    self.stopEvent.clear()
                    self.ui.DisplayLabel.clear()
                    self.ui.Close.setEnabled(False)
                    self.ui.Open.setEnabled(True)
                    break
