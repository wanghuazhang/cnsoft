# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import json
import math
import time
from timeit import default_timer as timer
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from qtpy import QtCore

from PlateRecognition.plateRecognition import *
from yolov3_deepsort.deep_sort import nn_matching, preprocessing
from yolov3_deepsort.deep_sort.detection import Detection
from yolov3_deepsort.deep_sort.tracker import Tracker
from yolov3_deepsort.tools import generate_detections as gdet
from yolov3_deepsort.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolov3_deepsort.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class YOLO(object):
    _defaults = {
        "model_path": 'yolov3_deepsort/model_data/yolo.h5',
        "anchors_path": 'yolov3_deepsort/model_data/yolo_anchors.txt',
        "classes_path": 'yolov3_deepsort/model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            # self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            self.yolo_model = load_model(model_path, compile=False)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, )) #实例化了一个输入占位符
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            # resize输入的图片
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255. #把data归一化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def detect_image_deepsort(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='yolov3_deepsort/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        return_boxs = []
        labels = []
        detect_class = ['car', 'bus', 'person', 'motorbike', 'truck']
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            #对检查目标先过滤一遍
            if predicted_class not in detect_class:
                continue
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            #添加车牌检测内容
            myclass = ['car', 'bus']

            #Todo 优化车牌识别
            #对车牌检测部分进行优化，添加检测条件：1/4 < x < 3/4 and 1/2 < y < 4/5
            # if image.size[0] * 1 / 3 < x < image.size[0] * 2 / 3 and image.size[1] * 3 / 5 < y < image.size[1] * 4 / 5:
            #     if w > 120 and h > 120:
            #         if predicted_class in myclass:
            #             imgPlate = image.crop((x, y, x + w, y + h))
            #             imgNp = np.asarray(imgPlate)
            #             carPlate = recognize_plate(imgNp)
            #             print(carPlate)
            #             if carPlate:
            #                 pstr = carPlate[0][0]
            #                 confidence = str(round(carPlate[0][1],3))
            #                 label += '/n' + pstr + ': ' + confidence

            return_boxs.append([x, y, w, h])
            # print(predicted_class, str([x, y, w, h]))
            labels.append(label)

        return return_boxs, labels

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    # Definition of the parameters
    max_cosine_distance = 0.5 #0.3
    nn_budget = None
    nms_max_overlap = 1.0 #1.0
    # deep_sort
    model_filename = 'yolov3_deepsort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    writeVideo_flag = False

    if video_path:
        vid = cv2.VideoCapture(video_path)
        file_name = (video_path.split('/')[-1]).split('.')[0]
        suffix = (video_path.split('/')[-1]).split('.')[1]
    else:
        vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    intervalTime = 1 / video_fps
    isOutput = True if output_path != "" else False

    frame_index = -1
    #要保存的json信息
    track_json_result = []
    plate_json_result = []

    # if isOutput:
    #     #注意保存视频文件时需要下载对应的编码库，如64位的Python，Windows，就下载openh264-1.7.0-win64.dll.bz2
    #     #下载完之后，解压到对应解释器所在目录，链接：https://github.com/cisco/openh264/releases
    #     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    #     print(output_path)
    #     output_file = output_path+ '/output_' + file_name + '.' + suffix
    #     print(output_file)
    #     out = cv2.VideoWriter(output_file, video_FourCC, video_fps, video_size)
    #     list_file = open(output_path + '/detection_' +  file_name + '.txt', 'w')
    #     frame_index = -1

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    #把False改为True即可以进行车牌检测
    #Todo 提高车牌检测效率，多线程？读取同一个视频文件资源竞争怎么解决？
    while False:
        plate_json_dict = {}
        frame_index = frame_index + 1
        if not frame_index % 2:
            continue
        ret, frame = vid.read()
        if ret != True:
            break
        # 车牌检测
        carPlate = recognize_plate(frame)
        for plate in carPlate:
            bbox = plate[2]
            cv2.putText(frame, plate[0], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            plate_json_dict['frame_index'] = frame_index
            plate_json_dict['plate'] = plate[0]
            plate_json_dict['box'] = []
            plate_json_dict['box'].append(bbox[0])
            plate_json_dict['box'].append(bbox[1])
            plate_json_dict['box'].append(bbox[2])
            plate_json_dict['box'].append(bbox[3])
        if isOutput:
            print(frame_index, plate_json_dict)
            plate_json_result.append(plate_json_dict)

    frame_index = -1

    while True:
        track_json_dict = {}
        frame_index = frame_index + 1
        # if not frame_index % 5:
        #     continue
        ret, frame = vid.read()
        if ret != True:
            break
        curr_time = timer()
        #exec_time代表前后两帧运行时间
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        print(frame_index, fps)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, labels = yolo.detect_image_deepsort(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature, lable, intervalTime) for bbox, feature, lable in zip(boxs, features, labels)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        track_json_dict['frame_index'] = frame_index
        track_json_dict['body'] = []
        #计算车流量、人流量
        personCnt = 0
        carCnt = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            content = str(track.track_id) + ": " + track.label
            cv2.putText(frame, content, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            #保存track的json信息
            tmptrack = {}

            tmptrack['trackerID'] = str(track.track_id)
            tmptrack['class'] = (track.label.split(' ')[0])
            tmptrack['confidence'] = (track.label.split(' ')[1])
            tmptrack['speed'] = str(track.speed)
            # detect_class = ['car', 'bus', 'person', 'motorbike', 'truck']
            if tmptrack['class'] in ['car', 'bus', 'truck']:
                carCnt += 1
            else:
                personCnt += 1
            tmptrack['box'] = []
            tmptrack['box'].append(bbox[0])
            tmptrack['box'].append(bbox[1])
            tmptrack['box'].append(bbox[2])
            tmptrack['box'].append(bbox[3])
            print(tmptrack)
            track_json_dict['body'].append(tmptrack)

        track_json_dict['personCnt'] = str(personCnt)
        track_json_dict['carCnt'] = str(carCnt)

        inference = ""
        inference += "当前车辆数目：" + str(carCnt) + '\n'
        if carCnt <= 5:
            inference += "车流较少"
        elif carCnt <= 15:
            inference += "车流适中"
        else:
            inference += "车流较大"
        inference += '\n'
        inference += "当前行人数目：" + str(personCnt) + '\n'
        if personCnt <= 5:
            inference += "人流较少"
        elif personCnt <= 15:
            inference += "人流适中"
        else:
            inference += "人流较大"
        inference += '\n'
        print(inference)
        track_json_dict['inference'] = inference


        #车牌检测
        # carPlate = recognize_plate(frame)
        # for plate in carPlate:
        #     bbox = plate[2]
        #     cv2.putText(frame, plate[0], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        #     plate_json_dict['frame_index'] = frame_index
        #     plate_json_dict['plate'] = plate[0]
        #     plate_json_dict['box'] = bbox

        #暂时采取只将文件保存到本地的方法，不实时的show出来
        # cv2.imshow('result', frame)

        #直接保存视频文件
        # if isOutput:
        # # save a frame
        #     out.write(frame)
        #     list_file.write(str(frame_index) + ' ')
        #     if len(boxs) != 0:
        #         for i in range(0, len(boxs)):
        #             list_file.write(
        #                 str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
        #     list_file.write('\n')

        if isOutput:
            track_json_result.append(track_json_dict)
            # plate_json_result.append(plate_json_dict)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if isOutput:
        output_folder = output_path + '/output_'+file_name
        if not os.path.exists(output_folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(output_folder)
        output_track_json_file = output_folder + '/output_track_' + file_name + '.json'
        output_plate_json_file = output_folder + '/output_plate_' + file_name + '.json'
        print(output_track_json_file)
        with open(output_track_json_file, 'w', encoding='utf-8') as f:
            json.dump(track_json_result, f, ensure_ascii=False)
        with open(output_plate_json_file, 'w', encoding='utf-8') as f:
            json.dump(plate_json_result, f, ensure_ascii=False)

    # while True:
    #     return_value, frame = vid.read()
    #     image = Image.fromarray(frame)
    #     image = yolo.detect_image(image)
    #     result = np.asarray(image)
    #     curr_time = timer()
    #     exec_time = curr_time - prev_time
    #     prev_time = curr_time
    #     accum_time = accum_time + exec_time
    #     curr_fps = curr_fps + 1
    #     if accum_time > 1:
    #         accum_time = accum_time - 1
    #         fps = "FPS: " + str(curr_fps)
    #         curr_fps = 0
    #     cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.50, color=(255, 0, 0), thickness=2)
    #     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #     cv2.imshow("result", result)
    #     if isOutput:
    #         out.write(result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    yolo.close_session()

def detect_camera(yolo, th):
    # Definition of the parameters
    max_cosine_distance = 0.5 #0.3
    nn_budget = None
    nms_max_overlap = 1.0 #1.0
    model_filename = 'yolov3_deepsort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    # vid = cv2.VideoCapture(0)
    vid = th.th.cap
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_fps = vid.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 5

    intervalTime = 1 / video_fps
    frame_index = -1

    while True:
        frame_index += 1
        return_value, frame = vid.read()
        # RGB转BGR
        if frame is None:
            th.th.stopEvent.clear()
            th.th.ui.DisplayLabel.clear()
            th.th.ui.Close.setEnabled(False)
            th.th.ui.Open.setEnabled(True)

        if return_value is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            break

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, labels = yolo.detect_image_deepsort(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature, lable, intervalTime) for bbox, feature, lable in
                      zip(boxs, features, labels)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # 计算车流量、人流量
        personCnt = 0
        carCnt = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            content = str(track.track_id) + ": " + track.label
            cv2.putText(frame, content, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            tmptrack = {}

            tmptrack['trackerID'] = str(track.track_id)
            tmptrack['class'] = (track.label.split(' ')[0])
            tmptrack['confidence'] = (track.label.split(' ')[1])
            tmptrack['speed'] = str(track.speed)
            # detect_class = ['car', 'bus', 'person', 'motorbike', 'truck']
            if tmptrack['class'] in ['car', 'bus', 'truck']:
                carCnt += 1
            else:
                personCnt += 1
            tmptrack['box'] = []
            tmptrack['box'].append(bbox[0])
            tmptrack['box'].append(bbox[1])
            tmptrack['box'].append(bbox[2])
            tmptrack['box'].append(bbox[3])
            print(tmptrack)

        inference = ""
        inference += "当前车辆数目：" + str(carCnt) + '\n'
        if carCnt <= 5:
            inference += "车流较少"
        elif carCnt <= 15:
            inference += "车流适中"
        else:
            inference += "车流较大"
        inference += '\n'
        inference += "当前行人数目：" + str(personCnt) + '\n'
        if personCnt <= 5:
            inference += "人流较少"
        elif personCnt <= 15:
            inference += "人流适中"
        else:
            inference += "人流较大"
        inference += '\n'
        print(inference)
        th.changeText.emit("开始识别\n" + '第' + str(frame_index) + '帧')
        th.changeText.emit(inference)
        th.th.cursor = th.th.ui.textBrowser.textCursor()
        th.th.ui.textBrowser.moveCursor(th.th.cursor.End)

        # 暂时采取只将文件保存到本地的方法，不实时的show出来
        # cv2.imshow('result', frame)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img = img.scaled(th.th.ui.DisplayLabel.width(), th.th.ui.DisplayLabel.height())
        th.th.ui.DisplayLabel.setPixmap(QPixmap.fromImage(img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if True == th.th.stopEvent.is_set():
            # 关闭事件置为未触发，清空显示label
            th.th.stopEvent.clear()
            th.th.ui.DisplayLabel.clear()
            th.th.ui.Close.setEnabled(False)
            th.th.ui.Open.setEnabled(True)
            break

    yolo.close_session()