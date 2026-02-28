# # -*- coding: UTF-8 -*-
# import argparse
# # import time
# from pathlib import Path
# import sys
# import os

# import numpy as np
# import cv2
# import torch
# # import torch.backends.cudnn as cudnn
# # from numpy import random
# import copy
# import warnings

# from models.experimental import attempt_load
# from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path

# import face_align


# warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# # 加载模型权重
# def load_model(weights, device):
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     return model


# # 将一个图像中的一组关键点坐标（通常用于表示面部或其他物体的特征点）从一种尺寸缩放到另一种尺寸。
# def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
#     coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
#     coords[:, :10] /= gain
#     # clip_coords(coords, img0_shape)
#     coords[:, 0].clamp_(0, img0_shape[1])  # x1
#     coords[:, 1].clamp_(0, img0_shape[0])  # y1
#     coords[:, 2].clamp_(0, img0_shape[1])  # x2
#     coords[:, 3].clamp_(0, img0_shape[0])  # y2
#     coords[:, 4].clamp_(0, img0_shape[1])  # x3
#     coords[:, 5].clamp_(0, img0_shape[0])  # y3
#     coords[:, 6].clamp_(0, img0_shape[1])  # x4
#     coords[:, 7].clamp_(0, img0_shape[0])  # y4
#     coords[:, 8].clamp_(0, img0_shape[1])  # x5
#     coords[:, 9].clamp_(0, img0_shape[0])  # y5
#     return coords


# # 给图片加上 面部特征点（landmarks）、边界框和置信度
# def show_results(img, xyxy, conf, landmarks, class_num):
#     h, w, c = img.shape
#     tl = 1 or round(0.02 * (h + w) / 2) + 1  # line/font thickness
#     x1 = int(xyxy[0])
#     y1 = int(xyxy[1])
#     x2 = int(xyxy[2])
#     y2 = int(xyxy[3])
#     img = img.copy()

#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

#     clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

#     for i in range(5):
#         point_x = int(landmarks[2 * i])
#         point_y = int(landmarks[2 * i + 1])
#         cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

#     tf = max(tl - 1, 1)  # font thickness
#     label = str(conf)[:5]
#     cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#     return img


# # 检测人脸
# def detect(model, source, device, save_dir, img_size, save_img):
#     # Load model
#     conf_thres = 0.2
#     iou_thres = 0.5
#     imgsz = (img_size, img_size)

#     # Directories
#     # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     save_dir = Path(save_dir) / source
#     Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

#     # Dataloader
#     if webcam:
#         print('loading streams:', source)
#         dataset = LoadStreams(source, img_size=imgsz)
#         bs = 1  # batch_size
#     else:
#         print('loading images', source)
#         dataset = LoadImages(source, img_size=imgsz)
#         bs = 1  # batch_size
#     vid_path, vid_writer = [None] * bs, [None] * bs

#     for path, im, im0s, vid_cap in dataset:

#         if len(im.shape) == 4:
#             orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
#         else:
#             orgimg = im.transpose(1, 2, 0)

#         orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
#         img0 = copy.deepcopy(orgimg)
#         h0, w0 = orgimg.shape[:2]  # orig hw
#         r = img_size / max(h0, w0)  # resize image to img_size
#         if r != 1:  # always resize down, only resize up if training with augmentation
#             interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
#             img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

#         imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

#         img = letterbox(img0, new_shape=imgsz)[0]
#         # Convert from w,h,c to c,w,h
#         img = img.transpose(2, 0, 1).copy()

#         img = torch.from_numpy(img).to(device)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Inference
#         pred = model(img)[0]

#         # Apply NMS
#         pred = non_max_suppression_face(pred, conf_thres, iou_thres)
#         print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
#         if len(pred[0]) == 0:
#             with open('/mnt/workspace/output1.txt', 'a') as file:
#                 file.write(f'image {len(pred[0])} faces, path: {path}\n')

#         # 初始化变量，用于存储最大人脸的面积和对应的人脸图像路径
#         max_area = 0
#         max_face_img = None
#         max_face_path = None

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image

#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(Path(save_dir) / p.name)  # im.jpg

#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class

#                 det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

#                 for j in range(det.size()[0]):
#                     xyxy = det[j, :4].view(-1).tolist()
#                     conf = det[j, 4].cpu().numpy()
#                     landmarks = det[j, 5:15].view(-1).tolist()
#                     class_num = det[j, 15].cpu().numpy()

#                     # im0 = show_results(im0, xyxy, conf, landmarks, class_num)

#                     # 解包坐标
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     # # 裁剪面部区域
#                     # face_img = im0[y1:y2, x1:x2]
#                     # # 保存裁剪出来的面部图像
#                     # cv2.imwrite(save_path, face_img)
                    
#                     # 计算人脸面积
#                     face_area = (x2 - x1) * (y2 - y1)

#                     # 如果当前人脸面积大于已存储的最大人脸面积，则更新最大人脸面积和图像
#                     if face_area > max_area:
#                         max_area = face_area
#                         max_face_img = im0[y1:y2, x1:x2]
#                         max_face_path = save_path

#                 # 在循环结束后，保存最大的人脸图像
#                 if max_face_img is not None and max_face_path is not None:
#                     cv2.imwrite(max_face_path, max_face_img)


#             # cv2.imshow('result', im0)
#             # key = cv2.waitKey(0) & 0xFF
#             # if key == ord("q"):
#                 # break

#             # # Save results (image with detections)
#             # if save_img:
#             #     if dataset.mode == 'image':
#             #         cv2.imwrite(save_path, im0)
#             #     else:  # 'video' or 'stream'
#             #         if vid_path[i] != save_path:  # new video
#             #             vid_path[i] = save_path
#             #             if isinstance(vid_writer[i], cv2.VideoWriter):
#             #                 vid_writer[i].release()  # release previous video writer
#             #             if vid_cap:  # video
#             #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
#             #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             #             else:  # stream
#             #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
#             #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#             #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#             #         try:
#             #             vid_writer[i].write(im0)
#             #         except Exception as e:
#             #             print(e)


# """
# 执行命令:
#     cd face_detect/AnyFace/yolov5-face
#     python detect_face.py --weights weights/yolov5l6_best.pt --source [--数据集路径--] --img-size 224 --save-img
#     例:
#     python detect_face.py --weights weights/yolov5l6_best.pt --source CaVI_Dataset/Caricature/Aamir_Khan --img-size 224 --save-img
#     python detect_face.py --weights weights/yolov5l6_best.pt --img-size 224 --save-img
    
# """
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--save-img', action='store_true', help='save results')
#     opt = parser.parse_args()

#     path = '/mnt/workspace/datasets/CaVI_Dataset/'
#     classes = ['Caricature', 'Real']
#     image_paths = []
#     for kind in classes:
#         temp = []
#         kind_path = os.path.join(path, kind)
#         for identity in os.listdir(kind_path):
#             paths = os.path.join('CaVI_Dataset', kind, identity)
#             temp.append(paths)
#         image_paths.append(temp)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model(opt.weights, device)
#     save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

#     for identities in image_paths:
#         for identity in identities:
#             detect(model, identity, device, save_dir, opt.img_size, opt.save_img)

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model = load_model(opt.weights, device)
#     # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
#     # # identity = 'CaVI_Dataset/Real/Johny_depp/Johny_depp_r_5.jpg'
#     # identity = 'CaVI_Dataset/Caricature/giggs/giggs_c_2.jpg'
#     # detect(model, identity, device, save_dir, opt.img_size, opt.save_img)


# -*- coding: UTF-8 -*-
import argparse
# import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
# import torch.backends.cudnn as cudnn
# from numpy import random
import copy
import warnings

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path

import face_align


warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# 加载模型权重
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


# 将一个图像中的一组关键点坐标（通常用于表示面部或其他物体的特征点）从一种尺寸缩放到另一种尺寸。
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


# 给图片加上 面部特征点（landmarks）、边界框和置信度
def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.02 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


# 得到校正后的图像
def get_norm_crop(landmark, im):
    image_size = 112  # 处理后大小
    align_mode = 'arcface'  # 参考点选择
    # 判断是否检测到了，检测到进行根据五个特征点相似变换
    _landmark = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]  # 特征点
    _landmark = np.array(_landmark, dtype=float)
    # 选取参考点，进行相似性变换
    warped = face_align.norm_crop(im, landmark=_landmark, image_size=image_size, mode=align_mode)
    return warped


# 检测人脸
def detect(model, source, device, save_dir, img_size, save_img):
    # Load model
    conf_thres = 0.2
    iou_thres = 0.5
    imgsz = (img_size, img_size)

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir) / source
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)

        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
        if len(pred[0]) == 0:
            with open('/mnt/workspace/output3.txt', 'a') as file:
                file.write(f'image {len(pred[0])} faces, path: {path}\n')

        # 初始化变量，用于存储最大人脸的面积和对应的人脸图像路径
        max_area = 0
        landmarks_ = None
        max_face_img = None

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    # im0 = show_results(im0, xyxy, conf, landmarks, class_num)

                    # 解包坐标
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 计算人脸面积
                    face_area = (x2 - x1) * (y2 - y1)

                    # 如果当前人脸面积大于已存储的最大人脸面积，则更新最大人脸面积
                    if face_area > max_area:
                        # 更新最大人脸面积
                        max_area = face_area
                        landmarks_ = landmarks
                        # max_face_img = im0[y1:y2, x1:x2]
                        max_face_img = im0

                # 在循环结束后，对齐最大的人脸图像
                # cv2.imwrite(save_path, im0)
                if landmarks_ is not None:
                    warped_face = get_norm_crop(landmarks_, max_face_img)
                    cv2.imwrite(save_path, warped_face)


            # cv2.imshow('result', im0)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord("q"):
                # break

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         try:
            #             vid_writer[i].write(im0)
            #         except Exception as e:
            #             print(e)


"""
执行命令:
    cd face_detect/AnyFace/yolov5-face
    python detect_face.py --weights weights/yolov5l6_best.pt --source [--数据集路径--] --img-size 224 --save-img
    例:
    python detect_face.py --weights weights/yolov5l6_best.pt --source CaVI_Dataset/Caricature/Aamir_Khan --img-size 224 --save-img
    python detect_face.py --weights weights/yolov5l6_best.pt --img-size 224 --save-img
    
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    opt = parser.parse_args()

    path = '/mnt/workspace/datasets/CaVI_Dataset/'
    classes = ['Caricature', 'Real']
    image_paths = []
    for kind in classes:
        temp = []
        kind_path = os.path.join(path, kind)
        for identity in os.listdir(kind_path):
            paths = os.path.join('CaVI_Dataset', kind, identity)
            temp.append(paths)
        image_paths.append(temp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    for identities in image_paths:
        for identity in identities:
            detect(model, identity, device, save_dir, opt.img_size, opt.save_img)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model(opt.weights, device)
    # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # # identity = 'CaVI_Dataset/Real/Johny_depp/Johny_depp_r_5.jpg'
    # # identity = 'CaVI_Dataset/Caricature/giggs/giggs_c_2.jpg'
    # identity = 'CaVI_Dataset/Real/Aamir_Khan'
    # detect(model, identity, device, save_dir, opt.img_size, opt.save_img)



