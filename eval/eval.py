import cv2
import math
import numpy as np

def xywh2xyxy(x, w1, h1):
    label, x, y, w, h = x
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    return int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)

def calculate_iou_euclidean(box1, label_path,img_path):
    # 获取检测到的box
    x1_min, y1_min, x1_max, y1_max = box1

    # 获取ground truth box
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open('data/nuScenes/labels/'+label_path.replace('jpg','txt'), 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        # 绘制每一个目标
    for x in lb:
        # 反归一化并得到左上和右下坐标，画出矩形框
        x2_min, y2_min, x2_max, y2_max = xywh2xyxy(x, w, h)

    # 计算交集的坐标
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    # 计算交集区域的宽度和高度
    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    # 计算交集区域的面积
    inter_area = inter_width * inter_height

    # 计算每个Box的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算并集区域的面积
    union_area = area1 + area2 - inter_area

    # 计算IoU
    iou = inter_area / union_area

    # 计算欧氏距离
    x1_center = (x1_min + x1_max) / 2
    y1_center = (y1_min + y1_max) / 2
    
    x2_center = (x2_min + x2_max) / 2
    y2_center = (y2_min + y2_max) / 2
    
    # Calculate the Euclidean distance between the centers
    distance = math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)

    return iou, distance
