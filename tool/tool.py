import os
import cv2
import numpy as np

def intersect(box,boxes,width,height):
    '''
    a----b
    |    |
    c----d
    '''
    a, b, c, d, e, f, g, h = 1, 1, 1, 1, 1, 1, 1, 1

    x1, y1, x2, y2 = box

    # 存储 x 和 y 轴上的所有值
    x, y = [], []

    x.append(x1)
    x.append(x2)
    y.append(y1)
    y.append(y2)

    for i, bb in enumerate(boxes):

        x3, y3, x4, y4 = bb

        x.append(x3)
        x.append(x4)
        y.append(y3)
        y.append(y4)

        if x3 <= x1 <= x4 and y3 <= y1 <= y4:   # 左上角有相交 
            a = 0  

        if x3 <= x2 <= x4 and y3 <= y1 <= y4:   # 右上角有相交 
            b = 0  
  
        if x3 <= x1 <= x4 and y3 <= y2 <= y4:   # 左下角有相交 
            c = 0            

        if x3 <= x2 <= x4 and y3 <= y2 <= y4:   # 右下角有相交 
            d = 0  

        if x3 <= x1 <= x4 and y1 <= y3 <= y4 <= y2:   # x1夹心
            e = 0

        if x3 <= x2 <= x4 and y1 <= y3 <= y4 <= y2:   # x2夹心
            f = 0
        
        if y3 <= y1 <= y4 and x1 <= x3 <= x4 <= x2:   # y1夹心
            g = 0

        if y3 <= y2 <= y4 and x1 <= x3 <= x4 <= x2:   # y2夹心
            h = 0


    x.sort()
    y.sort()

    if a and c and e:   # 左上角和左下角无相交且夹心
        t = x.index(x1)
        if t == 0:  # x1是最小值
            x1 = 0
        else:
            x1 = x[t-1]

    if a and b and g:     # 左上角和右上角无相交且夹心
        t = y.index(y1)
        if t == 0: 
            y1 = 0
        else:
            y1 = y[t-1]

    if b and d and f:     # 右上角和右下角无相交且夹心
        t = x.index(x2)
        if t == len(x)-1:  # x2是最大值
            x2 = width
        else:
            x2 = x[t+1]
    
    if c and d and h:     # 左下角和右下角无相交且夹心
        t = y.index(y2)
        if t == len(y)-1:  # y2是最大值
            y2 = height
        else:
            y2 = y[t+1]

    return x1, y1, x2, y2
    
def expand_boxes(boxes, img_width, img_height):
    '''
    左上角(x1,y1)，右上角(x2,y1)
    左下角(x1,y2)，右下角(x2,y2)
    '''
    
    expanded_boxes = []

    for i, box in enumerate(boxes):
        # print(box)
        x1, y1, x2, y2 = box

        boxesss = boxes[:i]+boxes[i+1:]


        x1, y1, x2, y2 = intersect(box, boxesss, img_width, img_height)

        
        expanded_boxes.append((x1, y1, x2, y2))
    
    return expanded_boxes


# 判断文件夹是否为空
def is_file_empty(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            return len(content) == 0
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

