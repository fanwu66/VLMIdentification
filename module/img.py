import os
import cv2

def draw_box(link,x,k):
    # flag 判断该张图片中是否已经绘制了bbox
    x1,y1,x2,y2 = x
    src = 'Output'+ str(k) + '/' + link
    
    img = cv2.imread('nuScenes/Trainval/CAM_FRONT_RIGHT/'+link)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(src,img)


def images_to_video(image_folder, output_video, fps):
    # 获取图片文件夹中的所有图片文件名
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # 排序图片以确保按顺序合成视频

    # 读取第一张图片以获取视频的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放 VideoWriter 对象
    video.release()
    print(f"Video {output_video} has been created successfully.")

