import cv2
import re

pr = '''
Analyze the given image to determine the density of humans and the complexity of the scene. \
If the image shows a low-density scenario with few people, recommend a higher confidence threshold ('conf') value to reduce false detections. \
If the image shows a high-density scenario with a crowd of people, suggest a lower conf value to ensure all individuals are detected, even if it results in more false positives. \
Please provide the appropriate conf value based on this analysis. \n

conf: [Model will output a specific confidence threshold value between 0.4 and 0.8] \n
'''

def extract_number(sentence):
    # 正则表达式匹配0~1之间的数字，包括小数
    match = re.search(r'\b0(?:\.\d+)?|1(?:\.0*)?\b', sentence)
    if match:
        return float(match.group())
    return None

def tailor(v10,model,tokenizer,img):

    query = tokenizer.from_list_format([
        {'image': img},
        {'text': pr},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    conf_threshold = extract_number(response)

    threshold = (1-conf_threshold)*0.5 if conf_threshold is not None else 0.25

    # Run inference on 'bus.jpg' with arguments
    results = v10.predict(img, save=False, imgsz=[928,1600], conf=threshold)

    class_names = v10.names

    # 加载原始图像
    original_img = cv2.imread(img)
    tailor_dict = []

    for result in results:
        for i, (box, cls_id) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):

            if class_names[int(cls_id)]!='person':
                continue

            d = {}
            # 提取每个框的坐标
            x1, y1, x2, y2 = map(int, box)
            
            # 裁剪图像
            cropped_img = original_img[y1:y2, x1:x2]

            # 保存裁剪的图像
            output_path = f"Tailor/cropped_{i}.jpg"
            cv2.imwrite(output_path, cropped_img)

            d['path'] = output_path
            d[output_path] = [x1,y1,x2,y2]
            tailor_dict.append(d)

    return threshold, tailor_dict