import os
import re
import time
import shutil
import pandas as pd
from module.summary import summary
from tool.tool import is_file_empty
from eval.eval import calculate_iou_euclidean
from module.img import draw_box,images_to_video
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():

    # 读取 Excel 文件
    filenames = os.listdir("nuScenes/Trainval/CAM_FRONT_RIGHT")
    file_path = 'nuScenes/nus_trainval.xlsx'
    sheet_name = 'Sheet1'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # VLM 参数
    device = "cuda:1"
    tokenizer = AutoTokenizer.from_pretrained("Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()

    k = 1 # 视频序列号

    # 评估指标
    s_TP, s_FN, s_FP, s_TN, s_IDS, s_GT, s_dn, s_rate, s_time, s_len = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # 生成结果
    for index, row in df.iterrows():

        # 评估指标
        TP, FN, FP, TN, IDS = 0, 0, 0, 0, 0

        # 获取GT的值和列表
        with open('nuScenes/Trainval/seq/'+str(k)+'.txt', 'r', encoding='utf-8') as file:
            # 将每行的内容存储在列表中
            seqs = file.readlines()

        # 去除每行末尾的换行符（如果需要）
        seqs = [seq.strip() for seq in seqs]

        # 用于评估指标
        GT = len(seqs)

        # os.mkdir("Output"+str(k))

        start = row['Start']
        end = row['End']

        # 总图片帧数
        s_len = s_len + len(filenames[filenames.index(start):filenames.index(end)+1])

        # Summary Module
        text = row['Prompt']
        start_time = time.time()
        # info = summary(model,tokenizer,text)

        info, historyyy = model.chat(tokenizer, query="Extract key info from the 'text' to position the passenger, the response only include key info. text:\n"+text, history=None)

        # 遍历视频中的每个图片
        for link in filenames[filenames.index(start):filenames.index(end)+1]:
            # os.mkdir("Tailor")
            # 图片路径
            path = 'nuScenes/Trainval/CAM_FRONT_RIGHT/'+link

            # VLM Module
            flag = 0    #用于判断是否检测到物体

            query = tokenizer.from_list_format([
                {'image': path},
                {'text': info},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)

            # 因为他这里返回的坐标系不是正常的像素坐标系，而是将宽高都变为1000后的像素坐标，因此，x轴需*1.6，y轴需*0.8

            image = tokenizer.draw_bbox_on_latest_picture(response, history)

            # 定义Ground Truth中是否有Box
            flag_box = 0   # Ground Truth中没有Box
            if link.split('.')[0] in seqs:
                flag_box = 1    

            # 如果image不为空，则视为检测到Box
            if image:
                # 提取框的2D框的坐标
                match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', response)
                coordinates = [int(int(match.group(1))*1.6), int(int(match.group(2))*0.8), int(int(match.group(3))*1.6), int(int(match.group(4))*0.8)]
                # draw_box(link,coordinates,k)
                # image.save('result/baseline/'+ str(k) + '_' + link + '.jpg')

                # 判断Ground Truth中是否有Box
                if not flag_box:    # Ground Truth中没有Box
                    FP = FP + 1

                else:
                    # 计算IoU
                    iou, distance = calculate_iou_euclidean(coordinates,link,path)
                    if iou > 0.5:
                        TP = TP + 1
                        s_dn = s_dn + distance
                    else:
                        FN = FN + 1
                        IDS = IDS + 1

            else: # 未检测到物体
                # shutil.copyfile(path, 'Output'+ str(k) + '/' + link)

                if not flag_box:    # Ground Truth中没有Box
                    TN = TN + 1
                else:
                    FN = FN + 1

            # shutil.rmtree("Tailor")
        
        s_time = s_time + time.time() - start_time

        # 将结果保存为视频
        # images_to_video("Output"+str(k), "output"+str(k)+'.mp4', 5)
        # shutil.rmtree("Output"+str(k))
        k = k + 1

        if TP/(TP+FN) >= 0.5:
            s_rate = s_rate + 1

        
        with open("qwen.txt","a") as f:
            f.write(str(k-1)+' TP:'+str(TP)+" FN:"+str(FN)+" FP:"+str(FP)+" TN:"+str(TN)+" IDS:"+str(IDS)+" GT:"+str(GT)+'\n')

        # 评价指标
        s_TP = s_TP + TP
        s_FN = s_FN + FN
        s_FP = s_FP + FP
        s_TN = s_TN + TN
        s_IDS = s_IDS + IDS
        s_GT = s_GT + GT

    
    FPS = 1000 / (s_time / s_len)
    MOTA = 1 - (s_FN+s_FP+s_IDS) / (s_TP + s_FN)
    MOTP = s_dn / s_TP
    Recall = s_TP / (s_TP + s_FN)
    Accuracy = (s_TP + s_TN) / (s_TP + s_FN + s_FP + s_TN)
    Success_Rate = s_rate / (k - 1)
    print(s_TP, s_FN, s_FP, s_TN, s_IDS, s_GT, FPS, MOTA, MOTP, Recall, Accuracy, Success_Rate)

    with open("qwen.txt","a") as f:
            f.write('s_TP:'+str(s_TP)+" s_FN:"+str(s_FN)+" s_FP:"+str(s_FP)+" s_TN:"+str(s_TN)+" s_IDS:"+str(s_IDS)+" s_GT:"+str(s_GT)+" FPS:"+str(FPS)+" MOTA:"+str(MOTA)+" MOTP:"+str(MOTP)+" Recall:"+str(Recall)+" Accuracy:"+str(Accuracy)+" Success_Rate:"+str(Success_Rate)+'\n')

if __name__ == '__main__':
    main()