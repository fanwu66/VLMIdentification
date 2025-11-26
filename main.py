import os
import time
import shutil
import pandas as pd
from module.vlm import vlm
from ultralytics import YOLOv10
from module.summary import summary
from module.detection import tailor
from tool.tool import is_file_empty
from module.similarity import cal_similarity
from eval.eval import calculate_iou_euclidean
from module.img import draw_box,images_to_video
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():

    # 读取 Excel 文件
    filenames = os.listdir("data/nuScenes/image")
    file_path = 'data/nuScenes/image/nuScenes.xlsx'
    sheet_name = 'Sheet1'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # VLM 参数
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained("Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
    yolo10 = YOLOv10("yolov10x.pt")

    k = 1 # 视频序列号

    # 评估指标
    s_TP, s_FN, s_FP, s_TN, s_IDS, s_GT, s_dn, s_rate, s_time, s_len = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # 生成结果
    for index, row in df.iterrows():

        # 评估指标
        TP, FN, FP, TN, IDS = 0, 0, 0, 0, 0

        # 获取GT的值和列表
        with open('data/nuScenes/seq/'+str(k)+'.txt', 'r', encoding='utf-8') as file:
            # 将每行的内容存储在列表中
            seqs = file.readlines()

        # 去除每行末尾的换行符（如果需要）
        seqs = [seq.strip() for seq in seqs]

        # 用于评估指标
        GT = len(seqs)

        os.mkdir("Output"+str(k))

        start = row['Start']
        end = row['End']

        # 总图片帧数
        s_len = s_len + len(filenames[filenames.index(start):filenames.index(end)+1])

        # Summary Module
        text = row['Prompt']
        start_time = time.time()
        info = summary(model,tokenizer,text)
        # 遍历视频中的每个图片
        for link in filenames[filenames.index(start):filenames.index(end)+1]:
            v = ''
            os.mkdir("Tailor")
            # 图片路径
            path = 'data/nuScenes/image/'+link

            # a Module
            threshold, tailor_dicts = tailor(yolo10,model,tokenizer,path)

            # 定义conf 和 simility 
            v = v + str(threshold) + ' '

            # VLM Module
            flag = 0    #用于判断是否检测到物体

            for tailor_dict in tailor_dicts:
            
                response = vlm(model,tokenizer,tailor_dict['path'])

                # 计算相似度
                similarity = cal_similarity(model,tokenizer,device,response,info)
                v = v + str(similarity) + ' '
                # 如果有多个相似度，则取最大的相似度
                if similarity > flag:
                    flag = similarity
                    region = tailor_dict
            
            v = v + 'end\n'
            with open("vis.txt","a") as f:
                f.write(v)

            # 定义Ground Truth中是否有Box
            flag_box = 0   # Ground Truth中没有Box
            if link.split('.')[0] in seqs:
                flag_box = 1

            # 如果最大相似度满足阈值，则视为检测到Box
            if flag > 0.3:
                draw_box(link,region[region['path']],k)

                # 判断Ground Truth中是否有Box
                if not flag_box:    # Ground Truth中没有Box
                    FP = FP + 1

                else:
                    # 计算IoU
                    iou, distance = calculate_iou_euclidean(region[region['path']],link,path)
                    if iou > 0.5:
                        TP = TP + 1
                        s_dn = s_dn + distance
                    else:
                        FN = FN + 1
                        IDS = IDS + 1

            else: # 检测到了物体，但是相似度低于阈值, 因此视为未检测到正确物体
                shutil.copyfile(path, 'Output'+ str(k) + '/' + link)

                if not flag_box:    # Ground Truth中没有Box
                    TN = TN + 1
                else:
                    FN = FN + 1

            shutil.rmtree("Tailor")

        
        s_time = s_time + time.time() - start_time

        # 将结果保存为视频
        # images_to_video("Output"+str(k), "output"+str(k)+'.mp4', 5)
        # shutil.rmtree("Output"+str(k))
        k = k + 1

        if TP/(TP+FN) >= 0.5:
            s_rate = s_rate + 1


        with open("end.txt","a") as f:
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

    with open("end.txt","a") as f:
            f.write('s_TP:'+str(s_TP)+" s_FN:"+str(s_FN)+" s_FP:"+str(s_FP)+" s_TN:"+str(s_TN)+" s_IDS:"+str(s_IDS)+" s_GT:"+str(s_GT)+" FPS:"+str(FPS)+" MOTA:"+str(MOTA)+" MOTP:"+str(MOTP)+" Recall:"+str(Recall)+" Accuracy:"+str(Accuracy)+" Success_Rate:"+str(Success_Rate)+'\n')

if __name__ == '__main__':
    main()