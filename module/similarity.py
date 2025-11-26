from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from scipy.spatial.distance import cosine

# 改进
def encode_sentence(model,tokenizer,device,sentence):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)  # 将输入张量移动到cuda:5
    outputs = model(**inputs)
    # 取最后一个标记的logits向量作为句子的表示
    return outputs.logits[:, -1, :].squeeze()

def cal_similarity(model,tokenizer,device,sentence1,sentence2):

    embedding1 = encode_sentence(model,tokenizer,device,sentence1)
    embedding2 = encode_sentence(model,tokenizer,device,sentence2)

    # 将嵌入表示转换为Float32类型
    embedding1 = embedding1.detach().cpu().float().numpy()
    embedding2 = embedding2.detach().cpu().float().numpy()

    # 计算余弦相似度
    similarity = 1 - cosine(embedding1, embedding2)

    return similarity