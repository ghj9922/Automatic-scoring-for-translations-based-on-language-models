from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import pandas as pd

def fluency_score(rated_a):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入预训练模型的分词器
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    # 读取 GPT-2 预训练模型
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    score_list = []
    for step, s in enumerate(rated_a):
        print(s)
        # Put model in training mode.
        if not s:
            print('space sentence')
            score_list.append(1e6)
            continue
        # 使用 GPT2Tokenizer 对输入进行编码
        s = enc.encode(s)
        batch = torch.tensor([s]).to(device)
        loss = model(batch, lm_labels=batch)  # everage -logp
        # print(loss*len(s))
        # print(loss.item())
        score_list.append(loss.item())

    cutoff = np.quantile([-t for t in score_list], 0.05) # 计算沿指定轴的数据的第q个分位数
    modified_rating = np.array([cutoff if -t < cutoff else -t for t in score_list])
    normed_rating = (modified_rating - cutoff) / np.abs(cutoff)
    return normed_rating


file_path = "fluency_input.csv"
df = pd.read_csv(file_path, header=None)
sentences = df[1].fillna('').to_list()
score_list = fluency_score(sentences)
print(score_list)

df[3] = pd.Series(score_list)
df.to_csv('fluency_output.csv')