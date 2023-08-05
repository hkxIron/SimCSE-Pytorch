import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer

"""
题目：ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding
地址：https://arxiv.org/pdf/2109.04380.pdf
代码：https://github.com/caskcsg/sentemb/tree/main/ESimCSE

解决SimCSE的两个问题：

Dropout构建的正例均是相同长度的，会导致模型认为相同句子长度的句子更相似
SimCSE增加batchsize，引入更多负例，反而引起效果下降【猜测：更多的负例中，部分与正例接近质量不高】

核心改动点：

正例生成方式：重复一定单词
负例生成方式：动量序列，扩展负样本数量

正例生成方式
重复单词

负例生成方式
动量序列，扩展负样本数量
"""
# https://github.com/caskcsg/sentemb/tree/main/ESimCSE
class ESimcseModel(nn.Module):

    def __init__(self, pretrained_model:str, pooling:str, dropout=0.3):
        super(ESimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

# 冲量模型
class MomentumEncoder(ESimcseModel):
    """ MomentumEncoder """
    def __init__(self, pretrained_model:str, pooling:str):
        super(MomentumEncoder, self).__init__(pretrained_model, pooling)

class MultiNegativeRankingLoss(nn.Module):
    # code reference: https://github.com/zhoujx4/NLP-Series-sentence-embeddings
    def __init__(self):
        super(MultiNegativeRankingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def multi_negative_ranking_loss(self,
                                    embed_src:torch.Tensor,
                                    embed_pos:torch.Tensor,
                                    embed_neg:torch.Tensor,
                                    scale=20.0):
        '''
        embed_src:[batch, model_size]
        embed_pos:[batch, model_size]
        embed_neg:[max_neg_queue_size, model_size]

        scale is a temperature parameter
        '''

        if embed_neg is not None:
            # embed_pos: [batch+160, model_size]
            embed_pos = torch.cat([embed_pos, embed_neg], dim=0)

        # print(embed_src.shape, embed_pos.shape)
        # embed_src:[batch, model_size]
        # embed_pos: [batch+160, model_size]
        # similar_score: [batch, batch+160]
        # scores的意思：所有的src都计算与pos+neg中的每个样本的相似度
        scores = self.cos_sim(embed_src, embed_pos) * scale
        # labels:[batch], 意思是src应该与pos相似
        labels = torch.tensor(range(len(scores)),
                              dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]

        # scores: [batch, batch+160]
        # labels:[batch]
        return self.cross_entropy_loss(scores, labels)

    def cos_sim(self, a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
        """
        a:[batch, model_size]
        b:[max_neg_queue_size or batch, model_size]

        return:similar_score: [batch1, batch2]
        the function is same with torch.nn.F.cosine_similarity but processed the problem of tensor dimension
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)
        # a:[batch1, model_size]
        # b:[batch2, model_size]
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1) # 在某一维度上计算p范式，此处为2范式
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        # similar_score:[batch1, batch2]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

if __name__ == '__main__':
    import numpy as np
    input1 = torch.randn(100, 128)
    input2 = torch.randn(100, 128)
    output = F.cosine_similarity(input1, input2)
    print(output.shape)

    embed_src = torch.tensor(np.random.randn(32, 768))  # (batch_size, 768)
    embed_pos = torch.tensor(np.random.randn(32, 768))
    embed_neg = torch.tensor(np.random.randn(160, 768))

    ESimCSELoss = MultiNegativeRankingLoss()
    esimcse_loss = ESimCSELoss.multi_negative_ranking_loss

    res = esimcse_loss(embed_src, embed_pos, embed_neg)
    print(f"similar score:{res}")
