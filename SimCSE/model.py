import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer

# bert toturial:
# https://github.com/google-research/bert
class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model:str, pooling:str, dropout=0.3):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    # return: [batch, model_size=768]
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, token_type_ids:torch.Tensor)->torch.Tensor:
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, model_size=768]
        if self.pooling == 'pooler': # cls token -> mlp -> activate
            return out.pooler_output  # [batch, model_size=768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, seqlen, model_size] -> [batch, model_size=768, seqlen]
            # 在最后一维(seq维度)上avg_pool
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, model_size=768]
        if self.pooling == 'first-last-avg':
            # 本文的架构是 BartForConditionalGeneration, is_encoder_decoder=true, 有decoder层
            # hidden_states:Tuple[jnp.ndarray[(batch_size, num_heads, encoder_sequence_length, embed_size_per_head]],
            # config.json里：encoder有6层，decoder有6层,每层都有hidden_state
            # hidden_states:总共有13层，其中第0层为embedding层
            first = out.hidden_states[1].transpose(1, 2)  # [batch, model_size=768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, model_size=768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, model_size=768],在最后一维seq_len维进行卷积
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, model_size=768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, model_size=768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, model_size, 2] -> [batch, model_size=768]

def simcse_unsupervised_loss(y_pred:torch.Tensor, device:str, temperature=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, model_size=768]
    """
    # 得到y_pred对应的label, [0, 1, 2, 3, ..., batch_size-1, batch_size-2]
    # y_true:[0,1,2,..., 2*batch_size-1] , shape:[2*batch_size]
    y_true = torch.arange(end=y_pred.shape[0], device=device)
    #y_true: [batch_size * 2],
    #y_true value:
    #   偶数，y_true=index+1,
    #   奇数，y_true=index-1
    y_true = (y_true - y_true % 2 * 2) + 1

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # y_pred (tensor): [batch_size * 2, model_size=768]
    # sim: [batch_size * 2, 1, 768] * [1, batch_size * 2, 768]
    #   => [batch_size * 2, batch_size * 2]
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1) # 两两之间计算内积

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    # sim: [batch_size * 2, batch_size * 2]
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temperature  # 相似度矩阵除以温度系数

    # 计算相似度矩阵与y_true的交叉熵损失
    # sim: [batch_size * 2, batch_size * 2]
    # y_true: [batch_size * 2]
    loss = F.cross_entropy(input=sim, target=y_true)
    return torch.mean(loss)
