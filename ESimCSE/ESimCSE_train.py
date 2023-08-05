import os
from typing import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import sys

from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr
from transformers import BertModel, BertConfig, BertTokenizer, BatchEncoding, BertTokenizerFast

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ESimCSE_dataloader import TrainDataset, TestDataset, CollateFunc, load_sts_data, load_sts_data_unsup
from ESimCSE_Model import ESimcseModel, MomentumEncoder, MultiNegativeRankingLoss

def get_bert_input(source:BatchEncoding, device:str)->Tuple[BatchEncoding, BatchEncoding, BatchEncoding]:
    input_ids = source.get('input_ids').to(device)
    attention_mask = source.get('attention_mask').to(device)
    token_type_ids = source.get('token_type_ids').to(device)
    return input_ids, attention_mask, token_type_ids

def train(model:ESimcseModel,
          momentum_encoder:MomentumEncoder,
          train_dl:DataLoader,
          dev_dl:DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_func:Callable=MultiNegativeRankingLoss.multi_negative_ranking_loss,
          device:Union[str,torch.device]='cpu',
          save_path=None,
          gamma=0.95):

    model.train()
    best = 0
    for batch_idx, (batch_src_source, batch_pos_source, batch_neg_source) in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        input_ids_src, attention_mask_src, token_type_ids_src = get_bert_input(batch_src_source, device)
        input_ids_pos, attention_mask_pos, token_type_ids_pos = get_bert_input(batch_pos_source, device)

        neg_out = None
        if batch_neg_source:
            # input_ids_neg:[batch, seq_len]
            input_ids_neg, attention_mask_neg, token_type_ids_neg = get_bert_input(batch_neg_source, device)
            # 注意：momentum_encoder模型在这里使用
            # neg_out:[batch, model_size]
            neg_out = momentum_encoder(input_ids_neg, attention_mask_neg, token_type_ids_neg)
            # print(neg_out.shape)

        # src_out:[batch, model_size]
        # pos_out:[batch, model_size]
        src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)
        pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)

        loss = loss_func(src_out, pos_out, neg_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 注意：optimizer只更新model的参数，而不是momentum模型的参数

        #  Momentum Contrast Encoder Update
        for encoder_param, momentum_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
            # print("--", moco_encoder_param.data.shape, encoder_param.data.shape)
            # 在此处手动更新momentum模型的参数
            #冲量参数更新: 老的参数与新参数混合
            momentum_encoder_param.data = gamma \
                                      * momentum_encoder_param.data \
                                      + (1. - gamma) * encoder_param.data

        if batch_idx % 5 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = evaluation(model, dev_dl, device)
            model.train()
            if best < corrcoef:
                best = corrcoef
                # torch.save(model.state_dict(), save_path)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")

def evaluation(model:ESimcseModel, dataloader:DataLoader, device:str):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            # [batch, model_size]
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            # [batch, model_size]
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_path_sp = args.data_path + "cnsd-sts-train.txt"
    train_path_unsp = args.data_path + "cnsd-sts-train_unsup.txt"
    dev_path_sp = args.data_path + "cnsd-sts-dev.txt"
    test_path_sp = args.data_path + "cnsd-sts-test.txt"
    # pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    test_data_source = load_sts_data(test_path_sp)
    #tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    #tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    train_data_source = load_sts_data_unsup(train_path_unsp)
    train_sents = [data[0] for data in train_data_source]

    train_dataset = TrainDataset(train_sents)
    # 感觉直接用DATASET也可以处理，不需要CollateFunc?
    train_call_func = CollateFunc(tokenizer, max_len=args.max_length, q_size=args.q_size, dup_rate=args.dup_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=train_call_func)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]
    pretrain_model_path = "../data/bert_tiny/"
    model = ESimcseModel(pretrained_model=pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(args.device)
    # momentum对比模型
    momentum_encoder = MomentumEncoder(pretrained_model=pretrain_model_path, pooling=args.pooler).to(args.device)
    # 已确认model与momentum_encoder 两个不同的模型，他们的参数的内存地址也不一样

    ESimCSELoss = MultiNegativeRankingLoss()
    esimcse_loss = ESimCSELoss.multi_negative_ranking_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model,
          momentum_encoder,
          train_dataloader,
          test_dataloader,
          optimizer,
          esimcse_loss,
          args.device,
          args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save')
    parser.add_argument("--un_supervise", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--dup_rate", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--q_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=50, help="max length of input sentences")
    parser.add_argument("--data_path", type=str, default="../data/STS-B/")
    parser.add_argument("--pretrain_model_path", type=str,
                        #default="/data/Learn_Project/Backup_Data/bert_chinese")
                        default = "bert-tiny-chinese")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='first-last-avg', help='which pooler to use')

    args = parser.parse_args()
    logger.add("../log/train.log")
    logger.info("run run run")
    logger.info(args)
    main(args)
