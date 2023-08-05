# reference: https://github.com/vdogmcgee/SimCSE-Chinese-Pytorch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import argparse

from typing import *
from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import TrainDataset, TestDataset, load_sts_data, load_sts_data_unsup
from model import SimcseModel, simcse_unsupervised_loss
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import (
    BertTokenizerFast,
    AutoModel,
)


def train(model:SimcseModel, train_dl:DataLoader, dev_dl:DataLoader, optimizer:torch.optim.Optimizer, device:Union[str, torch.device], save_path:str):
    """模型训练函数"""
    model.train()
    best = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, seq_len]
        real_batch_num = source.get('input_ids').shape[0]
        # [batch * 2, seq_len]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(device)
        # [batch * 2, seq_len]
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(device)
        # [batch * 2, seq_len]
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(device)

        # out: [batch*2, model_size = 768]
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsupervised_loss(out, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = evaluation(model, dev_dl, device)
            model.train() # 设置成train模式,batchNormal,Dropout
            if best < corrcoef:
                best = corrcoef
                torch.save(model.state_dict(), save_path)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model to path:{save_path}")

def evaluation(model:SimcseModel, dev_data_loader:DataLoader, device:Union[str,torch.device]):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dev_data_loader:
            # source: [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            # source_pred: [batch, model_size=768]
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            # target_pred: [batch, model_size=768]
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            # sim:[batch]
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            # sim:[batch]
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            # label:[batch],人工标注的相似度分数
            label_array = np.append(arr=label_array, values=np.array(label))
    # corrcoef, 用spearmanr系数计算两个向量的相关性
    corr = spearmanr(label_array, sim_tensor.cpu().numpy())
    return corr.correlation

def train_model(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #args.device = torch.device("cpu")

    train_path_sp = args.data_path + "cnsd-sts-train.txt"
    train_path_unsp = args.data_path + "cnsd-sts-train_unsup.txt"
    dev_path_sp = args.data_path + "cnsd-sts-dev.txt"
    test_path_sp = args.data_path + "cnsd-sts-test.txt"
    #pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"
    pretrain_model_path = "../data/bert_tiny/"
    #pretrain_model_path = args.pretrain_model_path

    test_data_source = load_sts_data(test_path_sp)
    #tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    if args.un_supervise:
        train_data_source = load_sts_data_unsup(train_path_unsp)
        train_sents = [data[0] for data in train_data_source]
        train_dataset = TrainDataset(train_sents, tokenizer, max_len=args.max_length)
    else:
        train_data_source = load_sts_data(train_path_sp)
        # train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
        train_dataset = TestDataset(train_data_source, tokenizer, max_len=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]
    model = SimcseModel(pretrained_model=pretrain_model_path,
                        pooling=args.pooler,
                        dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, train_dataloader, test_dataloader, optimizer, args.device, args.save_path)

def test_model(args):
    pretrain_model_path = "../data/bert_tiny/"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_path_sp = args.data_path + "cnsd-sts-test.txt"
    #pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"
    pretrain_model_path = "../data/bert_tiny/"
    test_data_source = load_sts_data(test_path_sp)
    #tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    model = SimcseModel(pretrained_model=pretrain_model_path,
                        pooling=args.pooler,
                        dropout=args.dropout).to(args.device)
    corrcoef = evaluation(model, test_dataloader, args.device)
    print(corrcoef)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save')
    parser.add_argument("--un_supervise", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=float, default=5)
    parser.add_argument("--max_length", type=int, default=64, help="max length of input sentences")
    parser.add_argument("--data_path", type=str, default="../data/STS-B/")
    parser.add_argument("--pretrain_model_path", type=str,
                        #default="bert-base-chinese")
                        #default="distilbert-base-uncased")
                        default="bert-tiny-chinese")
                        #default="/data/Learn_Project/Backup_Data/bert_chinese")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='first-last-avg', help='which pooler to use')

    args = parser.parse_args()
    logger.add("../log/train.log")
    logger.info(args)
    train_model(args)
    test_model(args)
