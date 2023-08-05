from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer, BatchEncoding
from typing import *
from transformers import (
  BertTokenizerFast,
  AutoModel,
)

"""
有监督
蕴含句子对为正样本，矛盾句子对为负样本
query1, query2, similar_score
"""
def load_sts_data(path)->List[Tuple[str, str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("||")
            # 示例数据
            # 2012test-0025||北极熊在雪地上滑行。||一只北极熊在雪地上滑行。||5
            # query1, query2, similar_score
            data_source.append((line_split[1], line_split[2], line_split[3]))
        return data_source

"""
无监督的方式
利用同一个句子的两次dropout作为正样本，同一batch内的其它作为负样本
"""
def load_sts_data_unsup(path:str)->List[List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            # query:人们下了火车。
            line_split = line.strip().split("\n")
            data_source.append(line_split)
        return data_source

class TrainDataset(Dataset):
    def __init__(self, data:List, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int)->BatchEncoding:
        text = self.data[index]
        # 同一个text重复两次， 通过bert编码互为正样本
        # tokens:BatchEncoding, tokens['input_ids'].shape:[2, max_seq_in_batch]
        tokens = self.tokenizer([text, text], max_length=self.max_len,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        return tokens


class TestDataset(Dataset):
    def __init__(self, data:List, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text:str):
        return self.tokenizer(text,
                              max_length=self.max_len,
                              truncation=True,
                              padding='max_length',
                              return_tensors='pt')

    def __getitem__(self, index)->Tuple[BatchEncoding, BatchEncoding, int]:
        da = self.data[index]
        # data: ['一个男人在弹吉他。', '一个人在吹小号。', '1']
        # 格式为：文本1, 文本2, 相似度分数
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])


if __name__ == "__main__":
    import numpy as np

    train_path_sp = "../data/STS-B/" + "cnsd-sts-train.txt"
    dev_path_sp = "../data/STS-B/" + "cnsd-sts-dev.txt"
    #pretrain_model_path = "/Learn_Project/Backup_Data/macbert_chinese_pretrained"
    #pretrain_model_path = "bert-base-chinese"
    #pretrain_model_path = "bert-tiny-chinese"

    # query1, query2, similar_score
    train_data_source = load_sts_data(train_path_sp)
    # query1, query2, similar_score
    test_data_source = load_sts_data(dev_path_sp)
    #tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    # 两个query list进行合并
    train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
    train_dataset = TrainDataset(train_sents, tokenizer, max_len=64)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)

    for batch_idx, source in enumerate(train_dataloader, start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1)
        print(input_ids.shape, attention_mask.shape, token_type_ids.shape)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=256)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=12)

    for source, target, label in test_dataloader:
        # source        [batch, 1, seq_len] -> [batch, seq_len]
        source_input_ids = source.get('input_ids').squeeze(1)
        source_attention_mask = source.get('attention_mask').squeeze(1)
        source_token_type_ids = source.get('token_type_ids').squeeze(1)

        # target        [batch, 1, seq_len] -> [batch, seq_len]
        target_input_ids = target.get('input_ids').squeeze(1)
        target_attention_mask = target.get('attention_mask').squeeze(1)
        target_token_type_ids = target.get('token_type_ids').squeeze(1)
        # concat
        #label_array = np.append(label_array, np.array(label))

