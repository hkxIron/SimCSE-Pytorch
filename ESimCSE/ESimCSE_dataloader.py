import random
import jieba
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer, BatchEncoding
from transformers import (
    BertTokenizerFast,
    AutoModel,
)
from typing import *

# https://jmxgodlz.xyz/2022/11/05/2022-11-05-%E8%BF%9B%E5%87%BB%EF%BC%81BERT%E5%8F%A5%E5%90%91%E9%87%8F%E8%A1%A8%E5%BE%81/#more
def load_sts_data(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("||")
            # 示例数据
            # 2012test-0025||北极熊在雪地上滑行。||一只北极熊在雪地上滑行。||5
            # query1, query2, similar_score
            data_source.append((line_split[1], line_split[2], line_split[3]))
        return data_source

def load_sts_data_unsup(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("\n")
            # query:人们下了火车。
            data_source.append(line_split)
        return data_source

class TrainDataset(Dataset):
    def __init__(self, data:List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        return text

# collate:整理(文件或书等);核对，校勘，对照(不同来源的信息)
"""
merges a list of samples to form a mini-batch of Tensor(s).  
Used when using batched loading from a map-style dataset.
"""
class CollateFunc(object):
    def __init__(self, tokenizer, max_len=256, q_size=160, dup_rate=0.15):
        # 负样本队列
        self.negative_sample_queue = []
        # 负样本队列长度
        self.negative_sample_queue_size = q_size
        self.max_len = max_len
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition_normal(self, batch_text:List[str]):
        dst_text = list()
        for text in batch_text:
            actual_len = len(text)
            """
            正例生成方式：重复一定单词
            """
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))
            dup_word_index:List[int] = random.sample(list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            dst_text.append(dup_text)
        return dst_text

    def word_repetition_chinese(self, batch_text):
        ''' span duplicated for chinese
        '''
        dst_text = list()
        for text in batch_text:
            # 对于中文，按单词单位进行重复
            cut_text = jieba.cut(text, cut_all=False)
            text = list(cut_text)

            """
            正例生成方式：重复一定单词,以防止模型误认为字数相同的就是正样本
            """
            actual_len = len(text) # 单词的数量
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
                dst_text.append(dup_text)
            return dup_text

    #负例生成方式：动量序列，扩展负样本数量到160
    def negative_samples(self, batch_src_text:List[str]):
        batch_size = len(batch_src_text)
        negative_samples = None
        if len(self.negative_sample_queue) > 0:
            negative_samples = self.negative_sample_queue[:self.negative_sample_queue_size]
            # print("size of negative_samples", len(negative_samples))

        if len(self.negative_sample_queue) + batch_size >= self.negative_sample_queue_size: # q_size:160
            # 如果长度越出，将前面的样本清除
            del self.negative_sample_queue[:batch_size]

        self.negative_sample_queue.extend(batch_src_text)
        return negative_samples

    def __call__(self, batch_text:List[str])->Tuple[BatchEncoding, BatchEncoding, BatchEncoding]:
        '''
        input: batch_text: [batch_text,], shape:[batch]
        output: batch_src_text, batch_dst_text, batch_neg_text
        '''

        batch_pos_text = self.word_repetition_normal(batch_text)
        batch_neg_text = self.negative_samples(batch_text)
        # print(len(batch_pos_text))

        batch_tokens = self.tokenizer(batch_text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        batch_pos_tokens = self.tokenizer(batch_pos_text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')

        batch_neg_tokens = None
        if batch_neg_text:
            batch_neg_tokens = self.tokenizer(batch_neg_text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        # batch_tokens['input_ids']:[batch]
        # batch_pos_tokens['input_ids']:[batch]
        # batch_neg_tokens['input_ids']:[max_queue_size=160]
        return batch_tokens, batch_pos_tokens, batch_neg_tokens

class TestDataset(Dataset):
    def __init__(self, data:List, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer(text,
                              max_length=self.max_len,
                              truncation=True,
                              padding='max_length',
                              return_tensors='pt')

    def __getitem__(self, index:int):
        da = self.data[index]
        # data: ['一个男人在弹吉他。', '一个人在吹小号。', '1']
        # 格式为：文本1, 文本2, 相似度分数
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])


if __name__ == "__main__":
    train_path_sp = "../data/STS-B/" + "cnsd-sts-train.txt"
    dev_path_sp = "../data/STS-B/" + "cnsd-sts-dev.txt"
    #pretrain_model_path = "/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    train_data_source = load_sts_data(train_path_sp)
    test_data_source = load_sts_data(dev_path_sp)
    #tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
    train_dataset = TrainDataset(train_sents)

    train_call = CollateFunc(tokenizer, max_len=256, q_size=160, dup_rate=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=1, collate_fn=train_call)

    for batch_idx, (batch_tokens, batch_pos_tokens, batch_neg_tokens) in enumerate(train_dataloader, start=1):
        #print("--", batch_tokens.shape)
        #print("--", batch_tokens)
        source_input_ids = batch_tokens.get('input_ids').squeeze(1)
        print(source_input_ids.shape)
