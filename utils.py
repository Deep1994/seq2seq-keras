#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:06:07 2020

@author: dingpeng
"""
from preprocessing import clean_text
import collections
import json, os
import numpy as np
from keras import backend as K

def get_paraphrases(file):
    """得到数据集中所有的paraphrase,并进行预处理
    """
    paraphrases = []
    with open(file, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip().split("\t")
            if line[0] == "1":
                paraphrases.append([clean_text(line[1]).lower(), clean_text(line[2]).lower()])
    return paraphrases

"""
train_file = "./dataset/Quora_question_pair_partition/train.tsv"
train_paraphrases = get_paraphrases(file)

dev_file = "./dataset/Quora_question_pair_partition/dev.tsv"
dev_paraphrases = get_paraphrases(dev_file)

test_file = "./dataset/Quora_question_pair_partition/test.tsv"
test_paraphrases = get_paraphrases(test_file)
"""

def save_clean_data(clean_data, saved_path):
    """把预处理过后的数据保存起来，节省时间
    args:
        clean_data: 预处理之后的数据
        saved_path: 保存的文件路径
    """
    with open(saved_path, 'w') as f:
        for p in clean_data:
            f.write("1\t")
            f.write(p[0] + "\t")
            f.write(p[1] + "\n")
            
"""
save_clean_data(train_paraphrases, "./dataset/cleaned_train_paraphrases.tsv")
save_clean_data(dev_paraphrases, "./dataset/cleaned_dev_paraphrases.tsv")
save_clean_data(test_paraphrases, "./dataset/cleaned_test_paraphrases.tsv")
"""

def filter_data(filter_len, data_path):
    """过滤出指定长度以内的数据
    args:
        filter_len: 想要过滤的长度
        data_path: 未过滤之前的数据路径
    """
    filtered_data = []
    with open(data_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip().split("\t")
            if len(line[1].split()) < filter_len and len(line[2].split()) < filter_len:
                filtered_data.append([line[1], line[2]])
                
    return filtered_data
                
"""
# 139306 -> 103264
filtered_train_data = filter_data(15, "./dataset/cleaned_train_paraphrases.tsv")
# 5000 -> 3699
filtered_dev_data = filter_data(15, "./dataset/cleaned_dev_paraphrases.tsv")
# 5000 -> 3665
filtered_test_data = filter_data(15, "./dataset/cleaned_test_paraphrases.tsv")

# 保存起来
save_clean_data(filtered_train_data, "./dataset/filtered_train_paraphrases.tsv")
save_clean_data(filtered_dev_data, "./dataset/filtered_dev_paraphrases.tsv")
save_clean_data(filtered_test_data, "./dataset/filtered_test_paraphrases.tsv")
"""

def data_reader(file):
    """读取数据文件
    """
    data = []
    with open(file, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip().split("\t")
            data.append([line[1], line[2]])
            
    return data

"""
train_file =  "./dataset/filtered_train_paraphrases.tsv"
dev_file =  "./dataset/filtered_dev_paraphrases.tsv"
test_file =  "./dataset/filtered_test_paraphrases.tsv"

train_data = data_reader(train_file)
dev_data = data_reader(dev_file)
test_data = data_reader(test_file)
"""
        

def get_vocab(corpus, cached_file):
    """得到语料的词表
    args:
        corpus: sentence list
    """
    if os.path.exists(cached_file):
        words, word2id, id2word = json.load(open(cached_file))
        # id2word需要int转换一下，不然id是string类型的
        id2word = {int(i):w for i, w in id2word.items()}
    else:
        all_sentences = []
        for i in corpus:
            all_sentences.append(i[0])
            all_sentences.append(i[1])
            
        tokenized_sentences = []
        for s in all_sentences:
            tokenized_sentences.append(s.split())
        
        wordcounts = collections.Counter()
        for s in tokenized_sentences:
            for w in s:
                wordcounts[w] += 1
                
        words = [wordcount[0] for wordcount in wordcounts.most_common()]
        # 0: mask
        # 1: unk
        # 2: start
        # 3: end
        word2id = {w: i+4 for i, w in enumerate(words)}
        id2word = dict((i, w) for w, i in word2id.items())
        json.dump([words, word2id, id2word], open('./dataset/seq2seq_config.json', 'w'))
    
    return words, word2id, id2word

"""
corpus = train_data + dev_data
words, word2id, id2word = get_vocab(corpus, "./dataset/seq2seq_config.json")
"""

def str2id(s, word2id, start_end=False):
    """文字转整数id，找不到的用<unk>代替
    """
    if start_end: # 补上<start>和<end>标记
        ids = [word2id.get(w, 1) for w in s.split()]
        ids = [2] + ids + [3] # [2] for <start> and [3] for <end>
    else: # 普通转化
        ids = [word2id.get(w, 1) for w in s.split()]
    return ids
    

def id2str(ids, id2word):
    """id转文字，找不到的用空字符代替
    """          
    string = ' '.join([id2word.get(i, '') for i in ids])
    return string.strip()


def padding(x):
    """padding至batch内最大长度
    """
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


def seq_avgpool(x):
    """双向rnn之后的平均池化操作
    args:
        seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，
        先除去mask部分，然后再做avgpooling.
    """
    seq, mask = x
    return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    """双向rnn之后的最大池化操作
    args:
        seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，
        先除去mask部分，然后再做maxpooling.
    """
    seq, mask = x
    # 减去一个极大的数，相当于在最大池化的时候不考虑mask部分
    # 不然特征里面如果有负数，那最大池化的时候把mask的0也考虑进来了
    seq -= (1 - mask) * 1e10    
    return K.max(seq, 1)         


def gen_sent(s, model, word2id, id2word, topk=3, maxlen=25):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s, word2id)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开头，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(maxlen): # 强制要求输出不超过maxlen字
        proba = model.predict([xid, yid])[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6) # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 每一项选出topk
        _yid = [] # 暂存的候选目标序列
        _scores = [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(topk):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        best_one = np.argmax(scores)
        if yid[best_one][-1] == 3:
            return id2str(yid[best_one], id2word)
    # 如果maxlen字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])        
            
            
            
            
        
    