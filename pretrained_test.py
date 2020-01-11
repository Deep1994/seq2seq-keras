#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:12:36 2020

@author: dingpeng
"""

import numpy as np
from tqdm import tqdm
import os,json
import tensorflow as tf
import keras
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

from utils import *
from layers import *
from model import seq2seq
from nltk.translate.bleu_score import corpus_bleu

# 加载数据
train_file =  "./dataset/filtered_train_paraphrases.tsv"
dev_file =  "./dataset/filtered_dev_paraphrases.tsv"
test_file =  "./dataset/filtered_test_paraphrases.tsv"

train_data = data_reader(train_file)
dev_data = data_reader(dev_file)
test_data = data_reader(test_file)

corpus = train_data + dev_data
words, word2id, id2word = get_vocab(corpus, "./dataset/seq2seq_config.json")


model = seq2seq(words)

# load model
print("="*20 + "model loading" + "="*20)
model.load_weights('./saved_models/best_model.weights.h5')


references = []
candidates = []
# 看一些例子的生成结果
n = 1
with open("./results/prediction.txt", 'w') as f:
    for s in test_data:
        print(n)
        s1 = s[0]
        s2 = s[1]
        
        generated_s = gen_sent(s1, model, word2id, id2word)
    #    print("="*20)
    #    print("source: " + s1)
    #    print("reference: " + s2)
    #    print("generated: " + generated_s) 
        candidates.append(generated_s.split())
        references.append([s2.split()])
        f.write(s1 + '\t' + s2 + '\t' + generated_s + '\n')
        n += 1

# 计算bleu
bleu_2_score = corpus_bleu(references, candidates, (0.5, 0.5, 0, 0))
bleu_3_score = corpus_bleu(references, candidates, (1/3, 1/3, 1/3, 0))
bleu_4_score = corpus_bleu(references, candidates, (0.25, 0.25, 0.25, 0.25))
print("="*20)
print("bleu2: " + str(bleu_2_score)) 
print("bleu3: " + str(bleu_3_score)) 
print("bleu4: " + str(bleu_4_score)) 
print("="*20)

"""
bleu2: 0.3933923292653808
bleu3: 0.2938635100587324
bleu4: 0.22415337244246342
"""

    
    
