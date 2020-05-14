#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:05:04 2020

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

# 加载数据
train_file =  "./dataset/filtered_train_paraphrases.tsv"
dev_file =  "./dataset/filtered_dev_paraphrases.tsv"
test_file =  "./dataset/filtered_test_paraphrases.tsv"

train_data = data_reader(train_file)
dev_data = data_reader(dev_file)
test_data = data_reader(test_file)

corpus = train_data + dev_data
words, word2id, id2word = get_vocab(corpus, "./dataset/seq2seq_config.json")

def data_generator(data, batch_size):
    """数据生成器
    """
    X, Y = [], []
    # 非常awesome的while True
    # 可以使得最后一个batch(当数据不能被batch_size整除时)也参与训练(不够的话用开头数据补充)
    while True:
        for s in data:
            X.append(str2id(s[0], word2id=word2id))
            Y.append(str2id(s[1], word2id=word2id, start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def main():
    BATCH_SIZE = 64
    EPOCHS = 100
    EMBED_DIM = 300
    HIDDENSIZE = 128
    
    model = seq2seq(words)
    
    s1 = u'how can i stop my depression ?'
    s2 = u'what is best time to have sex ?'
    s3 = u'how is donald trump a better choice than hillary clinton ?'
    s4 = u'which is the best private university in india for engineering ?'
    
    
    class Evaluate(Callback):
        def __init__(self):
            self.lowest = 1e10
        def on_epoch_end(self, epoch, logs=None):
            # 训练过程中观察一两个例子，显示标题质量提高的过程
            print(gen_sent(s1, model, word2id, id2word))
            print(gen_sent(s2, model, word2id, id2word))
            print(gen_sent(s3, model, word2id, id2word))
            print(gen_sent(s4, model,word2id, id2word))
            # 保存最优结果
            if logs['loss'] <= self.lowest:
                self.lowest = logs['loss']
                model.save_weights('./saved_models/best_model.weights.h5')


    evaluator = Evaluate()
    
    model.fit_generator(data_generator(train_data, BATCH_SIZE),
                        steps_per_epoch=np.ceil(len(train_data)/BATCH_SIZE),
                        epochs=EPOCHS,
                        callbacks=[evaluator])
    

if __name__ == '__main__':
    main()
