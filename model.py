#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:59:05 2020

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

def seq2seq(words,
            embed_dim=300,
            hidden_dim=128):
    """模型定义
    args:
        words: 词表list，存储了train/dev data上的vocabularies
        embed_dim: 词向量维度
        hidden_dim: lstm隐层维度
    """
    x_in = Input(shape=(None,)) # [?, ?], 第一个维度是batch_size；
    y_in = Input(shape=(None,)) # [?, ?], 第二个维度是每个batch中的最大句子长度(不确定所以不要显示地传参)
    x, y = x_in, y_in
    
    # 得到mask
    ## 比如[2, 45, 13, 68, 25, 98, 0, 0, 0, 0]这个序列，最大长度是10，有效长度是6
    ## 于是[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]便是它的mask
    ## K.greater: 比较张量的每一元素和指定变量或张量的大小，返回一个bool类型的张量
    ## x_mask: [?, ?, 1]
    ## y_mask: [?, ?, 1]
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
    y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)
    
    # 加4是因为多了mask(0)、unk(1)、start(2)、end(3)
    embedding = Embedding(len(words)+4, embed_dim)
    x = embedding(x) # x: [?, ?, 300]
    y = embedding(y) # y: [?, ?, 300]
    
    # encoder, 双层双向LSTM，它的输出理论上包含了输入句子的全部信息
    x = LayerNormalization()(x)
    # x: [?, ?, 128]
    x = OurBidirectional(CuDNNLSTM(hidden_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)
    x = OurBidirectional(CuDNNLSTM(hidden_dim // 2, return_sequences=True))([x, x_mask])
    ## x_max: [?, 128]
    x_max = Lambda(seq_maxpool)([x, x_mask])
    
    # decoder, 双层单向LSTM
    ## y: [?, ?, 300]
    y = SelfModulatedLayerNormalization(hidden_dim // 4)([y, x_max])
    ## y: [?, ?, 128]
    y = CuDNNLSTM(hidden_dim, return_sequences=True)(y)
    y = SelfModulatedLayerNormalization(hidden_dim // 4)([y, x_max])
    ## y: [?, ?, 128]
    y = CuDNNLSTM(hidden_dim, return_sequences=True)(y)
    y = SelfModulatedLayerNormalization(hidden_dim // 4)([y, x_max])
    
    # attention交互
    ## xy: [?, ?, 128]
    xy = Attention(8, 16)([y, x, x, x_mask])
    ## xy: [?, ?, 256]
    xy = Concatenate()([y, xy])
    
    # 输出分类
    xy = Dense(embed_dim)(xy)
    xy = LeakyReLU(0.2)(xy)
    xy = Dense(len(words)+4)(xy)
    xy = Activation('softmax')(xy)
    
    # 交叉熵作为loss，但mask掉padding部分
    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    model = Model([x_in, y_in], xy)
    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-3))
    
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    