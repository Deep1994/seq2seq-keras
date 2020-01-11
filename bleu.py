#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:23:44 2020

@author: dingpeng
"""

from nltk.translate.bleu_score import corpus_bleu

#references = [[['this', 'is', 'a', 'test']], [['good', 'morning', 'ha', 'ha']], [['hello', 'world', 'ha', 'ha']]]
#candidates = [['this', 'is', 'a', 'test'], ['good', 'morning', 'ha', 'ha'], ['hello', 'world', 'ha', 'ha']]


references = []
candidates = []
with open('./results/prediction.txt', 'r') as f:
    for line in f:
        line = line.split('\t')
        references.append([line[1].split()])
        candidates.append(line[2].split())
        
#score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
score = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
print(score) 