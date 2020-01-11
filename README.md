# seq2seq-keras
seq2seq model implemented with keras, tested on Quora Question Pairs dataset.

# Results

data | BLEU2 |  BLEU3  |  BLEU4
-|-|-|-
QQP | 39.34 | 29.39 | 22.42

# Data split

+ train: 103264
+ test: 3665

# Some generated cases

=========================================================
+ source: what should i do to avoid sleeping in class ?
+ reference: how do i not sleep in a boring class ?
+ generated: how should i avoid sleeping during lectures ?

=========================================================

=========================================================
+ source: how can one root android devices ?
+ reference: how do i root an android device ?
+ generated: how do i root my android device ?

=========================================================

=========================================================
+ source: what the best way to learn microsoft office and its associated packages ?
+ reference: what is the best way to learn microsoft office ?
+ generated: how can i learn microsoft office ?

=========================================================

# Reference

[玩转Keras之seq2seq自动生成标题](https://kexue.fm/archives/5861) by 苏剑林
