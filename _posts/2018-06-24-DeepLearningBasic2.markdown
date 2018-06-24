---
layout:     post
title:      "深度学习进阶知识"
subtitle:   "需要了解的关于深度学习的知识和概念"
date:       2018-06-24 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - deeplearning
---
# 深度学习进阶知识

## batch size

batch size的大小表示一次性要给模型输入多少数据进行训练。
* 一次性输入所有数据，即传统的梯度下降法
* 一次输入一个数据，即随机梯度下降法或在线梯度下降法
* 一次性输入一部分数据，即batch梯度下降法

在同等的计算量之下（一定的时间内），使用整个样本集的收敛速度要远慢于使用少量样本的情况。换句话说，要想收敛到同一个最优点，使用整个样本集时，虽然迭代次数少，但是每次迭代的时间长，耗费的总时间是大于使用少量样本多次迭代的情况的。

**batch的size设置的不能太大也不能太小，因此实际工程中最常用的就是mini-batch，一般size设置为几十或者几百。**
[batch size如何选择](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484570&idx=1&sn=4c0b6b76a7f2518d77818535b677e087&chksm=970c2c4ca07ba55ad5cfe6b46f72dbef85a159574fb60b9932404e45747c95eed8c6c0f66d62#rd)