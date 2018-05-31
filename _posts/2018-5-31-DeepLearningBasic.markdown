---
layout:     post
title:      "深度学习基础知识"
subtitle:   "需要了解的关于深度学习的知识和概念"
date:       2018-05-31 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - deeplearning
---

# 深度学习基础知识

## Softmax回归
softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类。
![img](/img/in-post/softmax.jpg)
以上图为例，softmax直白来说就是将原来输出是3,1,-3通过softmax函数一作用，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标

TensorFlow官方文档MNIST数据的例子：
MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。
因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为：
evidence_i=sum[x=j](w_ij*x_j+b_i)
y=softmax(evidence)



参考资料
[详解Softmax函数](https://zhuanlan.zhihu.com/p/25723112)