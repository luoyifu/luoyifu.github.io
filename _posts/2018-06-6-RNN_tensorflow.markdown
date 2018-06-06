---
layout:     post
title:      "TensorFlow 循环神经网络（RNN）"
subtitle:   ""
date:       2018-06-06 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---
# TensorFlow循环神经网络（RNN）

## RNN

* 传统的神经网络是层与层之间是全连接的，但是每层之间的神经元是没有连接的（其实是假设各个数据之间是独立的）
> 这种结构不善于处理序列化的问题。比如要预测句子中的下一个单词是什么，这往往与前面的单词有很大的关联，因为句子里面的单词并不是独立的。
* RNN 的结构说明当前的的输出与前面的输出也有关，即隐层之间的节点不再是无连接的，而是有连接的

基本的结构如图，可以看到有个循环的结构，将其展开就是右边的结构：
![RNN结构图](/img/in-post/rnn.jpg)

如上图：
* 输入单元(inputs units): $\{ {x_0},{x_1}, \cdots \cdots ,{x_t},{x_{t + 1}}, \cdots \cdots \}$,
* 输出单元(output units)为：$\{ {o_0},{o_1}, \cdots \cdots ,{o_t},{o_{t + 1}}, \cdots \cdots \}$,
* 隐藏单元(hidden units)输出集: $\{ {s_0},{s_1}, \cdots \cdots ,{ost},{s_{t + 1}}, \cdots \cdots \}$
* 时间 t 隐层单元的输出为：${s_t} = f(U{x_t} + W{s_{t - 1}})$
> f就是激励函数，一般是sigmoid,tanh, relu等
> 计算${s_{0}}$时，即第一个的隐藏层状态，需要用到${s_{-1}}$，但是其并不存在，在实现中一般置为0向量
> （如果将上面的竖着立起来，其实很像传统的神经网络，哈哈）
* 时间 t 的输出为：${o_t}=Softmax(V{s_t})$
> 可以认为隐藏层状态${s_t}$是网络的记忆单元. ${s_t}$包含了前面所有步的隐藏层状态。而输出层的输出${o_t}$只与当前步的${s_t}$有关。
> （在实践中，为了降低网络的复杂度，往往${s_t}$只包含前面若干步而不是所有步的隐藏层状态）
* 在RNNs中，每输入一步，每一层都共享参数U,V,W，（因为是将循环的部分展开，天然应该相等）
* RNNs的关键之处在于隐藏层，隐藏层能够捕捉序列的信息。


参考资料
[RNN基础知识](http://lawlite.me/2017/06/14/RNN-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8CLSTM-01%E5%9F%BA%E7%A1%80/)