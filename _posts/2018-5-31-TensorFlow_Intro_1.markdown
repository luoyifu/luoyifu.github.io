---
layout:     post
title:      "TensorFlow 入门（1）"
subtitle:   ""
date:       2018-05-31 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - python
    - deeplearning
---

# TensorFlow 入门（1）

因为TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图, 然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算. 
* 节点（Nodes）在图中表示数学操作,也可以表示数据输入（feed in）的起点/输出（push out）的终点
* 图中的线（edges）则表示在节点间输入、输出关系，相互联系的多维数据数组, 即张量（tensor)
训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来.

## 占位符
占位符'placeholder'，用来暂时存储变量。Tensorflow如果想要从外部传入data, 那就需要用到'tf.placeholder()', 然后以这种形式传输数据   'sess.run(***, feed_dict={input: **})'.

'''
import tensorflow as tf
eg1:
#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

eg2:
x = tf.placeholder("float", [None, 784])
# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
'''

接下来, 传值的工作交给了'sess.run()', 需要传入的值放在了'feed_dict={}'并一一对应每一个'input. placeholder'与'feed_dict={}'是绑定在一起出现的。

## Variable
一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。
eg:
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
'''
tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。

'''
import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)
'''

如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 'init = tf.initialize_all_variables()'.到这里变量还是没有被激活，需要再在 sess 里, 'sess.run(init)' , 激活 init 这一步.


## Session会话控制
Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行'session.run()'可以获得你要得知的运算结果, 或者是你所要运算的部分.
