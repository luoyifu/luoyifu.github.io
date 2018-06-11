---
layout:     post
title:      "TensorBoard使用"
subtitle:   "TensorFlow可视化应用"
date:       2018-06-10 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
    - 数据可视化
---
# TensroBoard使用指南

TensorBoard 来展现你的 TensorFlow 图像，绘制图像生成的定量指标图以及附加数据。*TensorBoard 通过读取 TensorFlow 的事件文件来运行*。TensorFlow 的事件文件包括了你会在 TensorFlow 运行中涉及到的主要数据。
使用TensorBoard的第一步是从TensorFlow运行中获取数据，为此，需要`summary ops`。`summary ops`也是operation, 和`tf.matmul`（乘）等operation是一样的，需要输入tensor，输出tensor，在graph中运行。
* `summary ops`输出的tensors会写入文件并传送给tensorboard。如果需要可视化这些结果，需要evaluate `summary op`，取回结果，使用`tf.SummaryWriter`写入文件。

## 1 Tensorboard的数据形式

Tensorboard可以记录与展示以下数据形式： 
（1）标量Scalars 
（2）图片Images 
（3）音频Audio 
（4）计算图Graph 
（5）数据分布Distribution 
（6）直方图Histograms 
（7）嵌入向量Embeddings

## 2 Tensorboard的可视化过程

（1）首先肯定是先建立一个graph,你想从这个graph中获取某些数据的信息

（2）确定要在graph中的哪些节点放置summary operations以记录信息 
使用tf.summary.scalar记录标量 
使用tf.summary.histogram记录数据的直方图 
使用tf.summary.distribution记录数据的分布图 
使用tf.summary.image记录图像数据 
….

（3）operations并不会去真的执行计算，除非你告诉他们需要去run,或者它被其他的需要run的operation所依赖。而我们上一步创建的这些summary operations其实并不被其他节点依赖，因此，我们需要特地去运行所有的summary节点。但是呢，一份程序下来可能有超多这样的summary 节点，要手动一个一个去启动自然是及其繁琐的，因此我们可以使用tf.summary.merge_all去将所有summary节点合并成一个节点，只要运行这个节点，就能产生所有我们之前设置的summary data。

（4）使用tf.summary.FileWriter将运行后输出的数据都保存到本地磁盘中

（5）运行整个程序，并在命令行输入运行tensorboard的指令，之后打开web端可查看可视化的结果

## 3. 示例
使用`summary_writer`
`SummaryWriter` 的构造函数中包含了参数 `logdir`。这个 `logdir` 非常重要，所有事件都会写到它所指的目录下。此外，`SummaryWriter` 中还包含了一个可选择的参数 `GraphDef`。如果输入了该参数，那么 TensorBoard 也会显示你的图像。
```
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
```

SummaryWriter 的构造函数中包含了参数 logdir。这个 logdir 非常重要，所有事件都会写到它所指的目录下。此外，SummaryWriter 中还包含了一个可选择的参数 GraphDef。如果输入了该参数，那么 TensorBoard 也会显示你的图像。


```
# 进入代码所在目录
python xxx.py # 代码文件
# 运行之后，可以看到目录中出现了一个logs文件，然后运行：
tensorboard --logdir logs
```
运行上面的代码，查询当前目录，就可以找到一个新生成的文件，已命名为logs，我们需在终端上运行tensorboard，生成本地链接，当然你也可以将上面的代码直接生成一个py文档在终端运行，也会在终端当前目录生成一个logs文件，然后运行tensorboards --logdir logs指令，就可以生成一个链接，复制那个链接，在浏览器粘贴显示，对于tensorboard 中显示的网址打不开的情况, 请使用 http://localhost:6006 
输出结果如下图：
![s1](/img/in-post/tensorboard_s1.png)

为数据节点增加名称
```
x = tf.placeholder("float", [None,784], name='x_in')
y_ = tf.placeholder("float", [None, 10], name='y_in')
```
![s2](/img/in-post/tensorboard_s2.png)


```
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层
# 图层的名字就是with tf.name_scope()方法里的参数。

with tf.name_scope('inputs'):
  x = tf.placeholder("float", [None, 784], name='x_in')
  y_ = tf.placeholder("float", [None,10], name='y_in')
```
![s3](/img/in-post/tensorboard_s3.png)