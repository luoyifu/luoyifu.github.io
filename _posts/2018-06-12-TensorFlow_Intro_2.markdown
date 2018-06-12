---
layout:     post
title:      "TensorFlow 入门（2）"
subtitle:   "TensorFlow的基础知识"
date:       2018-06-12 21:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---
# TensorFlow入门基础

## 模型保存和加载
### 模型保存
Tensorflow的模型保存时有几点需要注意： 
1. 利用`tf.train.write_graph()`默认情况下只导出了网络的定义（没有权重weight）。 
2. 利用`tf.train.Saver().save()`导出的文件graph_def与权重是分离的。 

graph_def文件中没有包含网络中的Variable值（通常情况存储了权重），但是却包含了constant值，所以如果我们能把Variable转换为constant，即可达到使用一个文件同时存储网络架构与权重的目标

```
saver = tf.train.Saver()
with tf.Session() as sess:
    ...

    saver.save(sess,'model/model.ckpt')
```
模型保存后，在model目录将会有三个文件。在Tensorflow版本0.11之前，这三个文件为：meta、ckpt、checkpopint，它们保存的内容如下： 
* model.ckpt.meta保存计算图的结构，即神经网络的结构 
* checkpoint保存一个目录下所有的模型文件列表。 
* ckpt 保存程序中每一个变量的取值。 
在Tensorflow版本0.11之后，有四个文件分别为：meta、.data、.index、checkpoint。其中.data文件为模型中的训练变量。

### 模型加载
模型加载有两种方式，区别是是否包含图上的所有运算。
#### 包含所有运算
```
v1 = tf.Variable(tf.constant(1,shape = [1]),name='v1')
v2 = tf.Variable(tf.constant(2,shape = [1]),name='v2')
result = v1 + v2
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'model/model.ckpt')
    print(sess.run(v1+v2))
```

#### 不包含所有运算
```
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
with tf.Session() as sess:
    save.restore(sess,'model/model.ckpt')
    #获取节点名称
    result = tf.get_default_graph().get_tensor_by_name("add:0")
    print(sess.run(result))
```
### 注意
请记住，Tensorflow变量仅在会话中存在。因此，您必须在一个会话中保存模型，调用您刚刚创建的save方法。
```
with tf.Session() as sess:  
    init = tf.global_variables_initializer()  
    sess.run(init)  

    saver = tf.train.Saver()

    # 模型训练，计算
    # ……

    saver.save(sess,'model/model.ckpt')
```



## DropOut

参考资料
[tensorflow模型保存和恢复](https://blog.csdn.net/lovelyaiq/article/details/78646401)