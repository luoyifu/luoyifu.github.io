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
这种方法加载模型时和保存模型时的代码基本上是一致的，唯一不同的就是没有变量的初始化过程。模型加载的时候，如果某个变量没有被加载，则系统将会报错。

也可以用定义好的其他变量来替换模型保存时的变量名：(这里用x,y替换v1,v2)
```
v1 = tf.Variable(tf.constant(1,shape = [1]),name='v1')
v2 = tf.Variable(tf.constant(2,shape = [1]),name='v2')
result = v1 + v2
# 通过字典将变量重命名
saver = tf.train.Saver(
    {'x':v1,'y':v2})
with tf.Session() as sess:
   saver.restore(sess,'model/model.ckpt')
   out = tf.get_default_graph().get_tensor_by_name('add:0')
   print(sess.run(out))
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

如果保存好一个模型，想要在另一个模型中应用之前用到的模型。例如，构建CNN网络时，需要用到之前训练好的降噪自动编码器。

如果新模型中，想要给自动编码器赋予一个名称，使用`tf.name_scope()`函数，那么程序会报错。解决方法是，训练之前那个模型的时候，使用通用的`tf.name_scope()`函数先一步给之前的模型赋予一个名称。例如：
```
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

saver = tf.train.Saver()
with tf.Session() as sess:
    模型训练……
    saver.save(autoencoder.sess,'model/mnist_au_model.ckpt')

---
# 直接读取模型参数
# 将saver也放到tf.name_scope()函数里，在tensorboard显示时会更容易看
with tf.name_scope('AutoEncoder'):
  autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
    n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

  saver = tf.train.Saver()
  saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')

```

## DropOut

参考资料
<br>[tensorflow模型保存和恢复](https://blog.csdn.net/lovelyaiq/article/details/78646401)
<br>[保存和恢复模型](https://www.jianshu.com/p/c5da70ea8e41)