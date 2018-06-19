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
在训练神经网络时，为了防止过度拟合，可以使用dropout方法。该方法来源于Hilton的文章Improving neural networks by preventing co-adaptation of feature detectors.

Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
![dropout原理](/img/in-post/dropout.png)

按照hinton的文章，他使用Dropout时训练阶段和测试阶段做了如下操作：

　　在样本的训练阶段，在没有采用pre-training的网络时（Dropout当然可以结合pre-training一起使用），hintion并不是像通常那样对权值采用L2范数惩罚，而是对每个隐含节点的权值L2范数设置一个上限bound，当训练过程中如果该节点不满足bound约束，则用该bound值对权值进行一个规范化操作（即同时除以该L2范数值），说是这样可以让权值更新初始的时候有个大的学习率供衰减，并且可以搜索更多的权值空间（没理解）。

　　在模型的测试阶段，使用”mean network(均值网络)”来得到隐含层的输出，其实就是在网络前向传播到输出层前时隐含层节点的输出值都要减半（如果dropout的比例为50%），其理由文章说了一些，可以去查看（没理解）。

　　关于Dropout，文章中没有给出任何数学解释，Hintion的直观解释和理由如下：

1. 由于每次用输入网络的样本进行权值更新时，隐含节点都是以一定概率随机出现，因此不能保证每2个隐含节点每次都同时出现，这样权值的更新不再依赖于有固定关系隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况。
2. 可以将dropout看作是模型平均的一种。对于每次输入到网络中的样本（可能是一个样本，也可能是一个batch的样本），其对应的网络结构都是不同的，但所有的这些不同的网络结构又同时share隐含节点的权值。这样不同的样本就对应不同的模型，是bagging的一种极端情况。个人感觉这个解释稍微靠谱些，和bagging，boosting理论有点像，但又不完全相同。
3. native bayes是dropout的一个特例。Native bayes有个错误的前提，即假设各个特征之间相互独立，这样在训练样本比较少的情况下，单独对每个特征进行学习，测试时将所有的特征都相乘，且在实际应用时效果还不错。而Droput每次不是训练一个特征，而是一部分隐含层特征。
4. 还有一个比较有意思的解释是，Dropout类似于性别在生物进化中的角色，物种为了使适应不断变化的环境，性别的出现有效的阻止了过拟合，即避免环境改变时物种可能面临的灭亡。

TensorFlow构建DropOut时使用的函数时`tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None) `
* 第一个参数x：指输入;
* 第二个参数`keep_prob`: 设置神经元被选中的概率,在初始化时`keep_prob`是一个占位符,  `keep_prob = tf.placeholder(tf.float32)` 。tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
* train的时候才是dropout起作用的时候,test的时候不应该让dropout起作用
举例：
```
# train
train_step.run(feed_dict={x: batch_xs_1, y_: batch_ys, keep_prob: 0.5})
# test
accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
```
参考资料
<br>[tensorflow模型保存和恢复](https://blog.csdn.net/lovelyaiq/article/details/78646401)
<br>[保存和恢复模型](https://www.jianshu.com/p/c5da70ea8e41)