---
layout:     post
title:      "TensorFlow 入门（1）"
subtitle:   "TensorFlow的基础知识"
date:       2018-05-31 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---

# TensorFlow 入门（1）

因为TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图, 然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算. 
* 节点（Nodes）在图中表示数学操作,也可以表示数据输入（feed in）的起点/输出（push out）的终点
* 图中的线（edges）则表示在节点间输入、输出关系，相互联系的多维数据数组, 即张量（tensor)
训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来。

TensorFlow的操作一般分为2步，第一步是创建一个图，第二步是在session中进行图计算。

## 一、数据类型

### 1. 常数（constant types）

可以这样创建一个常数：
```
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='dot_production')
with tf.Session() as sess:  # 这一块参见后面的Session
    print(sess.run(x))
>> [[0, 2]
    [4, 6]]
```
还可以创建一些特殊值常数：
```
tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
tf.fill(dims, value, name=None)
tf.fill([2, 3], 8)
>> [[8, 8, 8], [8, 8, 8]]
```
序列创建（和numpy类似）
```
tf.linspace(start, stop, num, name=None)
tf.linspace(10.0, 13.0, 4)
>> [10.0, 11.0, 12.0, 13.0]
tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, limit=18, delta=3)
>> [3, 6, 9, 12, 15]
```

产生随机数：
```
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

### 2. Variable
常数会存在计算图的定义中，所以如果常量过多，加载计算图会很慢。

一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

#### 创建变量
```
a = tf.Variable(2, name='scalar')
b = tf.Variable([2, 3], name='vector')
c = tf.Variable([[0, 1], [2, 3]], name='matrix')

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。

```
import tensorflow as tf

state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)
```
#### 变量操作
变量有如下几个操作：
```
x = tf.Variable()
x.initializer # 初始化
x.eval() # 读取里面的值
x.assign() # 分配值给这个变量
```

#### 变量初始化
注意：如果你在 Tensorflow 中设定了变量，那么**必须对其进行初始化变量**。
所以定义了变量以后, 一定要定义 `init = tf.initialize_all_variables()`.到这里变量还是没有被激活，需要再在 sess 里, `sess.run(init)`, 激活 init 这一步。。记得在完全构建好模型并加载之后再运行这个操作。

官方文档的例子：
```
# 创建2个变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# 添加一个操作来初始化所有变量
init_op = tf.initialize_all_variables()

# 接下来，在部署模型的时候：
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...
```

初始化有两种方法：
1. 一次性初始化所有变量：
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

2. 初始化一部分变量：

```
# 初始化a,b
init_ab = tf.variable_initializer([a, b], name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)

# 对某一变量初始化
w = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(w.initializer)
```


**既然有全局初始化操作，为什么还要初始化某个变量呢？**因为，有时候需要要用另一个变量的初始化值给当前变量初始化。由于`tf.initialize_all_variables()`是并行地初始化所有变量，所以在有这种需求的情况下需要小心。
用其它变量的值初始化一个新的变量时，使用其它变量的`initialized_value()`属性。你可以直接把已初始化的值作为新变量的初始值，或者把它当做tensor计算得到一个值赋予新变量。

```
# 创建一个变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
# 创建另一个变量，和weights有相同值
w2 = tf.Variable(weights.initialized_value(), name="w2")
# 创建另一个变量，值是weights的2倍
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
```

另外，**如果想把Variable的值重置为初始值，就再run一次init操作。**

#### 保存和恢复
用`tf.train.Saver()`创建一个Saver来管理模型中的所有变量。
```
# 创建变量
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加操作初始化变量
init_op = tf.initialize_all_variables()

# 添加操作存取所有变量
saver = tf.train.Saver()

# 接下来，部署模型，初始化变量，做一些操作，存储variables
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # 存储variables
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print "Model saved in file: ", save_path
```
用同一个`Saver`对象来恢复变量。注意，当你从文件中恢复变量时，不需要事先对它们做初始化。

```
# 创建变量
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加ops存储变量
saver = tf.train.Saver()

# 接下来，部署模型，使用saver从磁盘读取数据，接下来进行操作
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print "Model restored."
  # Do some work with the model
  ...
```
如果你不给`tf.train.Saver()`传入任何参数，那么`saver`将处理`graph`中的所有变量。其中每一个变量都以变量创建时传入的名称被保存。有时候在检查点文件中明确定义变量的名称很有用。举个例子，你也许已经训练得到了一个模型，其中有个变量命名为"weights"，你想把它的值恢复到一个新的变量"params"中。

有时候仅保存和恢复模型的一部分变量很有用。再举个例子，你也许训练得到了一个5层神经网络，现在想训练一个6层的新模型，可以将之前5层模型的参数导入到新模型的前5层中。你可以通过给`tf.train.Saver()`构造函数传入Python字典，很容易地定义需要保持的变量及对应名称：键对应使用的名称，值对应被管理的变量。

注意：

* 如果需要保存和恢复模型变量的不同子集，可以创建任意多个saver对象。同一个变量可被列入多个saver对象中，只有当saver的restore()函数被运行时，它的值才会发生改变。
* 如果你仅在session开始时恢复模型变量的一个子集，你需要对剩下的变量执行初始化op。

```
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...
```

### 3. 占位符

在定义tensorflow图时，对于暂时不知道值的量，可以使用占位符，饭后再用`feed_dict`去赋值。

占位符`placeholder`，用来暂时存储变量。Tensorflow如果想要从外部传入data, 那就需要用到`tf.placeholder()`, 然后以这种形式传输数据   `sess.run(***, feed_dict={input: **})`.

```
import tensorflow as tf
eg1:
#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

eg2:
x = tf.placeholder("float", [None, 784])
# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
```

**需要注意的是：**dtype是必须要指定的参数，shape如果是None，说明任何大小的tensor都能够接受，使用shape=None很容易定义好图，但是在debug的时候这将成为噩梦，所以最好是指定好shape。

接下来, 传值的工作交给了`sess.run()`, 需要传入的值放在了`feed_dict={}`并一一对应每一个`input. placeholder`与`feed_dict={}`是绑定在一起出现的。

### 4. lazy loading
lazy loading是指你推迟变量的创建直到你必须要使用他的时候。下面我们看看一般的loading和lazy loading的区别。
```
# normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)

# lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y))
```
normal loading 会在图中创建x和y变量，同时创建x+y的运算，而lazy loading只会创建x和y两个变量。这不是一个bug，那么问题在哪里呢？

normal loading在session中不管做多少次x+y，只需要执行z定义的加法操作就可以了，而lazy loading在session中每进行一次x+y，就会在图中创建一个加法操作，如果进行1000次x+y的运算，normal loading的计算图没有任何变化，而lazy loading的计算图会多1000个节点，每个节点都表示x+y的操作。

这就是lazy loading造成的问题，这会严重影响图的读入速度。

## 二、Session会话控制
Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行`session.run()`可以获得你要得知的运算结果, 或者是你所要运算的部分.

我们在编写代码的时候，总是要先定义好整个图，然后才调用sess.run()。注意，**调用sess.run()的时候，tensorflow并没有计算整个图，只是计算了与想要fetch 的值相关的部分**

`Session.run`方法有2个参数，分别是`fetches`和`feed_dict`。参数名有时候可以省略，比如`sess.run(fetches=product)`和`sess.run(product)`是一样的。传递给`fetches`参数的既可以是Tensor也可以是Operation。如果传给`fetches`的是一个list，run返回的结果也是一个与之对应的list.

**sess.run() 中的feed_dict:**
`feed_dict`的作用是给使用`placeholder`创建出来的tensor赋值。
> 其实，他的作用更加广泛：feed 使用一个值临时替换一个 op 的输出结果. 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失.

加载 Tensorflow ，然后建立两个`matrix`,输出两个`matrix`矩阵相乘的结果。
```
import tensorflow as tf

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2) # matmul是矩阵乘法
```

因为`product`不是直接计算的步骤, 所以我们会要使用`Session`来激活`product`并得到计算结果. 有两种形式使用会话控制`Session`。
```
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# [[12]]

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
# [[12]]
```

TensorFlow中每个session是相互独立的。
```
W = tf.Variable(10) # 创建变量
sess1 = tf.Session() 
sess2 = tf.Session()
sess1.run(W.initializer) # 初始化sess1的变量
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8
print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42
sess1.close()
sess2.close()
```

**参考资料**
<br> [tensorflow学习笔记](https://zhuanlan.zhihu.com/p/28674996)
<br> [变量，创建，初始化——官方文档](http://www.tensorfly.cn/tfdoc/how_tos/variables.html)