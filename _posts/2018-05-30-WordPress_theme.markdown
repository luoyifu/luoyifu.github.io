---
layout:     post
title:      "WordPress主题自定义修改（1）"
subtitle:   "如何修改WordPress主题"
date:       2018-05-29 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - design
    - 生活
    - WordPress
    - 网站
---
# WordPress主题自定义修改（1）

WordPress主题是强大的，但面对各种各样的需求，我们仍然需要对其进行一些微调。如果是布局和设计上不满意，那么换个主题无疑是更快捷的选择。在做网站的第一周，我曾经尝试自己修改自带主题Busiprof，主题内各个文件的功能，相互之间的链接都莫名其妙。作为一个初学者，这条路似乎走不通。
这篇文章我先介绍WordPress主题的各个文件及基础知识，了解了这些才能在后续对主题进行自定义修改。

Content:
* 主题文件夹中各个文件功能介绍
* 文件执行顺序
* Post和Page
* 制作菜单很有必要

---
## 1. WordPress主题中的各个文件
主题存储在wp-content/themes文件夹中。
![themedoc](/img/in-post/themedoc1.JPG)
打开一个主题，以busiprof为例，可以看到许多文件：
![themedoc2](/img/in-post/themedoc2.JPG)

这些文件大致内容为：
    css（文件夹）：存放 css 文件
    functions（文件夹）：存放函数
    img（文件夹）：主题自带图片
    js（文件夹）：存放 js 文件
    languages（文件夹）：存放语言文件
    404.php：出现404错误时使用的文件
    archive.php：分类页面
    author.php：作者
    category.php： – 如果分类的缩略名为news，WordPress将会查找category-news.php(WordPress 2.9及以上版本支持) 
    comments.php：评论
    content.php
    footer.php：页脚
    functions.php：定义函数
    header.php：页头
    index.php：首页
    page.php：页面（page）
    screenshot.php：主题缩略图，在后台显示
    search.php：搜索页面
    searchform.php：搜索框
    sidebar.php：侧边栏
    single.php：文章（post）页面
    style.css：样式文件

## 2. 文件执行及顺序

 以主页为例，下面有2个文件 home.php 和 index.php，WordPress 程序会从你的主题文件夹中依次查找这两个文件（后台设置首页显示为”最新文章”的前提下）：

    如果找到 home.php，则使用 home.php 作为博客首页模板，即使你的主题文件夹中有 index.php；
    如果 home.php 未找到，则使用 index.php 作为首页模板；
    如果 home.php 和 index.php 都找不到，你的主题将不会被 WordPress 识别。

主页
    home.php
    index.php

文章页：
    single-{post_type}.php – 如果文章类型是videos（即视频），WordPress就会去查找single-videos.php（WordPress 3.0及以上版本支持）
    single.php
    index.php

页面
    自定义模板 – 在WordPress后台创建页面的地方，右侧边栏可以选择页面的自定义模板
    page-{slug}.php – 如果页面的缩略名是news，WordPress将会查找 page-news.php（WordPress 2.9及以上版本支持）
    page-{id}.php – 如果页面ID是6，WordPress将会查找page-6.php
    page.php
    index.php

分类
    category-{slug}.php – 如果分类的缩略名为news，WordPress将会查找category-news.php(WordPress 2.9及以上版本支持)
    category-{id}.php -如果分类ID为6，WordPress将会查找category-6.php
    category.php
    archive.php
    index.php

标签
    tag-{slug}.php – 如果标签缩略名为sometag，WordPress将会查找tag-sometag.php
    tag-{id}.php – 如果标签ID为6，WordPress将会查找tag-6.php（WordPress 2.9及以上版本支持）
    tag.php
    archive.php
    index.php

作者
    author-{nicename}.php – 如果作者的昵称为rami，WordPress将会查找author-rami.php（WordPress 3.0及以上版本支持）
    author-{id}.php – 如果作者ID为6，WordPress将会查找author-6.php（WordPress 3.0及以上版本支持）
    author.php
    archive.php
    index.php

日期页面
    date.php
    archive.php
    index.php

搜索结果
    search.php
    index.php

404 (未找到)页面
    404.php
    index.php

附件页面
    MIME_type.php – 可以是任何MIME类型 (image.php, video.php, audio.php, application.php 或者其他).
    attachment.php
    single.php
    index.php

## 3. Post和Page
页面中，有一点要格外注意。WordPress的逻辑将页面分为两类，Post和Page。这在仪表盘中也可以体现：
![yibiaopan](/img/in-post/yibiaopan.JPG)

* 文章（post）：博客最基本的组成，默认情况下就是博客首页按照倒序显示的文章。必须属于某个分类（category），逻辑上属于 WordPress 的循环函数之内，可用查询函数（query_posts 等）按照指定条件从数据库中取出你想要的文章，然后利用循环函数显示在首页。

在仪表盘”文章”->”写文章”中发表的就是文章（post）

* 页面（page）：在 WordPress 中，你可以发表 posts 或者 pages。当你需要一篇常规博客时，应该发表 post。默认情况下，post 就是在你博客首页以时间倒序显示的文章。而页面（page）则是那些像“关于我们”，“联系方式”等等类型的文章。页面（pages）是跳出博客时间逻辑的文章，常常用来发表关于你或者你的网站的与时间关系不大的信息（总是有时效性的信息）。当然，你可以用 page 来组织管理任何内容。除了一般的“关于我们”、“联系方式”等 page，还有一些常见的页面如版权、公告、法律信息、转载授权、公司信息等。

在仪表盘”页面”->”新建页面”中发表的就是页面（page）

## 4. 创建几个菜单很有必要
创建菜单需要点击外观->菜单，然后就看到如下页面：
![caidan](/img/in-post/caidan.JPG)
在顶端可以选择一个菜单进行编辑或者创建新的菜单。菜单可以是页面、分类目录、文章。菜单中可以设置子项目，在网站中显示下拉菜单的效果。
![caidan2](/img/in-post/caidan2.JPG)

> **需要多思考一下的问题**：WordPress中发表的文章是按时间顺序显示的，但我们看到网站很多需要分为不同栏目，我们希望不同栏目可以单独在菜单中找到。这样，“分类目录”添加进菜单就可以解决。一个菜单中可以既有分类目录，又有页面，还有文章。

菜单不止可以创建一个，比如我就创建了3个菜单，分别是“栏目、导航、友情链接”。WordPress主题中会在许多地方显示菜单，这个时候就是不同菜单的用武之处了。

比如在我的网站中，我的三个菜单分别起到不同作用：
![wangzhan-caidan](/img/in-post/wangzhan_caidan.jpg)
第一个栏目菜单在网站最顶端，导航菜单在侧边栏（sidebar区域，将在每个Page和Post的侧边栏显示），最后一个友情链接我放在了底端（footer区域）。



## 可以继续阅读一些参考资料
以上内容参考了许多，感谢:
[WordPress主题文件结构及执行](https://blog.csdn.net/liuxuekai/article/details/52371894)
[WordPress主题制作全过程教程](https://www.ludou.org/create-wordpress-themes-prepare.html)
[磊子的博客](http://www.favortt.com/)
[露兜博客](https://www.ludou.org/)
