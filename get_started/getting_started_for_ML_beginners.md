# 机器学习初学者用初始教程

本文档旨在阐明如何使用机器学习来根据种类对鸢尾花（Iris Flowers）分类（归类）。本文档将深度解析TensorFlow代码，并以此解释机器学习的基本方法。

如果您符合下列条件，说明您适合学习本教程：
* 您对机器学习一无所知
* 您有意学习如何编写TensorFlow程序
* 您懂（一点）Python编程

如果您已熟悉基本的机器学习概念但又从未接触过TensorFlow，请阅读[机器学习专家用Tensorflow教程](https://www.tensorflow.org/get_started/premade_estimators)

## 鸢尾花分类问题

想像这样一个场景，您是一位植物学家，正在寻找一种方式能够自动分类您所发现的鸢尾花。机器学习提供了许多可以分类花卉的方式。举个例子，一个复杂的机器学习项目可以基于图像来分类花卉。我们的目标稍小一些，只基于鸢尾花[萼片](https://en.wikipedia.org/wiki/Sepal)与[花瓣](https://en.wikipedia.org/wiki/Petal)的长度与宽度来将其分类。

鸢尾花属包含了大概300个种类，但本项目将只专注于一下三种：
* 山鸢尾（Iris setosa）
* 弗吉尼亚鸢尾花（Iris virginica）
* 变色鸢尾（Iris versicolor）

![Iris](https://www.tensorflow.org/images/iris_three_species.jpg)

**From left to right, Iris setosa (by Radomil, CC BY-SA 3.0), Iris versicolor (by Dlanglois, CC BY-SA 3.0), and Iris virginica (by Frank Mayfield, CC BY-SA 2.0).**



好在已经有人创建了[120种鸢尾花数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，包含萼片和花瓣的测度。该数据集是机器学习分类问题的入门典范。（[MNIST database](https://en.wikipedia.org/wiki/MNIST_database)  这一包含手写数字的数据集则是另一个典型）。该鸢尾花数据集的前5项如下：

| 萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 种   |
| -------- | -------- | -------- | -------- | ---- |
| 6.4      | 2.8      | 5.6      | 2.2      | 2    |
| 5.0      | 2.3      | 3.3      | 1.0      | 1    |
| 4.9      | 2.5      | 4.5      | 1.7      | 2    |
| 4.9      | 3.1      | 1.5      | 0.1      | 0    |
| 5.7      | 3.8      | 1.7      | 0.3      | 0    |

我们来介绍一些术语：

* 最后一列（种）我们称作[标签](https://developers.google.com/machine-learning/glossary/#label)；前四列叫做[特征](https://developers.google.com/machine-learning/glossary/#feature)；

