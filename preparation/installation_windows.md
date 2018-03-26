# 为 Windows 安装 TensorFlow

> 官方最新版本：三月 9, 2018  
> 本文对应版本：一月 4, 2018

> 本页面内容从[为 Ubuntu 安装 TensorFlow](installation_ubuntu.md) 直接复制并修改，如有纰漏请指出

本教程将指导你如何在 Windows 上安装 TensorFlow。

## 选择一个 TensorFlow 版本

你必须在以下几种 TensorFlow 中选择一种来安装：

* **仅支持 CPU 的 TensorFlow**，如果你的系统不具有 NVIDIA 的 GPU，你必须安装这个版本。注意，该版本的 TensorFlow 非常易于安装（一般只需 5 到 10 分钟），所以即使你有 NVIDIA GPU，我们也建议你先安装这个版本。

* **具有 GPU 支持的 TensorFlow**，TensorFlow 程序在 GPU 上的运行速度通常优于在 CPU 上，因此，如果你的系统中有符合后文要求的 NVIDIA GPU，而且你想要运行重视性能的应用程序的话，你应该安装这个版本。

### 将 NVIDIA 用于运行 TensorFlow（具有 GPU 支持） 的要求 {#nvidia-requirements}

如果你通过本教程中提供的途径安装了具有 GPU 支持的 TensorFlow，你必须在你的系统中安装以下 NVIDIA 软件：

* CUDA® 工具包 8.0，具体请查看 [NVIDIA 官方文档](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)。请确保你已按照 NVIDIA 官方文档中的指导，在 `%PATH%` 环境变量中添加了 Cuda 的相关路径名。

* 与 CUDA 工具包 8.0 相关的 NVIDIA 驱动。

* cuDNN v6.0，具体请查看 [NVIDIA 官方文档](https://developer.nvidia.com/cudnn)。注意，cuDNN 一般会装在和其他 CUDA DLL 不同的位置。请确保你已按照 NVIDIA 官方文档中的指导，在 `%PATH%` 环境变量中添加了 安装 cuDNN 的路径。

* 支持 CUDA 运算功能 3.0 或更高的显卡，[NVIDIA 官方页面](https://developer.nvidia.com/cuda-gpus)提供了支持的显卡列表。

如果你安装了上述软件的不同版本，请换到指定的版本。实际上 cuDNN 的版本必须完全匹配，否则 TensorFlow 找不到 `cuDNN64_6.dll` 的话就无法加载。要使用不同版本的 cuDNN，你必须从源码安装。

## 选择一种安装方式

你必须在以下几项中选择一种途径来安装 TensorFlow：

* “原生” pip

* Anaconda

原生 pip 安装 TensorFlow 会直接安装到你的系统上，不经过任何容器系统。*我们建议系统管理员使用原生 pip 安装，使多用户系统上的每一个用户都可以使用 TensorFlow。*由于原生 pip 安装位置不在一个单独的绝缘的容器中，pip 安装可能会与系统中其他基于 Python 的安装互相干扰。但是如果你熟悉 pip 以及你的 Python 环境，原生 pip 安装一般只需要一行命令。

在 Anaconda 中，你可能会用 conda 创建一个虚拟环境，但是我们建议你在 Anaconda 下使用 `pip install` 来安装 TensorFlow，而不是 `conda install`。

*注意*：conda 包是由社区支持的，没有官方支持，也就是说 TensorFlow 团队既没有测试，也没有维护 conda 包。请谨慎使用这个包。

## 使用原生 pip 安装

如果系统中没有安装以下版本的 Python，现在安装：

* [Python 3.5.x 64-bit](https://www.python.org/downloads/release/python-352/)

* [Python 3.6.x 64-bit](https://www.python.org/downloads/release/python-362/)

Windows 上的 TensorFlow 支持 Python 3.5.x 和 Python 3.6.x。注意 Python 3 自带 pip3 包管理工具，你可以用它来安装 TensorFlow。

要安装 TensorFlow，启动终端，再输入对应的 `pip3 install` 命令。要安装仅支持 CPU 的 TensorFlow，输入以下命令：

```
C:\> pip3 install --upgrade tensorflow
```

要安装支持 GPU 的 TensorFlow，输入以下命令：

```
C:\> pip3 install --upgrade tensorflow-gpu
```

## 使用 Anaconda 安装

*Anaconda 安装是由社区支持的，没有官方支持。*

按照以下步骤在 Anaconda 环境下安装 TensorFlow：

1. 按照 [Anaconda 下载站](https://www.continuum.io/downloads)的指导下载安装 Anaconda。

2. 通过以下命令，创造一个 conda 环境名为 tensorflow 来运行某个版本的 Python：

```
C:> conda create -n tensorflow python=3.5
```

3. 通过以下命令，激活 conda 环境：

```
C:> activate tensorflow
(tensorflow)C:>  # 你的提示符应该发生了变化
```

4. 输入对应的 `pip3 install` 命令，在你的 conda 环境安装 TensorFlow。要安装仅支持 CPU 的 TensorFlow，输入以下命令：

```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow
```

要安装支持 GPU 的 TensorFlow，输入以下命令（单行）：

```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu
```

## 验证安装 {#validate-your-installation}

运行 terminal。

如果是使用 Anaconda 安装的，激活你的环境。

在 shell 中调用 python：

```
$ python
```

在 python 交互 shell 中输入下列简短程序：

```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

如果系统输出下列内容，就说明你已经准备就绪了：

```
Hello, TensorFlow!
```

如果你是 TensorFlow 新手，可以看这篇[教程](../get_started/tf_intro.md)。

如果系统输出的不是问候，而是错误消息，请看[常见安装问题](#common-installation-problems)。

这里有一份[在 Windows 上安装 TensorFlow 的脚本](https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c)。

## 常见安装问题 {#common-installation-problems}

我们以 Stack Overflow 上的问题和回答作为常见安装问题以及解决方法。下表包含 Stack Overflow 关于常见安装问题的回答的链接。如果你遇到下表中没有的错误信息或者其他安装问题，你可以在 Stack Overflow 上搜索。如果 Stack Overflow 没有相关内容，在 Stack Overflow 上提问，并加上 `tensorflow` 标签。

| 链接                                       | 错误信息                                     |
| ---------------------------------------- | ---------------------------------------- |
| [41007279](https://stackoverflow.com/q/41007279) | ```[...\stream_executor\dso_loader.cc] Couldn't open CUDA library nvcuda.dll``` |
| [41007279](https://stackoverflow.com/q/41007279) | ```[...\stream_executor\cuda\cuda_dnn.cc] Unable to load cuDNN DSO``` |
| [42006320](http://stackoverflow.com/q/42006320) | ```ImportError: Traceback (most recent call last): File ".../tensorflow/core/framework/graph_pb2.py", line 6, in  from google.protobuf import descriptor as _descriptor ImportError: cannot import name 'descriptor'``` |
| [42011070](https://stackoverflow.com/q/42011070) | ```No module named "pywrap_tensorflow"``` |
| [42217532](https://stackoverflow.com/q/42217532) | ```OpKernel ('op: "BestSplits" device_type: "CPU"') for unknown op: BestSplits``` |
| [43134753](https://stackoverflow.com/q/43134753) | ```The TensorFlow library wasn't compiled to use SSE instructions``` |