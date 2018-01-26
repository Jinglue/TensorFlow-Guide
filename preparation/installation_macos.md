# 为 macOS 安装 TensorFlow

> TensorFlow 页面底部上次更新日期：一月 4, 2018

> 本页面内容从[为 Ubuntu 安装 TensorFlow](installation_ubuntu.md) 直接复制并修改，如有纰漏请指出

本教程将指导你如何在 macOS 上安装 TensorFlow。

*注意*：自 1.2 版本起，TensorFlow 将不再为 macOS 提供 GPU 支持。

## 选择一种安装方式

你必须在以下几项中选择一种途径来安装 TensorFlow：

* virtualenv（虚拟环境）

* “原生” pip

* Docker

* Anaconda

* [从源码安装](installation_sources.md)

*我们推荐 virtualenv 安装。*[Virtualenv](https://virtualenv.pypa.io/en/stable/) 是一个虚拟的 Python 环境，它可以与其他 Python 开发项目的环境绝缘，避免程序被同一台机器上的其他 Python 程序干扰。Virtualenv 的安装过程不仅安装了 TensorFlow 本体，也安装了 TensorFlow 依赖的包（实际上非常方便）。要使用 TensorFlow 的时候，你只需要激活虚拟环境就可以了。总之，virtualenv 提供了安装运行 TensorFlow 的简单又可靠的途径。

原生 pip 安装 TensorFlow 会直接安装到你的系统上，不经过任何容器系统。*我们建议系统管理员使用原生 pip 安装，使多用户系统上的每一个用户都可以使用 TensorFlow。*由于原生 pip 安装位置不在一个单独的绝缘的容器中，pip 安装可能会与系统中其他基于 Python 的安装互相干扰。但是如果你熟悉 pip 以及你的 Python 环境，原生 pip 安装一般只需要一行命令。

Docker 则把 TensorFlow 与你机器上现存的包完全绝缘。Docker 容器包含 TensorFlow 以及它全部的依赖。注意，Docker 镜像体积可能非常大（几百 MB）。如果你希望把 TensorFlow 加入到你已用 Docker 的庞大的软件架构中去，那么你可以选择 Docker 安装。

在 Anaconda 中，你可能会用 conda 创建一个虚拟环境，但是我们建议你在 Anaconda 下使用 `pip install` 来安装 TensorFlow，而不是 `conda install`。

*注意*：conda 包是由社区支持的，没有官方支持，也就是说 TensorFlow 团队既没有测试，也没有维护 conda 包。请谨慎使用这个包。

## 使用 virtualenv 安装

通过以下步骤，用 Virtualenv 安装 TensorFlow：

1. 启动终端（也就是 shell），后面的步骤都在这个 shell 中执行。

2. 通过以下两条命令的其中一条，安装 pip 和 virtualenv：

```
$ sudo easy_install pip
$ pip install --upgrade virtualenv 
```

3. 通过以下两条命令的其中一条，创建一个虚拟环境：

```
$ virtualenv --system-site-packages targetDirectory # for Python 2.7
$ virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
```

其中，`targetDirectory` 指定了 virtualenv 文件路径的根目录。本教程的 `targetDirectory` 是 `~/tensorflow`，你也可以指定其他目录。

4. 通过以下两条命令的其中一条，激活虚拟环境：

```
$ source ~/tensorflow/bin/activate      # If using bash, sh, ksh, or zsh
$ source ~/tensorflow/bin/activate.csh  # If using csh or tcsh 
```

上述 `source` 命令应该已经把你的命令提示符变成了：

```
(tensorflow)$ 
```

5. 确保安装的 pip 版本为 8.1 或更高:

```
(tensorflow)$ easy_install -U pip
```

6. 通过以下命令中的其中一条，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
```

如果命令执行成功，跳过第 6 步，如果命令执行失败，执行第 6 步。

7. （可选）如果第 5 步失败了（一般是因为你用的是低于 8.1 的 pip），通过以下命令中的其中一条，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip install --upgrade tfBinaryURL   # Python 2.7
(tensorflow)$ pip3 install --upgrade tfBinaryURL  # Python 3.n 
```

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个页面](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 macOS、Python 2.7，通过以下命令，在已激活的 virtualenv 环境中安装 TensorFlow：

```
$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py2-none-any.whl
```

如果你遇到安装问题，请看[常见安装问题](#common-installation-problems)

### 下一步

安装 TensorFlow 之后，[验证安装](#validate-your-installation)。

注意，你每次使用 TensorFlow 之前都必须激活 virtualenv 环境。如果 virtualenv 环境没有激活，使用以下两条命令中的一条：

```
$ source ~/tensorflow/bin/activate      # bash, sh, ksh, or zsh
$ source ~/tensorflow/bin/activate.csh  # csh or tcsh
```

当 virtualenv 环境激活以后，你就可以从当前的 shell 中运行 TensorFlow 程序。你的提示符会提示你 tensorflow 环境已经激活：

```
(tensorflow)$ 
```

你可以通过以下命令，取消激活：

```
(tensorflow)$ deactivate 
```

提示符又会变回默认（由 `PS1` 环境变量定义）。

### 卸载 TensorFlow

要卸载 TensorFlow，你只需要删除你的文件路径，比如：

```
$ rm -r ~/tensorflow 
```

## 使用原生 pip 安装

通过 pip 安装 TensorFlow 有两种方法，一个比较简单，一个相对复杂。

*注意*：[需要的包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)中列出了 pip 会安装或升级的 TensorFlow 包。

### 前提条件：Python

要安装 TensorFlow，确保系统中安装的 Python 版本是否符合以下要求：

* Python 2.7

* Python 3.3+

如果你的系统没有安装以上版本的 Python，[现在安装](https://wiki.python.org/moin/BeginnersGuide/Download)。

安装 Python 时，你必须禁用 System Integrity Protection (SIP) 来允许从 Mac App Store 以外的来源安装软件。

### 前提条件：pip

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) 是用来安装和管理 Python 软件包的，如果你想使用原生 pip 安装，你的系统中必须有下列 pip 版本之一：

* `pip` 用于 Python 2.7

* `pip3` 用于 Python 3.n

在你安装 Python 的时候，你的系统可能已经装上了 `pip` 或 `pip3`，要辨别系统上安装的是 pip 还是 pip
3，使用以下命令之一：

```
$ pip -V  # for Python 2.7
$ pip3 -V # for Python 3.n 
```

我们强烈推荐 8.1 及更高版本的 pip 或 pip3。如果没有安装 8.1 或者更高版本，通过以下两条命令的其中一条安装 pip，或者将 pip 升级到最新版本：

```
$ sudo easy_install --upgrade pip
$ sudo easy_install --upgrade six 
```

### 安装 TensorFlow

假设前述先决条件的软件你已经在你的 Mac 上已经安装了，执行以下步骤：

1. 通过以下命令中的其中一条安装 TensorFlow：

```
$ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
$ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
```

如果上述命令执行成功，你现在可以[验证安装](#validate-your-installation)。

2. （可选）如果第一步失败了，通过以下两条命令的其中一条安装 TensorFlow：

```
$ sudo pip  install --upgrade tfBinaryURL   # Python 2.7
$ sudo pip3 install --upgrade tfBinaryURL   # Python 3.n 
```

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个页面](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 macOS、Python 2.7，通过以下命令安装 TensorFlow：

```
$ sudo pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py2-none-any.whl 
```

如果你遇到安装问题，请看[常见安装问题](#common-installation-problems)

### 下一步

安装 TensorFlow 之后，[验证安装](#validate-your-installation)。

### 卸载 TensorFlow

通过以下两条命令的其中一条，卸载 TensorFlow：

```
$ pip uninstall tensorflow
$ pip3 uninstall tensorflow 
```

## 使用 Docker 安装

通过以下步骤来使用 Docker 安装 TensorFlow：

1. 按照[Docker 官方文档](http://docs.docker.com/engine/installation/)，在你的机器上安装 Docker。

2. 启动一个包含 [TensorFlow binary images](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 里的其中一个镜像的 Docker 容器。

本节剩余部分将指导你如何启动一个 docker 镜像。

要启动一个仅支持 CPU 的 Docker 镜像（也就是不具有 GPU 支持），输入一条命令格式如下：

```
$ docker run -it -p hostPort:containerPort TensorFlowImage
```

其中：

* _-p_ hostPort:containerPort 是可选参数。如果你计划从 shell 中运行 TensorFlow 程序，忽略这个参数。如果你计划用 Jupyter Notebook 运行，把 _hostPort_ 和 _containerPort_ 都设为 8888。如果你想要在容器中运行 TensorBoard，再加一个 `-p`，在另加的 `-p` 中把 _hostPort_ 和 _containerPort_ 都设为 6006。

* _TensorFlowCPUImage_ 是必需的，它确定了这个 Docker 容器的镜像。可以从以下几个值中指定：

  * `gcr.io/tensorflow/tensorflow`，TensorFlow CPU 镜像。

  * `gcr.io/tensorflow/tensorflow:latest-devel`，最新版本的 TensorFlow CPU 镜像加源代码。

  `gcr.io` 是谷歌容器表。注意有些镜像可以在 [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/) 中找到。

比如，以下命令会在 Docker 容器中启动最新版本的 TensorFlow CPU 镜像，让你可以在 shell 中运行 TensorFlow 程序：

```
$ docker run -it gcr.io/tensorflow/tensorflow bash
```

以下命令也会在 Docker 容器中启动最新版本的 TensorFlow CPU 镜像。但是这个容器允许你在 Jupyter Notebook 中运行 TensorFlow 程序：

```
$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

初次运行时 Docker 会下载 TensorFlow 镜像。

### 下一步

现在你应该[验证安装](#validate-your-installation)。

## 使用 Anaconda 安装

按照以下步骤在 Anaconda 环境下安装 TensorFlow：

1. 按照 [Anaconda 下载站](https://www.continuum.io/downloads)的指导下载安装 Anaconda。

2. 通过以下命令，创造一个 conda 环境名为 tensorflow 来运行某个版本的 Python：

```
$ conda create -n tensorflow python=2.7 # or python=3.3, etc.
```

3. 通过以下命令，激活 conda 环境：

```
$ source activate tensorflow
(tensorflow)$  # 你的提示符应该发生了变化
```

4. 通过以下命令，在你的 conda 环境下安装 TensorFlow：

```
(tensorflow)$ pip install --ignore-installed --upgrade tfBinaryURL
```

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个表](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 macOS、Python 2.7，通过以下命令，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip install --ignore-installed --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py2-none-any.whl
```

## 验证安装 {#validate-your-installation}

要验证你的 TensorFlow 安装，执行以下步骤：

1. 确保环境已经准备好运行 TensorFlow 程序。

2. 运行一个简短的 TensorFlow 程序。

### 准备你的环境

如果你是使用原生 pip、virtualenv 或 Anaconda 方式安装的，执行下列步骤：

1. 运行 terminal。

2. 如果是使用 virtualenv 或者 Anaconda 安装的，启动你的容器。

3. 如果安装了 TensorFlow 源代码，将目录切换到任意一个不包含 TensorFlow 源代码的目录。

如果你是通过 Docker 安装的，启动一个可以运行 bash 的 Docker 容器，比如：

```
$ docker run -it gcr.io/tensorflow/tensorflow bash
```

### 运行一段简短的 TensorFlow 程序

在 shell 中调用 python：

```
$ python
```

在 python 交互 shell 中输入下列简短程序：

```
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

如果系统输出下列内容，就说明你已经准备就绪了：

```
Hello, TensorFlow!
```

如果你是 TensorFlow 新手，可以看这篇[教程](../get_started/tf_intro.md)。

如果系统输出的不是问候，而是错误消息，请看[常见安装问题](#common-installation-problems)。

## 常见安装问题 {#common-installation-problems}

我们以 Stack Overflow 上的问题和回答作为常见安装问题以及解决方法。下表包含 Stack Overflow 关于常见安装问题的回答的链接。如果你遇到下表中没有的错误信息或者其他安装问题，你可以在 Stack Overflow 上搜索。如果 Stack Overflow 没有相关内容，在 Stack Overflow 上提问，并加上 `tensorflow` 标签。

| 链接                                       | 错误信息                                     |
| ---------------------------------------- | ---------------------------------------- |
| [42006320](http://stackoverflow.com/q/42006320) | ```ImportError: Traceback (most recent call last): File ".../tensorflow/core/framework/graph_pb2.py", line 6, in  from google.protobuf import descriptor as _descriptor ImportError: cannot import name 'descriptor'``` |
| [33623453](https://stackoverflow.com/q/33623453) | ```IOError: [Errno 2] No such file or directory:   '/tmp/pip-o6Tpui-build/setup.py'``` |
| [35190574](https://stackoverflow.com/questions/35190574) | ```SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify   failed``` |
| [42009190](http://stackoverflow.com/q/42009190) | ```Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow   Found existing installation: setuptools 1.1.6   Uninstalling setuptools-1.1.6:   Exception:  ...   [Errno 1] Operation not permitted:   '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' ``` |
| [33622019](https://stackoverflow.com/q/33622019) | ```ImportError: No module named copyreg``` |
| [37810228](http://stackoverflow.com/q/37810228) | pip 安装时系统提示：```OSError: [Errno 1] Operation not permitted``` |
| [33622842](http://stackoverflow.com/q/33622842) | pip 安装时`import tensorflow` 触发错误：```Traceback (most recent call last):   File "", line 1, in    File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py",     line 4, in      from tensorflow.python import *     ...   File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py",     line 22, in      serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02       \x03(\x0b\x32       .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01       \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')   TypeError: __init__() got an unexpected keyword argument 'syntax'``` |
| [42075397](http://stackoverflow.com/q/42075397) | `pip install` 命令触发以下错误：```... You have not agreed to the Xcode license agreements, please run 'xcodebuild -license' (for user-level acceptance) or 'sudo xcodebuild -license' (for system-wide acceptance) from within a Terminal window to review and agree to the Xcode license agreements. ...   File "numpy/core/setup.py", line 653, in get_mathlib_info      raise RuntimeError("Broken toolchain: cannot link a simple C program")  RuntimeError: Broken toolchain: cannot link a simple C program``` |

## TensorFlow Python 包的 URL {#url-of-the-tensorflow-python-package}

一些安装方法需要 TensorFlow Python 包的 URL，URL 的选择，取决于下列三个因素：

* 操作系统

* Python 版本

### Python 2.7

```
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py2-none-any.whl
```

### Python 3.4、3.5 或者 3.6

```
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py3-none-any.whl
```

## Protobuf pip package 3.1 {#protobuf-pip-package-31}

如果你没有遇到与 protobuf pip package 相关的问题，你可以跳过这一节。

*注意*：如果你的 TensorFlow 程序运行缓慢，你可能是遇到了 protobuf pip package 相关的问题。

TensorFlow pip package 依赖 protobuf pip package 3.1 版本。从 PyPI （使用 pip install protobuf 调用）下载的 protobuf pip package 是一个纯 Python 库，用 Python 实现的 proto 序列化、反序列化的速度，能比 C++ 实现慢 10 到 50 倍。Protobuf 也支持用高速的 C++ 翻译器扩展 Python 包，纯 Python 的库中没有这个扩展。我们已经创建了一个包含扩展的 pip 包，要安装这个包，调用以下命令中的一条：

* Python 2.7：

```
$ pip install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/protobuf-3.1.0-cp27-none-macosx_10_11_x86_64.whl
```

* Python 3.n：

```
$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/protobuf-3.1.0-cp35-none-macosx_10_11_x86_64.whl
```

安装这个 protobuf 包会覆盖已有的 protobuf 包。注意，这个 pip 包支持超过 64MB 的 protobuf，可以解决以下问题：

```
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207]
A protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
```
