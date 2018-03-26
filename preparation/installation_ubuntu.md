# 为 Ubuntu 安装 TensorFlow

> 官方最新版本：三月 9, 2018  
> 本文对应版本：一月 4, 2018

本教程将指导你如何在 Ubuntu 上安装 TensorFlow。教程可能也适用于其他 Linux 系统，但我们只在 Ubuntu 14.04 及更高版本上通过了测试，并且只对 Ubuntu 14.04 及更高版本提供支持。

## 选择一个 TensorFlow 版本

你必须在以下几种 TensorFlow 中选择一种来安装：

* **仅支持 CPU 的 TensorFlow**，如果你的系统不具有 NVIDIA 的 GPU，你必须安装这个版本。注意，该版本的 TensorFlow 非常易于安装（一般只需 5 到 10 分钟），所以即使你有 NVIDIA GPU，我们也建议你先安装这个版本。

* **具有 GPU 支持的 TensorFlow**，TensorFlow 程序在 GPU 上的运行速度通常优于在 CPU 上，因此，如果你的系统中有符合后文要求的 NVIDIA GPU，而且你想要运行重视性能的应用程序的话，你应该安装这个版本。

### 将 NVIDIA 用于运行 TensorFlow（具有 GPU 支持） 的要求 {#nvidia-requirements}

如果你通过本教程中提供的途径安装了具有 GPU 支持的 TensorFlow，你必须在你的系统中安装以下 NVIDIA 软件：

* CUDA® 工具包 8.0，具体请查看 [NVIDIA 官方文档](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)。请确保你已按照 NVIDIA 官方文档中的指导，在 `LD_LIBRARY_PATH` 环境变量中添加了 Cuda 的路径。

* 与 CUDA 工具包 8.0 相关的 NVIDIA 驱动。

* cuDNN v6.0，具体请查看 [NVIDIA 官方文档](https://developer.nvidia.com/cudnn)。请确保你已按照 NVIDIA 官方文档中的指导，创建了 `CUDA_HOME` 环境变量。

* 支持 CUDA 运算功能 3.0 或更高的显卡，[NVIDIA 官方页面](https://developer.nvidia.com/cuda-gpus)提供了支持的显卡列表。

* libcupti-dev 库，即 NVIDIA CUDA 配置工具接口，用于提供高级配置的支持。使用以下命令安装：
`$ sudo apt-get install libcupti-dev`

如果你安装了上述软件的旧版本，请升级到指定的版本。如果无法升级，通过以下方式你依然可以运行具有 GPU 支持的 TensorFlow：

* 按照文档[从源码安装 TensorFlow](installation_sources.md)来安装 TensorFlow。

* 安装或升级到至少下列版本的 NVIDIA 软件：

  * CUDA 工具包 7.0 或更高

  * cuDNN v3 或更高

  * 支持 CUDA 运算功能 3.0 或更高的显卡

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

1. 通过以下两条命令的其中一条，安装 pip 和 virtualenv：

```
$ sudo apt-get install python-pip python-dev python-virtualenv # for Python 2.7
$ sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
```

2. 通过以下两条命令的其中一条，创建一个虚拟环境：

```
$ virtualenv --system-site-packages targetDirectory # for Python 2.7
$ virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
```

其中，`targetDirectory` 指定了 virtualenv 文件路径的根目录。本教程的 `targetDirectory` 是 `~/tensorflow`，你也可以指定其他目录。

3. 通过以下两条命令的其中一条，激活虚拟环境：

```
$ source ~/tensorflow/bin/activate # bash, sh, ksh, or zsh
$ source ~/tensorflow/bin/activate.csh  # csh or tcsh
```

上述 `source` 命令应该已经把你的命令提示符变成了：

```
(tensorflow)$ 
```

4. 确保安装的 pip 版本为 8.1 或更高:

```
(tensorflow)$ easy_install -U pip
```

5. 通过以下命令中的其中一条，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
(tensorflow)$ pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
(tensorflow)$ pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU
```

如果命令执行成功，跳过第 6 步，如果命令执行失败，执行第 6 步。

6. （可选）如果第 5 步失败了（一般是因为你用的是低于 8.1 的 pip），通过以下命令中的其中一条，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip install --upgrade tfBinaryURL   # Python 2.7
(tensorflow)$ pip3 install --upgrade tfBinaryURL  # Python 3.n 
```

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个页面](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 Linux、Python 3.4、仅支持 CPU，通过以下命令，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp34-cp34m-linux_x86_64.whl
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
$ rm -r targetDirectory
```

## 使用原生 pip 安装

通过 pip 安装 TensorFlow 有两种方法，一个比较简单，一个相对复杂。

*注意*：[需要的包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)中列出了 pip 会安装或升级的 TensorFlow 包。

### 前提条件：Python 和 Pip

Ubuntu 内置了 Python，先通过 `python -V` 命令确认一下系统中安装的 Python 版本是否符合以下要求：

* Python 2.7

* Python 3.4+

pip 或 pip3 包管理工具通常在 Ubuntu 上已经预装了，先通过 `pip -V` 或者 `pip3 -V` 命令确认一下安装的版本。我们强烈推荐 8.1 及更高版本的 pip 或 pip3。如果没有安装 8.1 或者更高版本，通过以下两条命令的其中一条安装 pip，或者将 pip 升级到最新版本：

```
$ sudo apt-get install python-pip python-dev   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev # for Python 3.n
```

### 安装 TensorFlow

假设前述先决条件的软件你的 Linux 上已经安装了，执行以下步骤：

1. 通过以下命令中的其中一条安装 TensorFlow：

```
$ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
$ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
$ pip install tensorflow-gpu  # Python 2.7; GPU support
$ pip3 install tensorflow-gpu # Python 3.n; GPU support 
```

如果上述命令执行成功，你现在可以[验证安装](#validate-your-installation)。

2. （可选）如果第一步失败了，通过以下两条命令的其中一条安装 TensorFlow：

```
$ sudo pip  install --upgrade tfBinaryURL   # Python 2.7
$ sudo pip3 install --upgrade tfBinaryURL   # Python 3.n 
```

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个页面](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 Linux、Python 3.4、仅支持 CPU，通过以下命令安装 TensorFlow：

```
(tensorflow)$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp34-cp34m-linux_x86_64.whl
```

如果你遇到安装问题，请看[常见安装问题](#common-installation-problems)

### 下一步

安装 TensorFlow 之后，[验证安装](#validate-your-installation)。

### 卸载 TensorFlow

通过以下两条命令的其中一条，卸载 TensorFlow：

```
$ sudo pip uninstall tensorflow  # for Python 2.7
$ sudo pip3 uninstall tensorflow # for Python 3.n
```

## 使用 Docker 安装

通过以下步骤来使用 Docker 安装 TensorFlow：

1. 按照[Docker 官方文档](http://docs.docker.com/engine/installation/)，在你的机器上安装 Docker。

2. 可选，按照[Docker 官方文档](https://docs.docker.com/engine/installation/linux/linux-postinstall/)，创建一个名为 `docker` 的 Linux 用户组，允许启动容器时不使用 sudo 命令。(如果跳过这一步，你每一次调用 Docker 都将需要 sudo）。

3. 要安装支持 GPU 的 TensorFlow，你必须先安装 github 上的[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

4. 启动一个包含 [TensorFlow binary images](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 里的其中一个镜像的 Docker 容器。

本节剩余部分将指导你如何启动一个 docker 镜像。

### 仅支持 CPU

要启动一个仅支持 CPU 的 Docker 镜像（也就是不具有 GPU 支持），输入一条命令格式如下：

```
$ docker run -it -p hostPort:containerPort TensorFlowCPUImage
```

其中：

* _-p_ hostPort:containerPort 是可选参数。如果你计划从 shell 中运行 TensorFlow 程序，忽略这个参数。如果你计划用 Jupyter Notebook 运行，把 _hostPort_ 和 _containerPort_ 都设为 8888。如果你想要在容器中运行 TensorBoard，再加一个 `-p`，在另加的 `-p` 中把 _hostPort_ 和 _containerPort_ 都设为 6006。

* _TensorFlowCPUImage_ 是必需的，它确定了这个 Docker 容器的镜像。可以从以下几个值中指定：

  * `gcr.io/tensorflow/tensorflow`，TensorFlow CPU 镜像。

  * `gcr.io/tensorflow/tensorflow:latest-devel`，最新版本的 TensorFlow CPU 镜像加源代码。

  * `gcr.io/tensorflow/tensorflow:version`，指定版本的 TensorFlow CPU 镜像（比如 1.1.0rc1）。

  * `gcr.io/tensorflow/tensorflow:version-devel`，指定版本的 TensorFlow CPU 镜像（比如 1.1.0rc1）加源代码。

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

### GPU 支持

比起安装具有 GPU 支持的 TensorFlow，你应该先确保你的系统符合[NVIDIA 软件要求](#nvidia-requirements)。要启动具有 Nvidia GPU 支持的 docker 容器，输入如下格式的命令：

```
$ nvidia-docker run -it -p hostPort:containerPort TensorFlowGPUImage
```

其中：

* _-p_ hostPort:containerPort 是可选参数。如果你计划从 shell 中运行 TensorFlow 程序，忽略这个参数。如果你计划用 Jupyter Notebook 运行，把 _hostPort_ 和 _containerPort_ 都设为 8888。如果你想要在容器中运行 TensorBoard，再加一个 `-p`，在另加的 `-p` 中把 _hostPort_ 和 _containerPort_ 都设为 6006。

* _TensorFlowCPUImage_ 是必需的，它确定了这个 Docker 容器的镜像。可以从以下几个值中指定：

  * `gcr.io/tensorflow/tensorflow:latest-gpu`，TensorFlow GPU 镜像。

  * `gcr.io/tensorflow/tensorflow:latest-devel-gpu`，最新版本的 TensorFlow GPU 镜像加源代码。

  * `gcr.io/tensorflow/tensorflow:version-gpu`，指定版本的 TensorFlow GPU 镜像（比如 0.12.1）。

  * `gcr.io/tensorflow/tensorflow:version-devel-gpu`，指定版本的 TensorFlow GPU 镜像（比如 0.12.1）加源代码。

我们建议安装其中一个最新版本。比如以下命令会在 Docker 容器中启动最新版本的 TensorFlow GPU 镜像，让你可以在 shell 中运行 TensorFlow 程序：

```
$ nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash
```

以下命令也会在 Docker 容器中启动最新版本的 TensorFlow GPU 镜像。但是这个容器允许你在 Jupyter Notebook 中运行 TensorFlow 程序：

```
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu
```

以下命令会安装一个旧版本的 TensorFlow（0.12.1）：

```
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:0.12.1-gpu
```

初次运行时 Docker 会下载 TensorFlow 镜像。更多细节请见 [TensorFlow docker 须知](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)。

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

其中的 `tfBinaryURL` 指定了 TensorFlow Python 包的 URL。`tfBinaryURL` 的合适值取决于操作系统、Python 版本，以及 GPU 支持。你可以从[这个表](#url-of-the-tensorflow-python-package)找到适用你的系统的值。例如你的情况是 Linux、Python 3.4、仅支持 CPU，通过以下命令，在已激活的 virtualenv 环境中安装 TensorFlow：

```
(tensorflow)$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp34-cp34m-linux_x86_64.whl
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
| [36159194](https://stackoverflow.com/q/36159194) | ImportError: libcudart.so.Version: cannot open shared object file:   No such file or directory |
| [41991101](https://stackoverflow.com/q/41991101) | ImportError: libcudnn.Version: cannot open shared object file:   No such file or directory |
| [36371137](http://stackoverflow.com/q/36371137)and [here](#protobuf-pip-package-31) | libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A   protocol message was rejected because it was too big (more than 67108864 bytes).   To increase the limit (or to disable these warnings), see   CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h. |
| [35252888](https://stackoverflow.com/q/35252888) | Error importing tensorflow. Unless you are using bazel, you should   not try to import tensorflow from its source directory; please exit the   tensorflow source tree, and relaunch your python interpreter from   there. |
| [33623453](https://stackoverflow.com/q/33623453) | IOError: [Errno 2] No such file or directory:   '/tmp/pip-o6Tpui-build/setup.py' |
| [42006320](http://stackoverflow.com/q/42006320) | ImportError: Traceback (most recent call last):   File ".../tensorflow/core/framework/graph_pb2.py", line 6, in    from google.protobuf import descriptor as _descriptor   ImportError: cannot import name 'descriptor' |
| [35190574](https://stackoverflow.com/questions/35190574) | SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify   failed |
| [42009190](http://stackoverflow.com/q/42009190) | Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow   Found existing installation: setuptools 1.1.6   Uninstalling setuptools-1.1.6:   Exception:   ...   [Errno 1] Operation not permitted:   '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' |
| [36933958](http://stackoverflow.com/questions/36933958) | ...   Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow   Found existing installation: setuptools 1.1.6   Uninstalling setuptools-1.1.6:   Exception:   ...   [Errno 1] Operation not permitted:   '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/    Versions/2.7/Extras/lib/python/_markerlib' |

## TensorFlow Python 包的 URL {#url-of-the-tensorflow-python-package}

一些安装方法需要 TensorFlow Python 包的 URL，URL 的选择，取决于下列三个因素：

* 操作系统

* Python 版本

* 是否支持 GPU

### Python 2.7

仅支持 CPU：

```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl
```

支持 GPU：

```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp27-none-linux_x86_64.whl
```

注意，GPU 支持需要的 NVIDIA 硬件和软件在[此处](#nvidia-requirements)已经列出。

### Python 3.4

仅支持 CPU：

```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp34-cp34m-linux_x86_64.whl
```

支持 GPU：

```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp34-cp34m-linux_x86_64.whl
```

注意，GPU 支持需要的 NVIDIA 硬件和软件在[此处](#nvidia-requirements)已经列出。

### Python 3.5

仅支持 CPU：

```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl
```

支持 GPU：

```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl
```

注意，GPU 支持需要的 NVIDIA 硬件和软件在[此处](#nvidia-requirements)已经列出。

### Python 3.6

仅支持 CPU：

```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
```

支持 GPU：

```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl
```

注意，GPU 支持需要的 NVIDIA 硬件和软件在[此处](#nvidia-requirements)已经列出。

## Protobuf pip package 3.1 {#protobuf-pip-package-31}

如果你没有遇到与 protobuf pip package 相关的问题，你可以跳过这一节。

*注意*：如果你的 TensorFlow 程序运行缓慢，你可能是遇到了 protobuf pip package 相关的问题。

TensorFlow pip package 依赖 protobuf pip package 3.1 版本。从 PyPI （使用 pip install protobuf 调用）下载的 protobuf pip package 是一个纯 Python 库，用 Python 实现的 proto 序列化、反序列化的速度，能比 C++ 实现慢 10 到 50 倍。Protobuf 也支持用高速的 C++ 翻译器扩展 Python 包，纯 Python 的库中没有这个扩展。我们已经创建了一个包含扩展的 pip 包，要安装这个包，调用以下命令中的一条：

* Python 2.7：

```
$ pip install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp27-none-linux_x86_64.whl
```

* Python 3.5：

```
$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp35-none-linux_x86_64.whl
```

安装这个 protobuf 包会覆盖已有的 protobuf 包。注意，这个 pip 包支持超过 64MB 的 protobuf，可以解决以下问题：

```
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207]
A protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
```
