# **如何在xparl中分发本地文件**

文件分发是分布式并行计算的重要功能。它负责把用户的代码还有配置文件分发到不同的机器上，让所有的机器都运行同样的代码进行并行计算。默认情况下，XPARL分发主文件所在目录下，所有py结尾文件。但是有时候用户需要分发一些特定的文件，比如模型文件、配置文件、子目录下的Python代码（用于import的子模块）。为了满足这个需求，parl.connect提供了接口，用户可直接指定需要分发的文件或代码。

### 例子

文件目录结构如下，我们想分发policy文件夹中的py文件。
我们可以在connect的时候传入想要分发的文件到`distributed_files`参数中，该参数支持正则表达式。

```
.
├── main.py
└── policy
    ├── agent.py
    ├── config.ini
    └── __init__.py
```

```python
parl.connect("localhost:8004", distributed_files=['./policy/*.py', './policy/*.ini'])
```
