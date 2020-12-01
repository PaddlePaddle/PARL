# **如何在xparl中debug**

经过并行修饰符修饰的类，并没有在本地运行，而是跑在了集群上，相应地，我们也没法在本机上看到打印的log，比如之前的代码。
```python
import parl

@parl.remote_class
class Actor(object):
  def hello_world(self):
      print("Hello world.")

  def add(self, a, b):
      return a + b

# 连接到集群（master节点）
parl.connect("localhost:6006")

actor = Actor()
actor.hello_world()# 因为计算是放在集群中执行，所以这里不会打印信息
```

这种情况下，我们应该怎么debug，定位问题呢？
这里推荐两个方案：

- 注释并行修饰符
先不在集群上跑并行，而是在本地跑起来，根据输出的日志debug，调试通过后再增加并行修饰符。但是这种方法在静态图的神经网络框架中可能会引发静态图重复定义的问题，所以在用paddle或者tensorflow的时候不建议采用这种方法。

- 根据xparl的日志服务查看
在本地脚本连接到xparl集群之后，xparl会在程序中输出logserver的地址，通过浏览器访问这个网站即可实时查看每个并行任务的对应输出。

<img src="./.images/log_server.png" width="500"/>
