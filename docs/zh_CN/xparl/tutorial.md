# **使用教程**
---
## 配置命令 
这个教程将会演示如何搭建一个集群。

搭建一个PARL集群，可以通过执行下面两个`xparl`命令：

### 启动集群
```bash
xparl start --port 6006
```

这个命令会启动一个主节点（master）来管理集群的计算资源，同时会把本地机器的CPU资源加入到集群中。命令中的6006端口只是作为示例，你可以修改成任何有效的端口。

### 加入其它机器资源
> 注意：如果你只有单台机器，可以忽略这部分教程。

如果你想加入更多的CPU计算资源到集群中，可以在其他机器上运行下面命令：

```bash
 xparl connect --address [MASTER_ADDRESS]:6006
```
它会启动一个工作节点（worker），并把当前机器的CPU资源加入到该master对应的集群。worker默认会把所有的CPU资源加入到集群中，如果你需要指定worker可使用的CPU数量，可以在上述命令上加入选项`--cpu_num [CPU_NUM]` (例如：----cpu_num 10)。

注意：启动集群后，你可以在任何时候和任何机器上，通过执行`xparl connect`命令把更多CPU资源加入到集群中。

## 示例
这里我们给出了一个示例来演示如何通过`@parl.remote_class`来进行并行计算。

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
actor.add(1, 2)  # 返回 3
```

## 关闭集群
在master机器上运行`xparl stop`命令即可关闭集群程序。当master节点退出后，运行在其他机器的worker节点也会自动退出并结束相关程序。

## 扩展阅读
我们现在已经知道了如何搭建一个集群，以及如何通过修饰符`@parl.remote_class`来使用集群。

在[下一个教程](./example.md)我们将会演示如何通过这个修饰符来打破Python的全局解释器锁（Global Interpreter Lock, GIL）限制，从而实现真正的多线程计算。
