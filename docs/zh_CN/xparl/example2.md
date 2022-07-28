# **示例：通过XPARL 实现线性加速运算(后台模式)**

这个教程展示了如何通过并行修饰符`@parl.remote_class`，在**不使用**python的多线程情况下实现并行计算。

在上一个示例中，我们用PARL的修饰符和多线程实现了并行计算，但实际上PARL提供了一种更加简洁的方法来实现并行计算，无需手动创建函数。和上一个教程的区别是在修饰符中添加`wait=false`参数，这样执行函数的时候会立刻得到一个future对象，程序并不会阻塞在当前函数，后续可以通过调用future对象的`get`函数获取到执行结果。

在上一个示例中，我们实现的并行计算如下所示。

```python
import threading
import parl

@parl.remote_class
class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
threads = []
parl.connect("localhost:6006")
for _ in range(5):
    a = A()
    th = threading.Thread(target=a.run)
    th.start()
    threads.append(th)
for th in threads:
    th.join()
```

现在我们来看一下如何在**不使用**python的多线程情况下实现并行计算。

```python

import parl

@parl.remote_class(wait=False)
class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
        return ans

parl.connect("localhost:6006")
actors = [A() for _ in range(5)]
jobs = [actor.run() for actor in actors]
returns = [job.get() for job in jobs]

true_result = sum([i for i in range(100000000)])
for result in returns:
    assert result == true_result
```

这里有两点需要注意的地方：

1. 加入`wait=False`后，actor的函数调用不会阻塞主程序。

2. 在每个actor运行起来后，调用`job.get()`会阻塞当前程序直到job对应函数运行结束，并得到返回的结果。
