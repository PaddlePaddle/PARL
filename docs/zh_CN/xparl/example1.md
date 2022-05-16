# **示例：通过XPARL 实现线性加速运算**
<p align="center">
<img src="../../parallel_training/poster.png" width="300"/>
</p>

这个教程展示了如何通过并行修饰符`@parl.remote_class`，使用python的**多线程**也能够实现并行计算。

众所周知，python 的多线程并发性能并不好，很难达到传统的编程语言比如C++或者JAVA这种加速效果，主要的原因是python 有全局锁（GIL）的限制，使得其最多只能运用单核来记性运算。

## 串行运算
下面我们通过一个简单的例子来看下GIL对于python的影响。首先，我们跑下这段代码：
```python
class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
a = A()
for _ in range(5):
    a.run()
```

这段代码需要17.46秒的时间来计算5次的从1累加到1亿。

## 多线程计算
接下来我们通过python的原生多线程库改造下上面的代码，让它可以多线程跑起来。
```python
import threading

class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
threads = []
for _ in range(5):
    a = A()
    th = threading.Thread(target=a.run)
    th.start()
    threads.append(th)
for th in threads:
    th.join()
```
运行这段代码之后，居然需要**41.35秒**，比刚才的串行运算速度更慢。主要的原因是GIL限制了python只能单核运算，使用了多线程运算之后，触发了多线程竞争CPU的问题，反而延长了计算时间。

## PARL
```python
import threading
import parl

#这增加一行
@parl.remote_class
class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
threads = []
#这增加一行
parl.connect("localhost:6006")
for _ in range(5):
    a = A()
    th = threading.Thread(target=a.run)
    th.start()
    threads.append(th)
for th in threads:
    th.join()
```
这段代码只需要**4.3秒**就能跑完！PARL在这里做的改动只有两行代码，但是我们却看到了运算速度的极大提升，具体的效果对比可以看下图。

<p align="center">
<img src="../../parallel_training/elapsed_time.jpg" width="500"/>
</p>
