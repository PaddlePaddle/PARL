# Regression MAML/MAML++ with PARL

Implementation of [MAML](https://arxiv.org/abs/1703.03400) and [MAML++](https://arxiv.org/abs/1810.09502) with PyTorch and PARL that works for regression tasks.

## Benchmark result

We follow the regression task setting from [Meta-SGD](https://arxiv.org/pdf/1707.09835.pdf), where the model is going to learn different sine waves. We train and test the model with 5-shot tasks. The figure below shows the test losses of MAML and MAML++ on 10000 randomly generated sine waves.

<p align="center">
<img src=".benchmark/loss.png" alt="result"/>
</p>

| MAML(from Meta-SGD) | Mate-SGD(from Meta-SGD) | MAML (ours) | MAML++ (ours)|
| --- | --- | --- | --- |
| 1.13&plusmn;0.18 |0.90&plusmn;0.16|  0.93&plusmn;0.02 | 0.34&plusmn;0.01 |

## How to use

### Dependencies:

+ python>=3.7.0
+ pytorch==1.7.1
+ parl

### Start Training:

~~~
python3 train.py
~~~
