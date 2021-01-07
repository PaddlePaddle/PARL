# **教程：表格输出实验数据**
PARL提供了将训练过程中的指标输出到CSV表格的工具。工具导入方法：

`from parl.utils import CSVLogger`


### 使用教程
1. 传入CSV文件保存路径，并初始化CSVLogger

`csv_logger = CSVLogger("result.csv")`

2. 输出以字典形式记录的指标

参数
- result (dict) – 需要输出到CSV文件的指标字典

方法

`csv_logger.log_dict({"loss": 1, "reward": 2})`

### 完整例子
```python
from parl.utils import CSVLogger

csv_logger = CSVLogger("result.csv")
csv_logger.log_dict({"loss": 1, "reward": 2})
csv_logger.log_dict({"loss": 3, "reward": 4})

```
#### 预期结果

result.csv文件内容如下：

```
loss,reward
1,2
3,4
```
