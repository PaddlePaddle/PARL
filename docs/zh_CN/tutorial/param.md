# **教程：模型参数管理**
场景1: 在训练过程中，我们有时候需要把训练好的模型参数保存到本地，用于后续的部署或者评估。

当用户构建好agent之后，可以直接通过agent的相关接口来完成参数的存储。
```python
agent = AtariAgent()
# 保存参数到 ./model_dir
agent.save('./model_dir')
# 恢复参数到这个agent上
agent.restore('./model_dir')
```

场景2: 并行训练过程中，经常需要把最新的模型参数同步到另一台服务器上，这时候，需要把模型参数拿到内存中，然后再赋值给另一台机器上的agent（actor)。

```python
#--------------Agent---------------
weights = agent.get_weights()
#--------------Remote Actor--------------
actor.set_weights(weights)
```
