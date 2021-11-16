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

场景3: 在训练完成后，需要把训练好的模型结构和参数保存到本地，用于后续的推理部署。

直接通过agent的相关接口来完成网络结构和参数的存储。

```python
# 保存网络结构和参数到./inference_model_dir
agent.save_inference_model('./inference_model_dir', [[None, 128]], ['float32'])
```

对于Actor-Critic类算法，只需要保存其中的Actor网络。

```python
# 保存Actor-Critic算法的策略网络结构和参数到./inference_ac_model_dir
agent.save_inference_model('./inference_ac_model_dir', [[None, 128]], ['float32'], agent.alg.model.actor_model)
```
