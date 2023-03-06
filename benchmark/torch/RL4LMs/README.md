## Reproduce (Reconfiguration) Summarization in RL4LMs using PARL

> Paper: [Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241)
> 
> Official code: [RL4LMs](https://github.com/allenai/RL4LMs)
> 
> Other code referenced: [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)


### Main contribution

- Change from **\{ trainer: \{ ppo: \{ env, rollout_buffer, policy/model \} \} \}** to 
  **\{trainer: \{env, rollout_buffer, agent: \{ ppo: \{ model \} \} \} \}** according PARL architecture.

### Running command

```bash
python train.py --config_path configs/summarization/t5_ppo.yml
```