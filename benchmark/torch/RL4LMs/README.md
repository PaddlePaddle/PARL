## Reproduce Summarization-RLHF in RL4LMs using PARL

> Paper: [Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241)

### Background

- Summarization task in NLP: Summarization is the task of producing a shorter version 
  of one document that preserves most of the input's meaning. 
- RLHF: The abbreviation of Reinforcement Learning with Human Feedback, which uses human knowledge to train RL algorithms.
 More information is available in the Hugging Face blog [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)

### Main contribution

- Build new Summarization-RLHF framework using PARL
- Use PARL parallel training

### How to use

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Start training
```bash
# start xparl
xparl start --port 8811 --cpu_num 10

# start training
python train.py
```

### Code Reference

- Official code: [RL4LMs](https://github.com/allenai/RL4LMs)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
