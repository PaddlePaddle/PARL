minimal example
---------------------

``本教程的目标:
演示如何通过EvoKit库来解决经典的CartPole 问题。``

*本教程假定读者曾经使用过PaddlePaddle, 了解基本的进化算法迭代流程。*

CartPole 介绍
#############
CartPole又叫倒立摆。小车上放了一根杆，杆会因重力而倒下。为了不让杆倒下，我们要通过移动小车，来保持其是直立的。如下图所示。
在每一个时间步，模型的输入是一个4维的向量,表示当前小车和杆的状态，模型输出的信号用于控制小车往左或者右移动。当杆没有倒下的时候，每个时间步，环境会给1分的奖励；当杆倒下后，环境不会给任何的奖励，游戏结束。

.. image:: ../../examples/QuickStart/performance.gif
  :width: 300px

step1: 生成预测网络
########################
根据上面的环境介绍，我们需要构造一个神经网络，输入为4维的向量，输出为2维的概率分布向量（表示左/右）移动的概率。
在这里，我们使用Paddle来实现预测网络，并保存到本地。

.. code-block:: python

	from paddle import fluid
	
	def net(obs, act_dim):
	    hid1 = fluid.layers.fc(obs, size=20)
	    prob = fluid.layers.fc(hid1, size=act_dim, act='softmax')
	    return prob
	
	if __name__ == '__main__':
	    obs_dim = 4
	    act_dim = 2
	    obs = fluid.layers.data(name="obs", shape=[obs_dim], dtype='float32')
	    prob = net(obs, act_dim)
	
	    exe = fluid.Executor(fluid.CPUPlace())
	    exe.run(fluid.default_startup_program())
	    fluid.io.save_inference_model(
	        dirname='cartpole_init_model',
	        feeded_var_names=['obs'],
	        target_vars=[prob],
	        params_filename='params',
	        model_filename='model',
	        executor=exe)

step2: 构造ESAgent
###################
- 根据配置文件构造一个ESAgent
- 调用 ``load_inference_model`` 函数加载模型参数

配置文件主要是用于指定进化算法类型（比如Gaussian或者CMA）,使用的optimizer类型（Adam或者SGD）。

.. code-block:: c++

	ESAgent agent = ESAgent(config_path);
  agent->load_inference_model(model_dir);

	//附：DeepES配置项示范
	seed: 1024 //随机种子，用于复现
	gaussian_sampling { //高斯采样相关参数
	  std: 0.5
	}
	optimizer { //离线更新所用的optimizer 类型以及相关超级参数
	  type: "Adam"
	  base_lr: 0.05
	}

step3: 生成用于采样的Agent
###################

主要关注三个接口：

- clone(): 生成一个用于sampling的agent。
- add_noise()：给这个agent的参数空间增加噪声，同时返回该噪声对应的唯一信息，这个信息得记录在log中，用于线下更新。
- predict()：提供预测接口。

.. code-block:: c++

	auto sampling_agent = agent.clone();
	auto sampling_info = sampling_agent.add_noise();
	sampling_agent.predict(feature);

step4: 用采样的数据更新模型参数
###################

用户提供两组数据：
- 采样参数过程中用于线下复现采样噪声的key
- 扰动参数后，新参数的评估结果

.. code-block:: c++

	agent.update(info, rewards);

主代码以及注释
#################

以下的代码演示通过多线程同时采样, 提升解决问题的效率。

.. code-block:: c++

	int main(int argc, char* argv[]) {
	    std::vector<CartPole> envs;
      // 构造10个环境，用于多线程训练
	    for (int i = 0; i < ITER; ++i) {
	        envs.push_back(CartPole());
	    }
	
      // 初始化ESAgent
	    std::string model_dir = "./demo/cartpole_init_model";
	    std::string config_path = "./demo/cartpole_config.prototxt";
	    std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>(model_dir, config_path);
	
      // 生成10个agent用于同时采样
	    std::vector< std::shared_ptr<ESAgent> > sampling_agents;
	    for (int i = 0; i < ITER; ++i) {
	        sampling_agents.push_back(agent->clone());
	    }
	
	    std::vector<SamplingInfo> noisy_keys;
	    std::vector<float> noisy_rewards(ITER, 0.0f);
	    noisy_keys.resize(ITER);
	    omp_set_num_threads(10);
	
      // 共迭代100轮
	    for (int epoch = 0; epoch < 100; ++epoch) {
	        #pragma omp parallel for schedule(dynamic, 1)
	        for (int i = 0; i < ITER; ++i) {
	            std::shared_ptr<ESAgent> sampling_agent = sampling_agents[i];
	            SamplingInfo key;
	            bool success = sampling_agent->add_noise(key);
	            float reward = evaluate(envs[i], sampling_agent);
              // 保存采样的key以及对应的评估结果
	            noisy_keys[i] = key;
	            noisy_rewards[i] = reward;
	        }
          // 更新模型参数，注意：参数更新后会自动同步到sampling_agent中
	        bool success = agent->update(noisy_keys, noisy_rewards);
	
	        int reward = evaluate(envs[0], agent);
	        LOG(INFO) << "Epoch:" << epoch << " Reward: " << reward;
	    }
	}

如何运行demo
#################

- 下载代码

  在icode上clone代码，我们的仓库路径是： ``baidu/nlp/deep-es``

- 编译demo

  通过bcloud的云端集群编译即可，命令为： ``bb``

- 运行demo

  编译完成后，我们需要增加动态库查找路径：

  ``export LD_LIBRARY_PATH=./output/so/:$LD_LIBRARY_PATH``

  运行demo： ``./output/bin/evokit_demo``

问题解决
####################

在使用过程中有任何问题，请加hi群: 1692822 (PARL官方答疑群)进行咨询，开发同学会直接回答任何的使用问题。
