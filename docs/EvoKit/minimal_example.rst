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
	        dirname='init_model',
	        feeded_var_names=['obs'],
	        target_vars=[prob],
	        params_filename='params',
	        model_filename='model',
	        executor=exe)

step2: 构造ESAgent
###################

- 调用 ``load_config`` 加载配置文件。
- 调用 ``load_inference_model`` 函数加载模型参数。
- 调用 ``init_solver`` 初始化solver。

配置文件主要是用于指定进化算法类型（比如Gaussian或者CMA）,使用的optimizer类型（Adam或者SGD）。

.. code-block:: c++

    ESAgent agent = ESAgent();
    agent.load_config(config);
    agent.load_inference_model(model_dir);
    agent.init_solver();

    // 附：EvoKit配置项示范
    solver {
        type: BASIC_ES
        optimizer { // 线下Adam更新
            type: ADAM
            base_lr: 0.05
            adam {
                beta1: 0.9
                beta2: 0.999
                epsilon: 1e-08
            }
        }
        sampling { // 线上高斯采样
            type: GAUSSIAN_SAMPLING
            gaussian_sampling {
                std: 0.5
                cached: true
                seed: 1024
                cache_size : 100000
            }
        }
    }


step3: 生成用于采样的Agent
#################################

主要关注三个接口：

- 调用 ``clone`` 生成一个用于sampling的agent。
- 调用 ``add_noise`` 给这个agent的参数空间增加噪声，同时返回该噪声对应的唯一信息，这个信息得记录在log中，用于线下更新。
- 调用 ``predict`` 提供预测接口。

.. code-block:: c++

    auto sampling_agent = agent.clone();
    auto sampling_info = sampling_agent.add_noise();
    sampling_agent.predict(feature);

step4: 用采样的数据更新模型参数
#################################

用户提供两组数据：

- 采样参数过程中用于线下复现采样噪声的sampling_info
- 扰动参数后，新参数的评估结果

.. code-block:: c++

    agent.update(sampling_infos, rewards);

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
        std::string model_dir = "./demo/cartpole/init_model";
        std::string config_path = "./demo/cartpole/config.prototxt";
        std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>();
        agent->load_config(config_path); // 加载配置

        agent->load_inference_model(FLAGS_model_dir); // 加载初始预测模型
        agent->init_solver(); // 初始化solver，注意要在load_inference_model后执行
    
        // 生成10个agent用于同时采样
        std::vector<std::shared_ptr<ESAgent>> sampling_agents;
        for (int i = 0; i < ITER; ++i) {
            sampling_agents.push_back(agent->clone());
        }
    
        std::vector<SamplingInfo> sampling_infos;
        std::vector<float> rewards(ITER, 0.0f);
        sampling_infos.resize(ITER);
        omp_set_num_threads(10);
    
        // 共迭代100轮
        for (int epoch = 0; epoch < 100; ++epoch) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < ITER; ++i) {
                std::shared_ptr<ESAgent> sampling_agent = sampling_agents[i];
                SamplingInfo sampling_info;
                sampling_agent->add_noise(sampling_info);
                float reward = evaluate(envs[i], sampling_agent);
                // 保存采样的sampling_info以及对应的评估结果reward
                sampling_infos[i] = sampling_info;
                rewards[i] = reward;
            }
            // 更新模型参数，注意：参数更新后会自动同步到sampling_agent中
            agent->update(sampling_infos, rewards);
    
            int reward = evaluate(envs[0], agent);
            LOG(INFO) << "Epoch:" << epoch << " Reward: " << reward; // 打印每一轮reward
        }
    }

如何运行demo
#################

- 下载代码

  在icode上clone代码，我们的仓库路径是： ``baidu/nlp/deep-es`` ``TO DO: 修改库路径``

- 编译demo

  通过bcloud的云端集群编译即可，命令为： ``bb``

- 运行demo

  编译完成后，我们需要增加动态库查找路径：

  ``export LD_LIBRARY_PATH=./output/so/:$LD_LIBRARY_PATH``

  运行demo： ``./output/bin/cartpole/train``

问题解决
####################

在使用过程中有任何问题，请加hi群: 1692822 (PARL官方答疑群)进行咨询，开发同学会直接回答任何的使用问题。
