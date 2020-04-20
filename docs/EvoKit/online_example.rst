Example for Online Products
#########################

``本教程的目标: 演示通过EvoKit库上线后，如何迭代算法，更新模型参数。``

在实际的产品线中，线上无法实时拿到用户日志，经常是通过保存用户点击/时长日志，在线下根据用户数据更新模型，然后再推送到线上，完成算法的更新。
本教程继续围绕经典的CartPole环境,展示如何通过在线采样/离线更新的方式，来更新迭代ES算法。

demo的完整代码示例放在demp/online_example文件夹中。

线上采样
---------------------

这部分的逻辑与上一个demo极度相似，主要的区别是采样返回的key以及评估的reward通过二进制的方式记录到log文件中。

.. code-block:: c++

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ITER; ++i) {
        std::shared_ptr<ESAgent> sampling_agent = sampling_agents[i];
        SamplingInfo key;
        bool success = sampling_agent->add_noise(key);
        float reward = evaluate(envs[i], sampling_agent);
        noisy_keys[i] = key;
        noisy_rewards[i] = reward;
    } 

    // save sampling information and log in binary fomrat
    std::ofstream log_stream(FLAGS_log_path, std::ios::binary);
    for (int i = 0; i < ITER; ++i) {
      std::string data;
      noisy_keys[i].SerializeToString(&data);
      int size = data.size();
      log_stream.write((char*) &noisy_rewards[i], sizeof(float));
      log_stream.write((char*) &size, sizeof(int));
      log_stream.write(data.c_str(), size);
    } 
    log_stream.close();



线下更新
-----------------------

在加载好之前记录的log之后，调用 ``update`` 函数进行更新，然后通过 ``save_inference_model`` 函数保存更新后的参数到本地，推送到线上。

.. code-block:: c++

    // load training data
    std::vector<SamplingInfo> noisy_keys;
    std::vector<float> noisy_rewards(ITER, 0.0f);
    noisy_keys.resize(ITER);
    std::ifstream log_stream(FLAGS_log_path);
    CHECK(log_stream.good()) << "[EvoKit] cannot open log: " << FLAGS_log_path;
    char buffer[1000];
    for (int i = 0; i < ITER; ++i) {
        int size;
        log_stream.read((char*) &noisy_rewards[i], sizeof(float));
        log_stream.read((char*) &size, sizeof(int));
        log_stream.read(buffer, size);
        buffer[size] = 0;
        std::string data(buffer);
        noisy_keys[i].ParseFromString(data);
    } 

    // update model and save parameter
    agent->update(noisy_keys, noisy_rewards);
    agent->save_inference_model(FLAGS_updated_model_dir);
