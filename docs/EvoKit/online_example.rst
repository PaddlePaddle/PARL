Example for Online Products
################################

``本教程的目标: 演示通过EvoKit库上线后，如何迭代算法，更新模型参数。``

在产品线中，线上无法实时拿到用户日志，经常是通过保存用户点击/时长日志，在线下根据用户数据更新模型，然后再推送到线上，完成算法的更新。
本教程继续围绕经典的CartPole环境,展示如何通过在线采样/离线更新的方式，来更新迭代ES算法。

demo的完整代码示例放在demp/online_example文件夹中。
``TO DO: 文件夹``

初始化solver
---------------------
构造solver，对它初始化，并保存到文件。初始化solver仅需在开始时调用一次。

.. code-block:: c++

    std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>();
    agent->load_config(FLAGS_config_path);
    agent->load_inference_model(FLAGS_model_dir);
    agent->init_solver();
    agent->save_solver(FLAGS_model_dir);


线上采样
---------------------
加载模型和solver，记录线上采样返回的sampling_info以及评估的reward，并通过二进制的方式记录到log文件中。

.. code-block:: c++

    std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>();
    agent->load_config(FLAGS_config_path);
    agent->load_inference_model(FLAGS_model_dir);
    agent->load_solver(FLAGS_model_dir);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ITER; ++i) {
        std::shared_ptr<ESAgent> sampling_agent = sampling_agents[i];
        SamplingInfo sampling_info;
        sampling_agent->add_noise(sampling_info);
        float reward = evaluate(envs[i], sampling_agent);
        sampling_infos[i] = sampling_info;
        rewards[i] = reward;
    } 

    // save sampling information and log in binary fomrat
    std::ofstream log_stream(FLAGS_log_path, std::ios::binary);
    for (int i = 0; i < ITER; ++i) {
        std::string data;
        sampling_infos[i].SerializeToString(&data);
        int size = data.size();
        log_stream.write((char*) &rewards[i], sizeof(float));
        log_stream.write((char*) &size, sizeof(int));
        log_stream.write(data.c_str(), size);
    } 
    log_stream.close();


线下更新
-----------------------
在加载好之前记录的log之后，调用 ``update`` 函数进行更新，然后通过 ``save_inference_model`` 和 ``save_solver`` 函数保存更新后的参数到本地，推送到线上。

.. code-block:: c++

    std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>();
    agent->load_config(FLAGS_config_path);
    agent->load_inference_model(FLAGS_model_dir);
    agent->load_solver(FLAGS_model_dir);

    // load training data
    std::vector<SamplingInfo> sampling_infos;
    std::vector<float> rewards(ITER, 0.0f);
    sampling_infos.resize(ITER);
    std::ifstream log_stream(FLAGS_log_path);
    CHECK(log_stream.good()) << "[EvoKit] cannot open log: " << FLAGS_log_path;
    char buffer[1000];
    for (int i = 0; i < ITER; ++i) {
        int size;
        log_stream.read((char*) &rewards[i], sizeof(float));
        log_stream.read((char*) &size, sizeof(int));
        log_stream.read(buffer, size);
        buffer[size] = 0;
        std::string data(buffer);
        sampling_infos[i].ParseFromString(data);
    } 

    // update model and save parameter
    agent->update(sampling_infos, rewards);
    agent->save_inference_model(FLAGS_updated_model_dir);
    agent->save_solver(FLAGS_updated_model_dir);


主代码
-----------------------

将以上代码分别编译成可执行文件。

- 初始化solver: ``init_solver`` 。
- 线上采样: ``online_sampling`` 。
- 线下更新: ``offline update`` 。

.. code-block:: shell

    #------------------------init solver------------------------
    ./init_solver \
        --model_dir="./model_warehouse/model_dir_0" \
        --config_path="config.prototxt"


    for ((epoch=0;epoch<200;++epoch));do
    #------------------------online sampling------------------------
        ./online_sampling \
            --log_path="./sampling_log" \
            --model_dir="./model_warehouse/model_dir_$epoch" \
            --config_path="./config.prototxt"

    #------------------------offline update------------------------
        next_epoch=$((epoch+1))
        ./offline_update \
            --log_path='./sampling_log' \
            --model_dir="./model_warehouse/model_dir_$epoch" \
            --updated_model_dir="./model_warehouse/model_dir_${next_epoch}" \
            --config_path="./config.prototxt"
    done
