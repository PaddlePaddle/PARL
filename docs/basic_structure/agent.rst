Agent (*Generate Data Flow*)
===============================

Methods
--------
1. __init__(self, algorithm, gpu_id=None)

    Call build_program here and run initialization for default_startup_program.

2. build_program(self)

    Use define_predict and define_learn in Algorithm to build training program and prediction program. This will be called
    by __init__ method in class Agent.

3. predict(self, obs)

    Predict the action with current observation of the enviroment. Note that this function will only do the prediction and it doesn't try any exploration.
    To explore in the action space, you should create your process in `sample` function below.
    Basically, this function is often used in test process.

4. sample(self, obs)

    Predict the action given current observation of the enviroment. 
    Additionaly, action will be added noise here to explore a new trajectory. 
    Basically, this function is often used in training process.

5. learn(self, obs, action, reward, next_obs, terminal)

    Pass data to the training program to update model. This method is the training interface for Agent.
