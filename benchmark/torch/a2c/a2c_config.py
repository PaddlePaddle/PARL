config = {

    #==========  remote config ==========
    'master_address': 'localhost:8010',

    #==========  env config ==========
    'env_name': 'BreakoutNoFrameskip-v4',
    'env_dim': 84,

    #==========  actor config ==========
    'actor_num': 5,
    'env_num': 5,
    'sample_batch_steps': 20,

    #==========  learner config ==========
    'max_sample_steps': int(1e7),
    'gamma': 0.99,
    'lambda': 1.0,

    # start learning rate
    'start_lr': 0.001,
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'get_remote_metrics_interval': 10,
    'log_metrics_interval_s': 10,
    'entropy_coeff': -0.05,
    'learning_rate': 3e-4
}
