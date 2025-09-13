from rl_games.torch_runner import Runner


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    return runner