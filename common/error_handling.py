class LastElementError(Exception):
    """
    Raised when the last element or an episode-end element is sampled.

    Attributes:
        idx(int): the index of the element being picked
        is_episode_end(bool): whether the element is an episode end
    """
    def __init__(self, idx, is_episode_end):
        self._idx = idx
        self._episode_end = is_episode_end
        self.message = 'The element at ' + str(idx) + ' is '
        if is_episode_end:
            self.message += 'an episode end.'
        else:
            self.message += 'the last element.'


