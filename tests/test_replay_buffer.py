import numpy as np
import unittest
from error_handling import LastElementError
from replay_buffer import Experience, Sample, ReplayBuffer

class ExperienceTest(Experience):
    def __init__(self, obs, reward, action, state, done):
        super(ExperienceTest, self).__init__(obs, reward, action, done)
        self.state = state

class TestReplayBuffer(unittest.TestCase):
    def test_single_instance_replay_buffer(self):
        capacity = 30
        episode_len = 4
        buf = ReplayBuffer(capacity, ExperienceTest)
        total = 0
        expect_total = 0
        for i in xrange(10*capacity):
            e = ExperienceTest(np.zeros(10), i, i, np.ones(20), (i+1) % episode_len == 0)
            buf.add(e)
            # check the circular queue in the buffer
            self.assertTrue(len(buf) == min(i+1, capacity))
            if (len(buf) < 2): # need at least two elements
                continue
            # should raise error when trying to pick up the last element
            with self.assertRaises(LastElementError):
                t = Sample(i % capacity, 1)
                buf.get_experiences(t)
            expect_total += len(buf)
            # neither last element nor episode end should be picked up
            for s in buf.sample(len(buf)):
                try:
                    exps = buf.get_experiences(s)
                    total += 1
                except LastElementError as err:
                    self.fail('test_single_instance_replay_buffer raised '
                              'LastElementError: ' +  err.message)
        # check the total number of elements added into the buffer
        self.assertTrue(total == expect_total)

if __name__ == '__main__':
    unittest.main(exit = False)
