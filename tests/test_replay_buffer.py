import numpy as np
import unittest
from replay_buffer import Experience, Sample, ReplayBuffer

class ExperienceTest(Experience):
    def __init__(self, obs, reward, action, state, done):
        super(ExperienceTest, self).__init__(obs, reward, action, done)
        self.state = state

class TestReplayBuffer(unittest.TestCase):
    def test_single_instance_replay_buffer(self):
        capacity = 30
        buf = ReplayBuffer(capacity, ExperienceTest)
        total = 0
        expect_total = 0
        for i in xrange(10*capacity):
            e = ExperienceTest(np.zeros(10), i, i, np.ones(20), (i+1) % 4 == 0)
            buf.add(e)
            self.assertTrue(len(buf) == min(i+1, capacity))
            if (len(buf) < 2): # need at least two elements
                continue
            expect_total += len(buf)
            for s in buf.sample(len(buf)):
                total += 1
                self.assertTrue(buf.has_next(s.i))
                exps = buf.get_experiences(s)
                for e in exps:
                    self.assertTrue(not e.done)
        self.assertTrue(total == expect_total)

if __name__ == '__main__':
    unittest.main()
