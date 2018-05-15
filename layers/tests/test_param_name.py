import unittest
import pprl.layers as layers


class TestParamName(unittest.TestCase):
    def test_name_number(self):
        self.fc1 = layers.fc(100)
        self.fc2 = layers.fc(100)
        self.embedding = layers.embedding(128)
        self.dynamic_grus = []
        for i in range(5):
            self.dynamic_grus.append(layers.dynamic_gru(50))
        self.assertEqual(self.fc1.param_name, "fc_0.w")
        self.assertEqual(self.fc2.param_name, "fc_1.w")
        self.assertEqual(self.embedding.param_name, "embedding_0.w")
        self.assertEqual(self.embedding.bias_name, None)
        for i, gru in enumerate(self.dynamic_grus):
            self.assertEqual(gru.param_name, "dynamic_gru_%d.w" % i)


if __name__ == '__main__':
    unittest.main()
