from unittest import TestCase
from vbfml.util import LRIDictBuffer

class TestLRIDictBuffer(TestCase):
    def setUp(self):
        self.buffer_size = 5
        self.buffer = LRIDictBuffer(buffer_size=self.buffer_size)
        self.test_items = {
            "a": "b",
            1: "c",
            3: 4.5,
            7: 8,
            121231231: 129381023,
            "sdajsld": 0,
            -1: -0.5,
        }

    def test_buffer(self):
        # Buffer initalizes empty
        self.assertEqual(len(self.buffer), 0)

        forgotten_keys = []
        for i, (key, value) in enumerate(self.test_items.items()):

            # Test insertion and read back
            self.buffer[key] = value
            expected_length = min(i + 1, self.buffer_size)
            self.assertEqual(len(self.buffer), expected_length)
            self.assertEqual(self.buffer[key], value)

            # Store keys that will have been removed after loop
            if i < len(self.test_items) - self.buffer_size:
                forgotten_keys.append(key)

        # Test that early keys have really been removed
        for key in forgotten_keys:
            self.assertFalse(key in self.buffer)
