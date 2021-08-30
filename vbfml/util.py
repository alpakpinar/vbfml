from dataclasses import dataclass


@dataclass
class LRIDictBuffer(dict):
    """
    Least-recently-inserted buffered dictionary.

    A dictionary with fixed maximum size. If the maximum
    size is reached and another insertion is made,
    the oldest item is removed.

    Implementation relies on dict insertion ordering,
    which is guaranteed since python 3.7.
    """

    buffer_size: int = 10

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if len(self) > self.buffer_size:
            self.forget_oldest()

    def forget_oldest(self):
        self.pop(next(iter(self)))
