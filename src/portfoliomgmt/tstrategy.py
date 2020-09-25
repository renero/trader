import enum


class TStrategy(enum.Enum):

    fifo = 'fifo'
    lifo = 'lifo'

    def __str__(self):
        return self.value
