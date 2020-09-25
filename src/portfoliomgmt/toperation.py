import enum


class TOperation(enum.Enum):

    bear = 'bear'
    bull = 'bull'

    def __str__(self):
        return self.value
