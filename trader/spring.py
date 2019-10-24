from logger import Logger


class Spring:
    def __init__(self, starting_point: float, log: Logger,
                 max_shrink: float = 0.02):
        self.log = log
        self.has_position = False
        self.starting_point = starting_point
        self.max_value = self.starting_point
        self.max_shrink = max_shrink

    def anchor(self, value):
        self.has_position = True
        self.starting_point = value
        self.max_value = self.starting_point
        self.log.debug('BUY at {}'.format(value))

    def release(self):
        self.has_position = False
        self.log.debug('Sell')

    def stretch(self, new_value: float) -> float:
        if self.has_position is False:
            return 0.0
        if new_value >= self.max_value:
            self.max_value = new_value
        return self.max_value

    def breaks(self, new_value: float) -> bool:
        if self.has_position is False:
            return False
        if new_value >= self.max_value:
            self.max_value = new_value
            self.log.debug('Updated max: {:.2f}'.format(self.max_value))
            return False
        else:
            ratio = (self.max_value - new_value) / self.max_value
            if ratio > self.max_shrink:
                self.log.debug(
                    'Breaks!! as max({}) - current({}) ratio is {:.2f}'.format(
                        self.max_value, new_value, ratio))
                self.max_value = new_value
                return True
            else:
                return False
