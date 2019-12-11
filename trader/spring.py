class Spring:
    """
    This class implements a kind of spring that stretches whenever new values
    are higher than its initial position, but alerts whenever those maximum
    levels decrease beyond a given threshold.
    Used to control that from a given point, values always grow and never
    shrink.
    """

    def __init__(self, configuration, starting_point: float):
        self.params = configuration
        self.log = self.params.log

        self.has_position = False
        self.starting_point = starting_point
        self.max_value = self.starting_point
        self.max_shrink = self.params.stop_drop_rate

    def anchor(self, value):
        self.has_position = True
        self.starting_point = value
        self.max_value = self.starting_point
        self.log.debug('BUY at {}'.format(value))

    def release(self):
        self.has_position = False
        self.log.debug('Sell')

    def better(self, x: float, y: float) -> bool:
        """
        Depending on whether we're in BEAR or BULL mode, determines if X is
        better than Y. In BEAR mode, X < Y is better, and in BULL mode
        X >= Y is better.
        """
        if self.params.mode == 'bear':
            return x < y
        else:
            return x >= y

    def breaks(self, new_value: float) -> bool:
        if self.has_position is False:
            return False
        if self.better(new_value, self.max_value):
            self.max_value = new_value
            self.log.debug('Updated abs.max: {:.2f}'.format(self.max_value))
            return False
        else:
            ratio = abs(self.max_value - new_value) / self.max_value
            if ratio > self.max_shrink:
                self.log.debug(
                    'Breaks!! as max({}) - current({}) ratio is {:.2f}'.format(
                        self.max_value, new_value, ratio))
                self.max_value = new_value
                return True
            else:
                return False

    def check(self, action, price):
        """
        Check if the new price drops significantly, and update positions.
        :param action: the action decided by the Deep Q-Net
        :param price: the current price.
        :return: the action, possibly modified after check
        """
        if self.breaks(price):
            self.log.debug('! STOP DROP overrides action to SELL')
            self.log.warn('! STOP DROP overrides action to SELL')
            action = self.params.action.index('sell')

        if action == self.params.action.index('buy'):
            self.anchor(price)
        elif action == self.params.action.index('sell'):
            self.release()

        return action
