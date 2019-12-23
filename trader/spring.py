class spring:
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
        self.log.debug('Spring anchored at {:.2f}, stretched to {:.2f}'.format(
            value, self.max_value))

    def release(self):
        self.has_position = False
        self.log.debug('Spring released...')

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
            self.log.debug('Stretched abs.max: {:.2f}'.format(self.max_value))
            return False
        else:
            ratio = self.compute_ratio(new_value)
            if ratio > self.max_shrink:
                msg = 'Breaks! max({:.2f}) - current({:.2f}) ratio is {:.4f}'
                self.log.debug(msg.format(self.max_value, new_value, ratio))
                self.max_value = new_value
                return True
            else:
                return False

    def compute_ratio(self, new_value):
        # Avoid dividing by zero
        if self.max_value == 0.:
            max_val = 0.000001
        else:
            max_val = self.max_value
        ratio = abs(max_val - new_value) / max_val
        return ratio

    def check(self, action, price, is_failed_action):
        """
        Check if the new price drops significantly, and update positions.

        :param action:     The action decided by the Deep Q-Net
        :param price:      The current price.
        :is_failed_action: Boolean indicating if the action proposed by
                           the RL is feasible or not. If not, the action is
                           considered a failed action. Examples are attempts
                           to purchase shares without budget, or selling
                           non existing actions.

        :return:           The action, possibly modified after check
        """
        if self.breaks(price):
            self.log.debug('! STOP DROP overrides action to SELL')
            action = self.params.action.index('sell')

        # Check if action is failed
        if is_failed_action is False:
            if action == self.params.action.index('buy'):
                self.anchor(price)
            elif action == self.params.action.index('sell'):
                self.release()

        return action

    def correction(self, action, environment):
        """
        Check if we have to force operation due to stop drop
        """
        if self.params.stop_drop is True:
            # is this a failed action?
            is_failed_action = environment.portfolio.failed_action(
                action, environment.price_)
            action = self.check(action, environment.price_, is_failed_action)
        return action
