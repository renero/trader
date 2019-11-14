from pandas import DataFrame, Series


# TODO Hacer esta clase generica para poder manejar varios dataframes

class Memory:

    def __init__(self, configuration):
        self.params = configuration
        # Create a DataFrame within the configuration to store all the values
        # that are relevant to later perform data analysis.
        # The YAML file contains the column names in a parameter called
        # table_headers.
        self.results = DataFrame(columns=self.params.table_headers)

    def record_action(self, action_name):
        """
        Record action selected in results table.
        :param action_name:
        :return: None
        """
        last_index = self.results.shape[0] - 1
        self.results.loc[last_index, 'action'] = action_name

    def record_values(self, portfolio, t: int):
        """
        Records the values from portfolio in a new memory position
        :param portfolio:
        :param t: instant in time during simulation
        :return: None
        """
        values = [t] + portfolio.values_to_record()
        row = Series(dict(zip(self.params.table_headers, values)))
        self.results = self.results.append(row, ignore_index=True)

    def record_reward(self, reward, current_state):
        """
        Append reward and current state into the last line recorded.
        :param reward:
        :param current_state:
        :return: None
        """
        last_index = self.results.shape[0] - 1
        self.results.loc[last_index, 'reward'] = reward
        self.results.loc[last_index, 'state'] = current_state

    def reset(self):
        if self.results.shape[0] > 0:
            self.results = self.results[0:0]
