import matplotlib.pyplot as plt


class Chart:

    @staticmethod
    def reinforcement(r_avg_list, plot: bool = False):
        if plot is False:
            return
        plt.plot(r_avg_list)
        plt.ylabel('Average reward per game')
        plt.xlabel('Number of games')
        plt.show()
