# import matplotlib.pyplot as plt
# import numpy as np
# from math import log10, pow
#
#
# class Chart:
#
#     @staticmethod
#     def chart(array,
#               metric_name='',
#               chart_type='line',
#               ma: bool = False):
#         """
#         Plots a chart of the array passed
#         :param array:
#         :param metric_name: label to be displayed on Y axis
#         :param chart_type: either 'line' or 'scatter'
#         :param ma: moving average?
#         :return:
#         """
#         if ma is True:
#             magnitude = int(log10(len(array))) - 1
#             period = int(pow(10, magnitude))
#             if period == 1:
#                 period = 10
#             data = np.convolve(array, np.ones((period,)) / period, mode='valid')
#         else:
#             data = array
#         if chart_type == 'scatter':
#             plt.scatter(range(len(data)), data)
#         else:
#             plt.plot(data)
#         plt.ylabel(metric_name)
#         plt.xlabel('Number of games')
#         plt.show()
