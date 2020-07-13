import pandas as pd
import numpy as np


class CSPredict(object):

    def __init__(self, params, X_test, y_test=None, oh_encoder=None):
        super(CSPredict, self).__init__()
        self.params = params
        self.log = params.log

        self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test
        if oh_encoder is not None:
            self.oh_encoder = oh_encoder
        self.num_testcases = self.X_test.shape[0]
        self.pred_length = len(self.oh_encoder.states)
        self.result = pd.DataFrame([], columns=[
            'act_body_sign', 'pred_body_sign',
            'act_body_val', 'pred_body_val',
            'act_open_sign', 'pred_open_sign',
            'act_open_val', 'pred_open_val',
            'act_high_sign', 'pred_high_sign',
            'act_high_val', 'pred_high_val',
            'act_low_sign', 'pred_low_sign',
            'act_low_val', 'pred_low_val',
            'act_close_sign', 'pred_close_sign',
            'act_close_val', 'pred_close_val'])
        self.result_cols = np.array(list(self.result))
        self.body_cols = self.result_cols[0:4]
        self.move_cols = self.result_cols[4:self.result_cols.shape[0]]

    def body_row(self, cse_actual, cse_predicted):
        return [cse_actual[0], cse_predicted[0],
                cse_actual[-1], cse_predicted[-1]]

    def predict_body_batch(self, nn):
        positive_all = 0
        positive_sign = 0
        positive_shape = 0

        for j in range((self.X_test.shape[0]) - 2):
            y = nn.predict(self.X_test[j:j + 1, :, :])
            y_pred = nn.hardmax(y[0])
            cse_predicted = self.oh_encoder.decode(y_pred)[0]
            cse_actual = self.oh_encoder.decode(
                self.y_test[j:j + 1, :])[0]
            positive_all += int(cse_actual == cse_predicted)
            positive_sign += int(cse_actual[0] == cse_predicted[0])
            positive_shape += int(cse_actual[-1] == cse_predicted[-1])
            self.log.debug(
                'predicted: {} / actual: {}'.format(cse_predicted, cse_actual))

            results = self.body_row(cse_actual, cse_predicted)
            self.result.loc[j, self.body_cols] = results

        # Remove columns from the MOVE part
        self.result.drop(self.move_cols, axis=1, inplace=True)

        self.log.info(
            'Pos.Rate (all/sign/body): {:.3f} / {:.3f} / {:.3f}'.format(
                (positive_all / self.num_testcases),
                (positive_sign / self.num_testcases),
                (positive_shape / self.num_testcases)))

        return ((positive_all / self.num_testcases),
                (positive_sign / self.num_testcases),
                (positive_shape / self.num_testcases))

    def move_row(self, move_actual, move_predicted):
        return [
            move_actual[0][0], move_predicted[0][0],
            move_actual[0][1], move_predicted[0][1],
            move_actual[1][0], move_predicted[1][0],
            move_actual[1][1], move_predicted[1][1],
            move_actual[2][0], move_predicted[2][0],
            move_actual[2][1], move_predicted[2][1],
            move_actual[3][0], move_predicted[3][0],
            move_actual[3][1], move_predicted[3][1]]

    def predict_move_batch(self, nn):
        pos_open = 0
        pos_close = 0
        pos_high = 0
        pos_low = 0

        num_predictions = int(self.y_test.shape[1] / self.pred_length)

        for j in range((self.X_test.shape[0]) - 2):
            y = nn.predict(self.X_test[j:j + 1, :, :])
            #
            # TODO: Check if:
            #   y[0][i * pred_length:(i * pred_length) + pred_length - 1])
            #
            y_pred = [
                nn.hardmax(
                    y[0][i * self.pred_length:( \
                        i * self.pred_length) + self.pred_length])
                for i in range(num_predictions)
            ]
            move_predicted = [
                self.oh_encoder.decode(y_pred[i])[0]
                for i in range(num_predictions)
            ]
            move_actual = [
                self.oh_encoder.decode(
                    self.y_test[i:i + 1, :])[0]
                for i in range(num_predictions)
            ]
            pos_open += int(move_actual[0] == move_predicted[0])
            pos_high += int(move_actual[1] == move_predicted[1])
            pos_low += int(move_actual[2] == move_predicted[2])
            pos_close += int(move_actual[3] == move_predicted[3])

            results = self.move_row(move_actual, move_predicted)
            self.result.loc[j, self.move_cols] = results

        # Remove columns from the MOVE part
        self.result.drop(self.body_cols, axis=1, inplace=True)

        self.log.info(
            'Pos.Rate (O/H/L/C): {:.4f} : {:.4f} : {:.4f} : {:.4f} ~ {:.4f}'.
                format((pos_open / self.num_testcases),
                       (pos_high / self.num_testcases),
                       (pos_low / self.num_testcases),
                       (pos_close / self.num_testcases),
                       ((pos_open + pos_high + pos_low + pos_close) /
                        (self.num_testcases * 4))))
