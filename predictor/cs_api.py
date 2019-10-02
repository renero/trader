import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame

from logger import Logger

log = Logger(3)


def predict_close(ticks, encoder, nn, params):
    """
    From a list of ticks, make a prediction of what will be the next CS.

    :param ticks: a dataframe of ticks with the expected headers and size
        corresponding to the window size of the network to be used.
    :param encoder: the encoder used to train the network
    :param nn: the recurrent network to make the prediction with
    :param params: the parameters file read from configuration.

    :return: the close value of the CS predicted.
    """
    # Check that the input group of ticks match the size of the window of
    # the network that is going to make the predict. That parameter is in
    # the window_size attribute within the 'encoder'.
    if ticks.shape[0] != encoder.window_size():
        info_msg = 'Tickgroup resizing: {} -> {}'
        log.info(info_msg.format(ticks.shape[0], encoder.window_size()))
        ticks = ticks.iloc[-encoder.window_size():, :]
        ticks.reset_index()

    # encode the tick in CSE and OH. Reshape it to the expected LSTM format.
    cs_tick = encoder.ticks2cse(ticks)
    cs_tick_body_oh = encoder.onehot['body'].encode(encoder.body(cs_tick))
    cs_tick_move_oh = encoder.onehot['move'].encode(encoder.move(cs_tick))

    input_body = cs_tick_body_oh.values[np.newaxis, :, :]
    input_move = cs_tick_move_oh.values[np.newaxis, :, :]

    # get a prediction from the proper networks, for the body part
    raw_prediction = nn['body'].predict(input_body)[0]
    pred_body_oh = nn['body'].hardmax(raw_prediction)
    pred_body_cs = encoder.onehot['body'].decode(pred_body_oh)

    # Repeat everything with the move:
    # get a prediction from the proper network, for the MOVE part
    pred_length = len(encoder.onehot['move']._states)
    num_predictions = int(input_move.shape[2] / pred_length)
    y = nn['move'].predict(input_move)[0]
    Y_pred = [
        nn['move'].hardmax(y[i * pred_length:(i * pred_length) + pred_length])
        for i in range(num_predictions)
    ]
    pred_move_cs = [
        encoder.onehot['move'].decode(Y_pred[i])[0] for i in
        range(num_predictions)
    ]

    # Decode the prediction into a normal tick (I'm here!!!)
    prediction_df = pd.DataFrame([], columns=params._cse_tags)
    prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
    this_prediction = dict(zip(params._cse_tags, prediction_cs))
    prediction_df = prediction_df.append(this_prediction, ignore_index=True)
    log.info('Net {} ID {} -> {}:{}|{}|{}|{}'.format(
        nn['body'].name,
        hex(id(nn)),
        prediction_df[params._cse_tags[0]].values[0],
        prediction_df[params._cse_tags[1]].values[0],
        prediction_df[params._cse_tags[2]].values[0],
        prediction_df[params._cse_tags[3]].values[0],
        prediction_df[params._cse_tags[4]].values[0],
    ))

    # Convert the prediction to a real tick
    pred = encoder.cse2ticks(prediction_df, cs_tick[-1])
    return pred['c'].values[-1]


def single_prediction(data: DataFrame, w_pos: int, nn, encoder, params):
    """
    Make a single prediction over a list of ticks. It uses all the
    networks loaded to produce all their predictions and their average in
    a dataframe
    :param data: data in OHLC
    :param w_pos: end position of window in data.
    :param nn: the nets to be used to perform the prediction
    :param encoder: the encoders of the nets
    :param params: the parameters of the config file.
    """
    model_names = list(params.model_names.keys())
    predictions = np.array([], dtype=np.float64)
    for name in model_names:
        w_size = encoder[name]._window_size
        # Select a window of data starting from 'w_pos', but if it is -1
        # that means that the window is the last w_size elements in data.
        if w_pos == -1:
            tick_group = data.tail(w_size + 1).iloc[-w_size - 1:-1]
        else:
            tick_group = data.iloc[w_pos - w_size:w_pos]
        next_close = predict_close(tick_group, encoder[name], nn[name], params)
        predictions = np.append(predictions, [next_close])

    # If the number of models is greater than 1, I also add statistics about
    # their result.
    if len(model_names) > 1:
        new_cols = ['actual', 'avg', 'avg_diff', 'median', 'med_diff', 'winner']
        # If I decide to use ensembles, I must add two new columns
        if params._ensemble is True:
            new_cols = new_cols + ['ensemble', 'ens_diff']
    else:
        new_cols = []

    df = pd.DataFrame([], columns=model_names + new_cols)
    df = df.append(dict(zip(params.model_names.keys(), predictions)),
                   ignore_index=True)

    if len(model_names) > 1:
        df['avg'] = df.mean(axis=1)
        df['median'] = df.median(axis=1)

    # When using ensemble, compute what the ensemble predicts, and add it.
    if params._ensemble:
        log.info('Refining prediction with ensemble.')
        with open(params._ensemble_path, 'rb') as file:
            ensemble_model = pickle.load(file)
        input_df = df[
            [u'10yw7', u'1yw7', u'1yw3', u'1yw10', u'median',
             u'5yw10', u'10yw3', u'5yw3', u'avg', u'5yw7']]
        df['ensemble'] = ensemble_model.predict(input_df)[0]

    return df
