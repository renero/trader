from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame

# log = Logger(3)
from cs_dictionary import CSDictionary
from cs_encoder import CSEncoder
from cs_nn import CS_NN


def single_prediction(
        data: DataFrame,
        w_pos: int,
        nn: Dict[str, CS_NN],
        encoder: CSEncoder,
        params: CSDictionary) -> DataFrame:
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
        w_size = encoder[name].params.window_size
        # Select a window of data starting from 'w_pos', but if it is -1
        # that means that the window is the last w_size elements in data.
        if w_pos == -1:
            tick_group = data.tail(w_size + 1).iloc[-w_size - 1:-1]
        else:
            tick_group = data.iloc[w_pos - w_size:w_pos]
        next_close = predict_next(tick_group, encoder[name], nn[name], params)
        predictions = np.append(predictions, [next_close])

    # If the number of models is greater than 1, I also add statistics about
    # their result.
    if len(model_names) > 1:
        new_cols = ['actual', 'avg', 'avg_diff', 'median', 'med_diff', 'winner']
    else:
        new_cols = []

    df = pd.DataFrame([], columns=model_names + new_cols)
    df = df.append(dict(zip(params.model_names.keys(), predictions)),
                   ignore_index=True)

    if len(model_names) > 1:
        df['avg'] = df.mean(axis=1)
        df['median'] = df.median(axis=1)

    return df


def predict_next(ticks: DataFrame,
                 encoder: CSEncoder,
                 nn: Dict[str, CS_NN],
                 params: CSDictionary) -> float:
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
    if ticks.shape[0] != encoder.params.window_size:
        info_msg = 'Tickgroup resizing: {} -> {}'
        params.log.info(
            info_msg.format(ticks.shape[0], encoder.params.window_size))
        ticks = ticks.iloc[-encoder.params.window_size():, :]
        ticks.reset_index()

    # encode the tick in CSE and OH. Reshape it to the expected LSTM format.
    cs_tick = encoder.transform(ticks[params.ohlc_tags])
    pred_body_cs = predict_body(cs_tick, encoder, nn)
    pred_move_cs = predict_move(cs_tick, encoder, nn)
    prediction_cs = pred_body_cs + pred_move_cs

    # Build a single row dataframe with the entire prediction
    prediction_df = pd.DataFrame(
        columns=params.cse_tags,
        data=np.array(prediction_cs).reshape(-1, len(params.cse_tags)))

    cs_values = '|'.join(
        '{}'.format(_) for _ in prediction_df[params.cse_tags].values[0])
    params.log.info(f"N2t {nn['body'].name} -> {cs_values}")

    # Convert the prediction to a real tick
    # TODO: Keep the date in predictions, so we can see for what date the
    #       prediction is being produced
    # TODO: Instead of returning only the CLOSE value, return the entire
    #       candlestick.
    pred = encoder.inverse_transform(prediction_df, cs_tick[-1])
    return pred['c'].values[-1]


def predict_body(cs_tick: List[CSEncoder],
                 encoder: CSEncoder,
                 nn: Dict[str, CS_NN]) -> List[str]:
    cs_tick_body_oh = encoder.onehot['body'].encode(encoder.body(cs_tick))
    input_body = cs_tick_body_oh.values[np.newaxis, :, :]
    # get a prediction from the proper networks, for the body part
    raw_prediction = nn['body'].predict(input_body)[0]
    pred_body_oh = nn['body'].hardmax(raw_prediction)
    pred_body_cs = encoder.onehot['body'].decode(pred_body_oh)
    return pred_body_cs.tolist()


def predict_move(cs_tick: List[CSEncoder],
                 encoder: CSEncoder,
                 nn: Dict[str, CS_NN]) -> List[str]:
    cs_tick_move_oh = encoder.onehot['move'].encode(encoder.move(cs_tick))
    input_move = cs_tick_move_oh.values[np.newaxis, :, :]
    # Repeat everything with the move:
    # get a prediction from the proper network, for the MOVE part
    pred_length = len(encoder.onehot['move'].states)
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
    return pred_move_cs
