import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cs_encoder import CSEncoder
from cs_nn import Csnn
from dataset import Dataset
from predict import Predict


def plot_body_prediction(raw_prediction, pred_body_cs):
    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(raw_prediction)
    winner_prediction = max(raw_prediction, key=abs)
    pos = np.where(raw_prediction == winner_prediction)[0][0]
    plt.plot(pos, winner_prediction, 'yo')
    plt.annotate(
        '{}={}'.format(pos, pred_body_cs[0]),
        xy=(pos, winner_prediction),
        xytext=(pos + 0.5, winner_prediction))
    plt.xticks(np.arange(0, len(raw_prediction), 1.0))
    ax.xaxis.label.set_size(6)


def plot_move_prediction(y, Y_pred, pred_move_cs, num_predictions,
                         pred_length):
    # find the position of the absmax mvalue in each of the arrays
    y_maxed = np.zeros(y.shape)
    for i in range(num_predictions):
        winner_prediction = max(Y_pred[i], key=abs)
        pos = np.where(Y_pred[i] == winner_prediction)[0][0]
        y_maxed[(i * pred_length) + pos] = winner_prediction

    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(y)
    for i in range(len(y_maxed)):
        if y_maxed[i] != 0.0:
            plt.plot(i, y[i], 'yo')
            plt.annotate(
                '{}={}'.format(i, pred_move_cs[int(i / pred_length)]),
                xy=(i, y[i]),
                xytext=(i + 0.6, y[i]))
    plt.xticks(np.arange(0, len(y), 2.0))
    ax.xaxis.label.set_size(2)
    for vl in [i * pred_length for i in range(num_predictions + 1)]:
        plt.axvline(x=vl, linestyle=':', color='red')
    plt.show()


def prepare_datasets(encoder, cse, subtypes):
    """
    Prepare the training and test datasets from an list of existing CSE, for
    each of the model names considered (body and move).

    :param encoder: The encoder used to build the CSE list.
    :param cse: The list of CSE objects
    :param params: The parameters read from file
    :return: The datasets for each of the models that need to be built. The
        names of the models specify the 'body' part and the 'move' part.
    """
    cse_data = {}
    oh_data = {}
    dataset = {}
    for subtype in subtypes:
        call_select = getattr(encoder, '{}'.format(subtype))
        cse_data[subtype] = Dataset().adjust(call_select(cse))
        oh_data[subtype] = encoder.onehot[subtype].encode(cse_data[subtype])
        dataset[subtype] = Dataset().train_test_split(oh_data[subtype])
    return dataset


def train_nn(dataset, subtypes):
    """
    Train a model.
    """
    nn = {}
    for subtype in subtypes:
        nn[subtype] = Csnn(None, subtype)

        window_size = dataset[subtype].X_train.shape[1]
        num_categories = dataset[subtype].X_train.shape[2]

        nn[subtype].build_model(window_size, num_categories).train(
            dataset[subtype].X_train, dataset[subtype].y_train).save()

    return nn


def load_nn(model_names, subtypes):
    """
    """
    nn = {}
    for name in model_names.keys():
        nn[name] = {}
        for subtype in subtypes:
            nn[name][subtype] = Csnn(name, subtype)
            nn[name][subtype].load(model_names[name][subtype])
    return nn


def load_encoders(model_names):
    """Load a encoder for each network"""
    encoder = {}
    for name in model_names:
        encoder[name] = CSEncoder().load(model_names[name]['encoder'])
    return encoder


def predict_testset(dataset, encoder, nn, subtypes):
    """
    Run prediction for body and move over the testsets in the dataset object
    :param dataset: the data
    :param encoder:
    :param nn:
    :param params:
    :return:
    """
    prediction = {}
    for name in subtypes:
        prediction[name] = Predict(dataset[name].X_test,
                                   dataset[name].y_test,
                                   encoder.onehot[name])
        call_predict = getattr(prediction[name],
                               'predict_{}_batch'.format(name))
        call_predict(nn[name])
    return prediction


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
        params.log.info(info_msg.format(ticks.shape[0], encoder.window_size()))
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
    # plot_move_prediction(y, Y_pred, pred_move_cs, num_predictions, pred_length)

    # Decode the prediction into a normal tick (I'm here!!!)
    prediction_df = pd.DataFrame([], columns=params._cse_tags)
    prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
    this_prediction = dict(zip(params._cse_tags, prediction_cs))
    prediction_df = prediction_df.append(this_prediction, ignore_index=True)
    params.log.info('Net {} ID 0x{} -> {}:{}|{}|{}|{}'.format(
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


def single_prediction(tick_group, nn, encoder, params):
    """
    Make a single prediction over a list of ticks. It uses all the
    networks loaded to produce all their predictions and their average in
    a dataframe
    """
    predictions = np.array([], dtype=np.float64)
    for name in params.model_names:
        next_close = predict_close(tick_group, encoder[name], nn[name], params)
        predictions = np.append(predictions, [next_close])
    model_names = list(params.model_names.keys())
    new_columns = ['actual', 'avg', 'avg_diff', 'median', 'med_diff',
                   'winner']

    # If I decide to use ensembles, I must add two new columns
    if params._ensemble is True:
        new_columns = new_columns + ['ensemble', 'ens_diff']

    df = pd.DataFrame([], columns=model_names + new_columns)
    df = df.append(dict(zip(params.model_names.keys(), predictions)),
                   ignore_index=True)
    df['avg'] = df.mean(axis=1)
    df['median'] = df.median(axis=1)

    # When using ensemble, compute what the ensemble predicts, and add it.
    if params._ensemble:
        params.log.info('Refining prediction with ensemble.')
        with open(params._ensemble_path, 'rb') as file:
            ensemble_model = pickle.load(file)
        input_df = df[
            [u'10yw7', u'1yw7', u'1yw3', u'1yw10', u'median',
             u'5yw10', u'10yw3', u'5yw3', u'avg', u'5yw7']]
        df['ensemble'] = ensemble_model.predict(input_df)[0]
    return df


def add_supervised_info(prediction, real_value, params):
    """
    Add to the single prediction the actual value expected (if any), the
    difference between the mean predicted value and the actual value, and
    which network name is the one producing the closest value
    :param prediction: the prediction made by all networks in a form of
    DataFrame with the columns being the name of the networks
    :param real_value: the actual value that followed the sequence presented
    to the set of networks. This is the value to be predicted.
    :param params: The parameters of the whole enchilada
    :return: The dataframe of the predictions enriched with the actual
    prediction, the difference between the mean and the actual, and the winner
    network.
    """

    def diff_with(column_label):
        return abs(prediction['actual'] - prediction[column_label])

    prediction['actual'] = real_value
    prediction['avg_diff'] = diff_with('avg')
    prediction['med_diff'] = diff_with('median')
    if params._ensemble is True:
        prediction['ens_diff'] = diff_with('ensemble')
    max_diff = 10000000.0
    winner = ''
    for name in params.model_names.keys():
        diff = abs(prediction[name].iloc[-1] - prediction['actual'].iloc[-1])
        if diff < max_diff:
            max_diff = diff
            winner = name
    prediction.loc[:, 'winner'] = winner
    return prediction
