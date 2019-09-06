import pickle
from os.path import splitext, basename

import numpy as np
import pandas as pd
from tabulate import tabulate

from cs_encoder import CSEncoder
from cs_nn import Csnn
from cs_predict import CSPredict
from dataset import Dataset
from utils.file_io import valid_output_name


def split_datasets(encoder, cse, subtypes):
    """
    Prepare the training and test datasets from an list of existing CSE, for
    each of the model names considered (body and move).

    :param encoder: The encoder used to build the CSE list.
    :param cse: The list of CSE objects
    :param subtypes: The parameters read from file
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


def predict_dataset(dataset, encoder, nn, subtypes=None, split='test'):
    """
    Run prediction for body and move over the testsets in the dataset object
    :param dataset: the dataset
    :param encoder: the encoder to be used
    :param nn: the network to be used to make the prediction
    :param subtypes: normally 'body' and 'move'
    :param split: whether to perform the prediction over the 'test' or 'train'
        splits
    :return:
    """
    if subtypes is None:
        subtypes = ['body', 'move']
    prediction = {}
    for name in subtypes:
        if split == 'test':
            prediction[name] = CSPredict(dataset[name].X_test,
                                         dataset[name].y_test,
                                         encoder.onehot[name])
        else:
            prediction[name] = CSPredict(dataset[name].X_train,
                                         dataset[name].y_train,
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

    # Decode the prediction into a normal tick (I'm here!!!)
    prediction_df = pd.DataFrame([], columns=params._cse_tags)
    prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
    this_prediction = dict(zip(params._cse_tags, prediction_cs))
    prediction_df = prediction_df.append(this_prediction, ignore_index=True)
    params.log.info('Net {} ID {} -> {}:{}|{}|{}|{}'.format(
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

    # If the number of models is greater than 1, I also add statistics about
    # their result.
    model_names = list(params.model_names.keys())
    if len(model_names) > 1:
        new_columns = ['actual', 'avg', 'avg_diff', 'median', 'med_diff',
                       'winner']
        # If I decide to use ensembles, I must add two new columns
        if params._ensemble is True:
            new_columns = new_columns + ['ensemble', 'ens_diff']
    else:
        new_columns = []

    df = pd.DataFrame([], columns=model_names + new_columns)
    df = df.append(dict(zip(params.model_names.keys(), predictions)),
                   ignore_index=True)

    if len(model_names) > 1:
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

    model_names = list(params.model_names.keys())
    prediction['actual'] = real_value
    if len(model_names) > 1:
        prediction['avg_diff'] = diff_with('avg')
        prediction['med_diff'] = diff_with('median')
        if params._ensemble is True:
            prediction['ens_diff'] = diff_with('ensemble')
        max_diff = 10000000.0
        winner = ''
        for name in params.model_names.keys():
            diff = abs(
                prediction[name].iloc[-1] - prediction['actual'].iloc[-1])
            if diff < max_diff:
                max_diff = diff
                winner = name
        prediction.loc[:, 'winner'] = winner
    return prediction


def reorder_predictions(predictions, params):
    """
    Reorder predictions: If I'm producing predictions for the training set I
    reorder 'actual' values column in first column position. If I'm only
    producing a single prediction then, I simply drop all columns referring
    to the case when I know the response.
    :param predictions: the data frame with the predictions
    :param params: the parameters file.
    :return: the predictions reordered.
    """
    if params._predict_training is False:
        predictions = predictions.drop(
            ['actual', 'avg_diff', 'med_diff', 'winner'], axis=1)
    else:
        # Reorder columns to set 'actual' in first position
        actual_position = list(predictions.columns).index('actual')
        avg_position = []
        if len(params.model_names.keys()) > 1:
            avg_position = list(predictions.columns).index('avg')
        columns = [actual_position] + [i for i in
                                       range(actual_position)] + avg_position
        predictions = predictions.iloc[:, columns]
    return predictions


def save_predictions(predictions, params, log):
    predictions = reorder_predictions(predictions, params)
    if params._save_predictions is True:
        # Find a valid filename and save everything
        filename = valid_output_name(
            filename='pred_{}_{}'.format(
                splitext(basename(params._ticks_file))[0],
                '_'.join(params.model_names)),
            path=params._predictions_path,
            extension='csv')
        predictions.to_csv(filename, index=False)
        log.info('predictions saved to: {}'.format(filename))
    else:
        print(tabulate(predictions, headers='keys', tablefmt='psql',
                       showindex=False, floatfmt=['.1f']))
