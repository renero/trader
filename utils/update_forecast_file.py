import json


def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


ohlc = read_json('../output/tmp_ohlcv.json')
ensemble = read_json('../output/tmp_forecast.json')
indicator = read_json('../output/tmp_konkorde.json')

forecast_items = [ohlc['Date'], float(ohlc['Close']), ensemble['w_avg'],
                  indicator['verde'], indicator['azul']]


def f(x):
    if type(x) == str:
        return x
    return '{:.2f}'.format(x)


csv_row = ','.join(map(f, forecast_items))
print(csv_row)
