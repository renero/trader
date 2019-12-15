#!/bin/bash

set -e 

OHLC="../data/^GDAXI/DAX_2018_2019.csv"

cd predictor
python predictor.py -c params.dax.yaml predict_training -f ${OHLC} --save -d 0
python predictor.py -c params.dax.yaml ensemble_predictions -f ../output/^GDAXI/pred_DAX_2018_2019_4yw30_4yw20_4yw10_1.csv --save -d 0

cd ../indicators
python indicators.py -c params.dax.yaml konkorde -f ${OHLC} --merge ../output/^GDAXI/forecast_DAX_2018_2019_4yw30_4yw20_4yw10_1.csv -d 0

