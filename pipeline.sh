#!/bin/bash

tail -50 data/acciona_2019.csv > /tmp/tmp_ohlcv.csv
cd predictor
python predictor.py -f /tmp/tmp_ohlcv.csv predict
python predictor.py -f ../output/pred_acciona_2019_8yw20_8yw10_8yw05.csv ensemble
cd indicator
python indicators.py -i ../data/acciona_2019.csv --today
cd trader
python trader.py predict -f ../output/forecast_acciona_konkorde_2019.csv
