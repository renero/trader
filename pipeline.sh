#!/bin/bash
set -e

# Get latest info from OHLC, and update file
cd retriever
python retriever.py --symbol ANA.MC --file ../data/acciona_2019.csv
cd ..

# Generate a small sample to run predictions on it (smaller = faster)
head -1 data/acciona_2019.csv > /tmp/tmp_ohlcv.csv
tail -50 data/acciona_2019.csv >> /tmp/tmp_ohlcv.csv

# Predict What will be the next value for stock, from each network trained.
cd predictor
python predictor.py -f /tmp/tmp_ohlcv.csv predict
# Produce the ensemble from all predictions from all networks
python predictor.py -f ../output/pred_acciona_2019_8yw20_8yw10_8yw05.csv ensemble

# Generate Konkorde index for the latest addition to the OHLC file
cd ../indicators
python indicators.py -i ../data/acciona_2019.csv --today

# Generate a trading recommendation
cd ../trader
python trader.py predict -f ../output/forecast_acciona_konkorde_2019.csv -m ../output/rl_model_forecast_acciona_konkorde_2018b_1
