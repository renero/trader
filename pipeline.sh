#!/bin/bash
set -e

SYMBOL="ANA.MC"
OHLC_FILE="../data/acciona_2019.csv"
TMP_OHLC="/tmp/tmp_ohlc.csv"
PREDS_FILE="../output/pred_acciona_2019_8yw20_8yw10_8yw05.csv"
FORECAST_FILE="../output/forecast_nov19.csv"
RL_MODEL="../output/rl_model_forecast_acciona_konkorde_2018b_1"

# Get latest info from OHLC, and update file
cd retriever
python retriever.py --symbol ${SYMBOL} --file ${OHLC_FILE}

# Generate a small sample to run predictions on it (smaller = faster)
head -1 ${OHLC_FILE} > ${TMP_OHLC}
tail -50 ${OHLC_FILE} >> ${TMP_OHLC}

# Predict What will be the next value for stock, from each network trained.
cd ../predictor
python predictor.py --file ${TMP_OHLC} predict
# Produce the ensemble from all predictions from all networks
python predictor.py --file ${PREDS_FILE} ensemble

# Generate Konkorde index for the latest addition to the OHLC file
cd ../indicators
python indicators.py -i ${OHLC_FILE} --today

# Update the forecast file with 
# - the closing for yesterday, 
# - the forecast for today
# - the values of the indicator (konkorde) for yesterday closing
cd ../updater
python updater.py --file ${FORECAST_FILE}

# Generate a trading recommendation
cd ../trader
python trader.py predict -f ${FORECAST_FILE} --model ${RL_MODEL}

echo "done."
