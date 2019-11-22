#!/bin/bash
set -e

SYMBOL="ANA.MC"
OHLC_FILE="../data/acciona_2019.csv"
TMP_OHLC="/tmp/tmp_ohlc.csv"
PREDS_FILE="../output/pred_acciona_2019_8yw20_8yw10_8yw05.csv"
FORECAST_FILE="../output/forecast_nov19.csv"
RL_MODEL="../staging/rl_model_acciona_2018b"
PORTFOLIO="../staging/portfolio_acciona_nov19b.json"
SCALER="../staging/scaler_konkorde_acciona_2018.pickle"

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
python indicators.py -f ${OHLC_FILE} --today --scaler-file ${SCALER}

# Update the forecast file with 
# - the closing for yesterday, 
# - the forecast for today
# - the values of the indicator (konkorde) for yesterday closing
cd ../updater
python updater.py --file ${FORECAST_FILE}

# Generate a trading recommendation
cd ../trader
python trader.py predict -f ${FORECAST_FILE} --model ${RL_MODEL} --portfolio ${PORTFOLIO}

# Extract the action to be taken and base price to send it over.
cd ..
LATEST_ACTION="output/tmp_action.json"
LATEST_OHLC="output/tmp_ohlcv.json"
BASEPRICE=`cat ${LATEST_OHLC}|cut -d ',' -f 3|cut -d ':' -f2|tr -d '"'`
ACTION=`cat ${LATEST_ACTION}|tr -d "}"|awk -F ':' '{print $2}'|tr -d '"'`
if [ "$ACTION" == "sell" ]; then
    REFERENCE="minimum at ${BASEPRICE}"
elif [ "$ACTION" == "buy" ]; then
    REFERENCE="maximum at ${BASEPRICE}"
else
    REFERENCE=""
fi
echo -n "The recommendation is... "
echo "${ACTION} ${REFERENCE}"

# Simulate the portfolio so far, to check how it goes.
echo "Simulation for existing portfolio ${PORTFOLIO}"
cd trader
python trader.py simulate --no-dump -f ${FORECAST_FILE} --model ${RL_MODEL} --debug 0
cd ..

