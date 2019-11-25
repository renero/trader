#!/bin/bash
#
# Main pipeline to run inside the docker and predict action to be taken
# today on a given stock
#

set -e

usage()
{
    cat <<EOF
    usage: $0 -s SYMBOL [-h] [-c CONFIG_FILE]

    optional arguments:
      -h, --help      show this help message and exit
      -c CONFIG_FILE, --config CONFIG_FILE
                      Relative path to configuration file to be used (YAML)
                      by predictor and trader (same name).
      -s SYMBOL       The acronym of the symbol to be retrieved from the
                      stock information provider.
EOF
}

#
# Main --check arguments
#

SYMBOL=""
CONFIG_FILE=params.yaml
while [ "$1" != "" ]; do
    case $1 in
        -c | --config )         shift
                                CONFIG_FILE=$1
                                ;;
        -s | --symbol )         shift
                                SYMBOL=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Check that argument SYMBOL has been passed
if [[ "$SYMBOL" == "" ]]; then
  usage
  exit 1
fi

# Set environment
OHLC_FILE="../data/${SYMBOL}/acciona_2019.csv"
TMP_OHLC="/tmp/${SYMBOL}/tmp_ohlc.csv"
PREDS_FILE="../staging/${SYMBOL}/pred_acciona_2019_8yw20_8yw10_8yw05.csv"
FORECAST_FILE="../staging/${SYMBOL}/forecast_nov19.csv"
RL_MODEL="../staging/${SYMBOL}/rl_model_acciona_2018b"
PORTFOLIO="../staging/${SYMBOL}/portfolio_acciona_nov19b.json"
SCALER="../staging/${SYMBOL}/scaler_konkorde_acciona_2018.pickle"
LATEST_ACTION="output/${SYMBOL}/tmp_action.json"
LATEST_OHLC="output/${SYMBOL}/tmp_ohlcv.json"


# Get latest info from OHLC, and update file
cd retriever
python retriever.py --symbol ${SYMBOL} --file ${OHLC_FILE}

# Generate a small sample to run predictions on it (smaller = faster)
head -1 ${OHLC_FILE} > ${TMP_OHLC}
tail -50 ${OHLC_FILE} >> ${TMP_OHLC}

# Predict What will be the next value for stock, from each network trained.
cd ../predictor
python predictor.py --config ${CONFIG_FILE} --file ${TMP_OHLC} predict
# Produce the ensemble from all predictions from all networks
python predictor.py --config ${CONFIG_FILE} --file ${PREDS_FILE} ensemble

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
python trader.py predict --config ${CONFIG_FILE} -f ${FORECAST_FILE} --model ${RL_MODEL} --portfolio ${PORTFOLIO}

# Extract the action to be taken and base price to send it over.
cd ..
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
python trader.py simulate --config ${CONFIG_FILE} --no-dump -f ${FORECAST_FILE} --model ${RL_MODEL} --debug 0
cd ..
