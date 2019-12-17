#!/bin/bash

set -ex
shopt -s nullglob

usage()
{
    cat <<EOF
usage: $0 -c CONFIG -s SYMBOL -f FILE [-h|--help]

This script resets the predictions and forecasts file, using the OHLCV file
specified in the argument.

optional arguments:
  -h, --help      show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                  The YAML parameters file to be used to run predictor and
                  indicators.
  -f FILE, --file FILE
                  The OHLCV file to be used to regenerate predictions and forecast.
  -s SYMBOL, --symbol SYMBOL
                  The acronym of the symbol to be retrieved from the
                  stock information provider.
EOF
}

#
# Main --check arguments
#
SYMBOL=""
OHLC=""
CONFIG=""
while [ "$1" != "" ]; do
    case $1 in
        -c | --config ) shift
                        CONFIG=$1
                        ;;
        -f | --file )   shift
                        OHLC=$1
                        ;;
        -s | --symbol ) shift
                        SYMBOL=$1
                        ;;
        -h | --help )   usage
                        exit
                        ;;
        * )             usage
                        exit 1
    esac
    shift
done

# Check that argument SYMBOL has been passed
if [ "$SYMBOL" == "" ] || [ "$OHLC" == "" ] || [ "$CONFIG" == "" ]; then
  usage
  exit 1
fi

cd predictor
python predictor.py -c ${CONFIG} predict_training -f ${OHLC} --save --output newpredictions
python predictor.py -c ${CONFIG} ensemble_predictions -f ../output/${SYMBOL}/newpredictions.csv --save --output newensemble

cd ../indicators
python indicators.py -c ${CONFIG} konkorde -f ${OHLC} --merge ../output/${SYMBOL}/newensemble.csv --output newforecast
