#!/bin/sh
#
# Retrieve symbol value for data and place it in a JSON file to later run
# the pipeline with the --no-retrieve flag.
#

set -e
shopt -s nullglob

usage()
{
    cat <<EOF
usage: $0 -d DATE -s SYMBOL [-h|--help]

optional arguments:
  -h, --help      show this help message and exit
  -d DATE, --date DATE
                  The date for which I will try to retrieve the OHLCV from.
  -s SYMBOL, --symbol SYMBOL
                  The acronym of the symbol to be retrieved from the
                  stock information provider.
EOF
}

#
# Main --check arguments
#
SYMBOL=""
DATE=""
while [ "$1" != "" ]; do
    case $1 in
        -d | --date )   shift
                        DATE=$1
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
if [ "$SYMBOL" == "" ] || [ "$DATE" == "" ]; then
  usage
  exit 1
fi

# Variables
NOW=$(date '+%F %T')
LOGHEADER="$NOW "
TMP_DIR="/tmp/trader"
LATEST_OHLC="${TMP_DIR}/${SYMBOL}/tmp_ohlc.json"
TMP_FILE="${TMP_DIR}/$$_1"
TMP_FILE2="${TMP_DIR}/$$_2"
URL="https://www.alphavantage.co/query\?"
APIKEY="HF9S3IZSBKSKHPV3"

# Create tmp directory if does not exist
if [ ! -d "${TMP_DIR}/${SYMBOL}" ]; then
  echo "${LOGHEADER} Creating TMP directories"
  mkdir -p "${TMP_DIR}/${SYMBOL}";
fi

# Retrieve and grep
echo "${LOGHEADER} Retrieving data..."
/usr/bin/curl --silent -G --data-urlencode "function=TIME_SERIES_DAILY" --data-urlencode "symbol=${SYMBOL}" \
  --data-urlencode "apikey=${APIKEY}" --data-urlencode "datatype=csv" ${URL} --output ${TMP_FILE}
/usr/bin/grep "${DATE}" "${TMP_FILE}" > ${TMP_FILE2}

/usr/local/bin/awk 'BEGIN {FS=",";}
{
  printf("{\"Date\": \"%s\", \"Open\": \"%s\", \"High\": \"%s\", \"Low\": \"%s\", \"Close\": \"%s\", \"Volume\": \"%d\"}", \
  $1, $2, $3, $4, $5, $6)
}' ${TMP_FILE2} > ${LATEST_OHLC}
/bin/rm ${TMP_FILE} ${TMP_FILE2}
echo "${LOGHEADER} Temporary OHLCV generated: ${LATEST_OHLC}"
