"""
This module computes stock indicators like Konkorde, from an OHLC file
(c) 2019, J. Renero
"""

import sys
from importlib import import_module

import pandas as pd

from indicator import Indicator
from ix_dictionary import IXDictionary
from logger import Logger

if __name__ == "__main__":
    params = IXDictionary(args=sys.argv)
    log: Logger = params.log

    # Call the proper constructor, from the name of indicator in arguments
    module = import_module(params.indicator_name)
    ix: Indicator = getattr(module, params.indicator_class)(params)

    # Decide what to do with the result
    if params.save is True:
        ix.save(params.today)
    elif params.merge:
        ix.save(params.today)
    else:
        if params.today:
            ix_value = ix.values.iloc[-1][ix.final_columns]
            print(ix_value.T.to_string())
            ix_value.to_json(params.json_indicator)
            log.info('Saved indicators to: {}'.format(params.json_indicator))
        else:
            pd.set_option('display.max_rows', -1)
            print(ix.values[ix.final_columns].to_string())
