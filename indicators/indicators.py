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
        ix.save()
    elif params.merge is True:
        ix.merge()
    else:
        if params.today:
            ix.register()
        else:
            pd.set_option('display.max_rows', -1)
            print(ix.values[ix.final_columns].to_string())
