from os import getcwd
from pathlib import Path
from yaml import safe_load, YAMLError
from utils.my_dict import MyDict
from datetime import date

import requests
import json


class MarketData(MyDict):
    def __init__(self, default_params_filename='params.yaml', default_data_folder='../data/', **kwargs):
        """
        Read the parameters from a default filename
        :return:
        """
        super().__init__(**kwargs)
        params = {}
        cwd = Path(getcwd())
        params_path: str = str(cwd.joinpath(default_params_filename))

        with open(params_path, 'r') as stream:
            try:
                params = safe_load(stream)
            except YAMLError as exc:
                print(exc)

        self.add_dict(self, params)

        # World Trading Data token not provided
        if not self._world_trading_data or '_api_token' not in self._world_trading_data or self._world_trading_data['_api_token'] == 'CHANGE_ME':
            raise AssertionError('World Trading Data API Token not provided.')

        self._currencies = {}

        for f in self._world_trading_data._forex:
            currencies = f['base']+f['convert_to']

            forex_path: str = str(cwd.joinpath(default_data_folder, 'forex_'+currencies.lower()+'_history.json'))

            with open(forex_path, 'r') as stream:
                self._currencies[currencies] = json.load(stream)

        self._default_data_folder = default_data_folder

    def getPrice(self):
        cwd = Path(getcwd())
        today = date.today()
        prices = {}

        for f in self._world_trading_data._forex:
            url = self._world_trading_data._base_url+'forex?base='+f['base']+'&api_token='+self._world_trading_data._api_token

            cur = f['base']+f['convert_to']

            r = requests.get(url)
            response = r.json()

            self._currencies[cur]['history'][today.strftime("%Y-%m-%d")] = response['data']['USD']

            prices[cur] = response['data']['USD']

            # Writing JSON data
            forex_path: str = str(cwd.joinpath(self._default_data_folder, 'forex_'+cur.lower()+'_history.json'))

            with open(forex_path, 'w') as stream:
                json.dump(self._currencies[cur], stream)

        return prices

if __name__== "__main__":
    md = MarketData()
    print(md.getPrice())
