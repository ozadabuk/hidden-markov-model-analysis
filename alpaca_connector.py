import alpaca_trade_api as tradeapi
import math
from datetime import datetime, timedelta

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
APCA_API_KEY_ID = "YOUR_API_KEY_HERE"
APCA_API_SECRET_KEY = "YOUR_SECRET_API_KEY_HERE"


api = tradeapi.REST(
    APCA_API_KEY_ID ,
    APCA_API_SECRET_KEY ,
    "https://paper-api.alpaca.markets"
)

def get_stock_returns(sym, date):
    from_date = date
    to_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    aggr = api.polygon.historic_agg_v2(sym, 1, 'minute', from_date, to_date).df
    aggr = aggr.between_time('9:30', '16:30')
    print(aggr.to_string())
    closings = aggr['close']
    log_returns = []
    for i in range(1, len(closings)):
        log_returns.append(math.log(closings[i] / closings[i-1]) * 100)

    return log_returns