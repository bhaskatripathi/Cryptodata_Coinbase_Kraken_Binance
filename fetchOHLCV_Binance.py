'Downloads OHLCV data from Binance for the given currency pairs, start and end dates, timeframe(1d- daily) and converts to pandas dataframe.
' The code also savse the pandas dataframe of each currency pair to .csv files and shows the count of records for the downloaded data.
' Copyright(c)-2023 - Bhaskar Tripathi, https://www.bhaskartripathi.com
import pandas as pd
import ccxt
from datetime import datetime

exchange = ccxt.binance()

# Define currency pairs
currency_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DASH/USDT', 'LTC/USDT']

# Define start and end timestamps (in milliseconds)
since = exchange.parse8601('2017-01-01T00:00:00Z')
end = exchange.parse8601('2023-04-01T00:00:00Z')

# Define batch size
batch_size = 500

for pair in currency_pairs:
    # Fetch data from exchange incrementally
    data = []
    start = since
    while start < end:
        ohlcv = exchange.fetch_ohlcv(
            symbol=pair,
            timeframe='1d',
            since=start,
            limit=batch_size,
            params={'endTime': end}
        )
        if len(ohlcv) == 0:
            break
        data += ohlcv
        start = ohlcv[-1][0] + 1

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
    df.set_index('Time', inplace=True)

    # Save data to CSV file
    filename = f'{pair.replace("/", "")}.csv'
    df.to_csv(filename)

    # Print status
    print(f'Saved data for {pair} to {filename}')
    print(f'Start date: {df.index.min().date()}')
    print(f'End date: {df.index.max().date()}')
    print(f'Number of records: {len(df)}')
