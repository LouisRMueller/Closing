import numpy as np

from funcs_base import *
from funcs_OrderBooks import *
import psycopg2

class ExtractAdjustedClearing():
    s: str
    crossed_frame: pd.DataFrame
    today_symbols: list
    _allOrders: pd.DataFrame

    def __init__(self, date: str):
        self.connection = psycopg2.connect(database='six', user='external', password='IsTSsh_bek_35_usdiFdfS',
                                           host='130.82.29.50', port='5433', sslmode='disable')
        # self.connection = psycopg2.connect(database='six', user='postgres', password='1234',
        #                               host='localhost', port='5433', sslmode='disable')
        pd.set_option("display.max_columns", 17)
        pd.set_option('display.width', 300)

        self._d = date
        print(f"{date} INITIATED")
        t0 = time.perf_counter()
        self.import_order_data(date)
        # print(f"\n{self._d} IMPORTED ({time.time() - t0:.2f} seconds)")
        print(f"{date} IMPORTED ({time.perf_counter() - t0:.2f} seconds)")
        self.process_orders()
        print(f"{self._d} CALCULATED ({time.perf_counter() - t0:.2f} seconds)")
        self.export_lagged_uncrossings()
        self.export_key_uncrossings()
        print(f"{self._d} EXPORTED ({time.perf_counter() - t0:.2f} seconds)")

    def import_order_data(self, date: str) -> None:
        sql = f"""SELECT date, symbol, price, bids, asks FROM close.auction_long WHERE date = '{date}' AND lag = 999 ORDER BY date, symbol, price"""

        orders = pd.read_sql(sql, self.connection, index_col=['date', 'symbol', 'price'])
        self._allOrders = orders.fillna(0).astype(int)

    def process_orders(self) -> None:
        """Uncross order books including removal of certain liquidity percentage"""
        removals = np.arange(0, 1.01, 0.05)
        data = self._allOrders

        grouper = data.groupby(['date', 'symbol'])
        values = data.reset_index()[['price', 'bids', 'asks']].to_numpy()

        indices = nb.typed.List(grouper.indices.values())

        out, names = process_uncrossing_numba(indices, values)
        frame = pd.DataFrame(out, index=grouper.groups, columns=names)
        frame.rename_axis(index=['date', 'symbol', 'lag'], inplace=True)
        self.crossed_frame = frame

    def export_lagged_uncrossings(self) -> None:
        export = self.crossed_frame[['price', 'volume']].reset_index()
        export = export[export.groupby('symbol')['volume'].transform(lambda values: np.nansum(values) > 0)]
        # print(export.head(60))
        valueString = data_to_value_string(data=export[['date', 'symbol', 'lag', 'price', 'volume']])

        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM close.crossed_long WHERE date = '{str(self._d)}'; ")
        cursor.execute(f"INSERT INTO close.crossed_long VALUES {valueString} ON CONFLICT DO NOTHING;")
        self.connection.commit()
        cursor.close()

    def export_key_uncrossings(self):
        export = self.crossed_frame.copy()
        export['preclose_midquote'] = export.groupby(['date', 'symbol'])['price'].transform(lambda values: values[0])
        export = export.loc[pd.IndexSlice[:, :, 999], :].reset_index()
        export = export[['date', 'symbol', 'preclose_midquote', 'price', 'volume',
                         'market_buy_exec', 'market_sell_exec', 'limit_buy_exec', 'limit_sell_exec',
                         'market_buy_not_exec', 'market_sell_not_exec', 'limit_buy_not_exec', 'limit_sell_not_exec']]
        # export['date'] = export['date'].astype(str)
        valueString = data_to_value_string(data=export)

        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM close.crossed_final_new WHERE date = '{str(self._d)}'")
        cursor.execute(f"INSERT INTO close.crossed_final_new VALUES {valueString} ON CONFLICT DO NOTHING;")
        self.connection.commit()
        cursor.close()

ExtractAdjustedClearing(date='2018-01-03')