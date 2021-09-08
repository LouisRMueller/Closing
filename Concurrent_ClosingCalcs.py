from time import time
import os
import copy
from funcs_base import *
from funcs_OrderBooks import *

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import concurrent.futures

START = '2018-02-14'
END = '2021-07-31'
NUM_RUNNERS = 8


class Research:
    all_orders: pd.DataFrame

    def __init__(self):
        pd.set_option("display.max_columns", 17)
        pd.set_option('display.width', 300)
        self.connection = psycopg2.connect(database='six', user='external', password='IsTSsh_bek_35_usdiFdfS',
                                           host='130.82.29.50', port='5433', sslmode='disable')
        # self.connection = psycopg2.connect(database='six', user='postgres', password='1234',
        #                               host='localhost', port='5433', sslmode='disable')

    def import_final_auction_books(self, date: str):
        sql = f"""SELECT date, symbol, price, bids, asks FROM close.auction_long WHERE date = '{date}' AND lag = 999 ORDER BY date, symbol, price"""

        orders = pd.read_sql(sql, self.connection, index_col=['date', 'symbol', 'price'])
        self.all_orders = orders.fillna(0).astype(int)
        self.today_symbols = self.all_orders.index.to_frame()['symbol'].unique()

    def import_stockdays(self, startDate: str, endDate: str):
        sql = f"""  SELECT * FROM data.stockdays
                    WHERE onbook_date BETWEEN '{startDate}' AND '{endDate}'
                    ORDER BY onbook_date, symbol"""

        imp = pd.read_sql(sql, self.connection, parse_dates=['onbook_date'])['onbook_date']
        return imp.astype(str).unique()



class SensitivityAnalysis(Research):
    collector: pd.DataFrame

    def __init__(self, date='2018-01-03'):
        t0 = time.perf_counter()
        super().__init__()
        self.import_final_auction_books(date=date)
        print(f"{date} IMPORTED ({time.perf_counter() - t0:.2f} seconds)")
        self.date = date

        self.process()
        self.results_to_database()
        print(f"{date} EXPORTED ({time.perf_counter() - t0:.2f} seconds)")


    def process(self) -> None:
        """
        This function is supposed to exeucte the required calculations and add it to an appropriate data format.
        It calls other helper functions in order to determine the results of the analysis.
        """
        percentages = np.arange(0, 1.01, 0.05).round(2)
        modes, sides = ['execution', 'liquidity', 'market'], ['bid', 'ask', 'both']
        index = pd.MultiIndex.from_product([[self.date], self.today_symbols, percentages], names=['date', 'symbol', 'removal'])
        collector = pd.DataFrame(columns=pd.MultiIndex.from_product([modes, sides]), index=index)
        # for symbol in ['ABBN', 'NOVN']:
        for symbol in self.today_symbols:
            frame = add_market_orders_if_none(self.all_orders.loc[(self.date, symbol)])
            close_volume = calculate_uncross(values=frame)['trade_vol']
            if np.isnan(close_volume):
                continue

            for perc in percentages:
                for mode, side in collector.columns:
                    adj_frame = remove_orders_numba(values=frame, mode=mode, side=side, perc=perc)
                    close_uncross = calculate_uncross(adj_frame)
                    collector.loc[(self.date, symbol, perc), (mode, side)] = close_uncross['price']

        self.collector = collector

    def plotClosingUncross(self):
        closeDict = self._remove_orders(date='2019-03-15', symbol='NESN', perc=0)
        uncrossDict = self._calc_uncross(bids=closeDict['bids'], asks=closeDict['asks'], exportFull=True)

        cumBookData = pd.DataFrame.from_dict(uncrossDict['cumData'])
        df = pd.melt(cumBookData, id_vars='prices', var_name='Side', value_name='Volume')
        closingPrice = uncrossDict['results']['price']
        closingVol = uncrossDict['results']['trade_vol']

        fig, ax = plt.subplots(1, 1, figsize=(20 / 2.5, 10 / 2.5))
        sns.lineplot(ax=ax, data=df, x='prices', y='Volume', hue='Side', markers='.', lw=1)
        ax.set_xlim(left=80, right=110)
        ax.set_ylim(bottom=0)
        l1, l2 = ax.lines[0], ax.lines[1]
        x1, y1 = l1.get_xydata()[:, 0], l1.get_xydata()[:, 1]
        x2, y2 = l2.get_xydata()[:, 0], l2.get_xydata()[:, 1]
        ax.fill_between(x1, y1, y2, where=y1 > y2, alpha=0.15)
        ax.fill_between(x2, y1, y2, where=y1 < y2, alpha=0.15)
        ax.axvline(closingPrice, color='k', lw=0.8, ls='solid')
        ax.axhline(closingVol, color='k', lw=1, ls='dotted')
        ax.set_title(f"NESN Closing Book on March 15, 2019 (Turnover: CHF {closingPrice * closingVol / 10 ** 6:.0f}mn.)")
        ax.set_xlabel('Calculated closing price: CHF {0: 0.3f}'.format(closingPrice))
        ax.set_ylabel(f'Number of shares matched: {closingVol / 10 ** 6 : 0.1f} million')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(f"{os.getcwd()}\\02 Slides January\\Figures\\AuctionBookNesn.pdf")
        plt.show()

    def results_to_database(self) -> None:
        export = self.collector.copy()
        mapper = export[('execution', 'bid')].groupby(['date', 'symbol']).transform(lambda x: False if np.isnan(x[0]) else True)
        export = export.loc[mapper, :].reset_index()

        valueString = data_to_value_string(data=export)

        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM close.sensitivity WHERE date = '{str(self.date)}'")
        cursor.execute(f"INSERT INTO close.sensitivity VALUES {valueString} ON CONFLICT DO NOTHING;")
        self.connection.commit()
        cursor.close()


# profiler(SensitivityAnalysis)

if __name__ == '__main__':
    orderedDays = Research().import_stockdays(startDate=START, endDate=END)
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_RUNNERS) as executor:
        executor.map(SensitivityAnalysis, orderedDays)

# if __name__ == '__main__':
#     orderedDays = Research().import_stockdays(startDate=START, endDate=END)
#     for i in range(0, len(orderedDays), NUM_RUNNERS):
#         daysSelection = orderedDays[i:i + NUM_RUNNERS]
#         with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_RUNNERS) as executor:
#             executor.map(SensitivityAnalysis, daysSelection)
