import pandas as pd

from funcs_base import *
from funcs_data import *
import psycopg2


class Connection:
    _imp_path = os.getcwd() + "\\import"
    _period_dict = {"01_Opening Auction": 'open_volume', '02_Continuous Trading': 'cont_volume',
                    '03_Closing Auction': 'close_volume', '04_Intraday Auction': 'intraday_volume', 'error': 'tradeatlast_volume'}
    START: str
    END: str
    LOCAL_IMPORT: bool

    _data_dict: dict
    _stockDays: pd.DataFrame
    QUANTILE_NUM = 3
    DATADIR = os.getcwd() + "\\Data"

    def __init__(self, from_db: bool = True, raw: bool = True):
        self.connection = psycopg2.connect(database='six', user='external', password='IsTSsh_bek_35_usdiFdfS',
                                           host='130.82.29.50', port='5433', sslmode='disable')
        # self.connection = psycopg2.connect(database='six', user='postgres', password='1234',
        #                               host='localhost', port='5433', sslmode='disable')

        pd.set_option("display.max_columns", 17)
        pd.set_option('display.width', 300)

        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        if from_db:
            start, stop = '2018-01-01', '2021-06-30'
            stocks = self.set_stock_universe(start, stop, min_days=125, min_avg_volume=100_000)
            master_data = self.import_master_database(start, stop, stocks)
            self.data = self.manipulate_data(master_data)
        else:
            if raw:
                master_data = self.import_master_local()
                self.data = self.manipulate_data(master_data)
            else:
                self.data = self.import_refined_data()

    def import_master_database(self, start: Optional[str] = None, end: Optional[str] = None, stocks: Optional[tuple] = None) -> pd.DataFrame:
        filename = "master_import"

        sql = f"""WITH prices AS (SELECT m.*,p.close_price, p.opening_price, p.last_price
                FROM main.prices p
                         INNER JOIN close.crossed_final_new m ON m.date = p.date AND m.symbol = p.symbol
                WHERE p.date BETWEEN '{start}' and '{end}'
                  AND p.symbol IN {tuple(stocks)}
                ORDER BY p.symbol, p.date),
     volumes AS (SELECT date,
                        symbol,
                        sum(CASE WHEN period = '01_Opening Auction' THEN turnover ELSE 0 END)    AS opening_volume,
                        sum(CASE WHEN period = '02_Continuous Trading' THEN turnover ELSE 0 END) AS continuous_volume,
                        sum(CASE WHEN period = '03_Closing Auction' THEN turnover ELSE 0 END)    AS closing_volume,
                        sum(CASE WHEN period = '04_Intraday Auction' THEN turnover ELSE 0 END)   AS intraday_auction_volume,
                        sum(CASE WHEN period = 'error' THEN turnover ELSE 0 END)                 AS trade_at_last_volume
                 FROM data.vols_phases
                 WHERE date BETWEEN '{start}' and '{end}'
                   AND symbol IN {tuple(stocks)}
                 GROUP BY symbol, date)
SELECT p.*,
       opening_volume,
       continuous_volume,
       closing_volume,
       intraday_auction_volume,
       trade_at_last_volume
FROM prices p
         LEFT JOIN volumes v
                   ON p.symbol = v.symbol AND p.date = v.date
ORDER BY p.symbol, p.date

        """
        imp = pd.read_sql(sql, self.connection, parse_dates=['date'], index_col=['symbol', 'date'])
        imp.to_hdf(f"{self.DATADIR}\\{filename}", key='df', mode='w')
        return imp

    def import_master_local(self) -> Union[pd.DataFrame, object]:
        return pd.read_hdf(f"{self.DATADIR}\\master_import", key='df')

    def import_refined_data(self) -> pd.DataFrame:

        return pd.DataFrame()

    def set_stock_universe(self, start: str, end: str, min_days: int, min_avg_volume: int) -> tuple:
        sql = f"""
SELECT symbol, date, turnover
FROM data.vols_phases
WHERE period = '03_Closing Auction'
  AND date BETWEEN '{start}' AND '{end}'
ORDER BY symbol, date"""
        volumes = pd.read_sql(sql, self.connection, index_col=['symbol', 'date'], parse_dates=['date'])

        # @nb.njit
        def filterSeries(values: np.ndarray, days: float, vol: float):
            if (values.shape[0] <= days) or (values.mean() <= vol):
                return np.full_like(values, np.nan)
            else:
                return np.full_like(values, values.mean())

        selection = volumes.groupby('symbol').transform(lambda x: filterSeries(x.to_numpy(), min_days, min_avg_volume)).dropna()
        stocks = selection.index.to_frame()['symbol'].unique()
        return tuple(stocks)

    def manipulate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = create_aggregated_overnight_data(data)
        return data
