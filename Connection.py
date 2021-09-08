import numpy as np
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

    _stockDays: pd.DataFrame
    QUANTILE_NUM = 4
    DATADIR = os.getcwd() + "\\Data"
    _FIGDIR = f"{os.getcwd()}\\Figures"
    _helpLineKwargs = dict(color='black', lw=0.8, zorder=1)
    _confLineKwargs = dict(color='black', lw=1, ls='dotted')
    _mainLineKwargs = dict(lw=1, ls='solid')
    # _mainLineKwargs = dict(color=sns.color_palette('plasma', 1)[0], lw=1, ls='solid')
    _errorKwargs = dict(facecolor='white', linestyle='dotted', edgecolor='black', linewidth=1)
    _shadingKwargs = dict(color='lightgrey', alpha=0.5)
    _palette = 'inferno_r'  # 'cubehelix'
    _figsizeNorm, _figsizeDouble, _figsizeTriple = (19 / 2.5, 10 / 2.5), (19 / 2.5, 20 / 2.5), (19 / 2.5, 28 / 2.5)

    def __init__(self):
        self.connection = psycopg2.connect(database='six', user='external', password='IsTSsh_bek_35_usdiFdfS',
                                           host='130.82.29.50', port='5433', sslmode='disable')
        # self.connection = psycopg2.connect(database='six', user='postgres', password='1234',
        #                               host='localhost', port='5433', sslmode='disable')

        start, stop = '2018-01-01', '2021-06-30'
        stocks = self.set_stock_universe(start, stop, top_stocks=100, minimum_days=250)
        self.master_data = self.import_master_database(start, stop, stocks)

    def import_master_database(self, start: Optional[str] = None, end: Optional[str] = None, stocks: Optional[tuple] = None) -> pd.DataFrame:
        filename = "master_import"

        # noinspection SqlShouldBeInGroupBy
        sql = f"""
WITH volumes AS (SELECT date,
                        symbol,
                        sum(CASE WHEN period = '01_Opening Auction' THEN turnover ELSE 0 END)    AS opening_volume,
                        sum(CASE WHEN period = '02_Continuous Trading' THEN turnover ELSE 0 END) AS continuous_volume,
                        sum(CASE WHEN period = '03_Closing Auction' THEN turnover ELSE 0 END)    AS closing_volume,
                        sum(CASE WHEN period = '04_Intraday Auction' THEN turnover ELSE 0 END)   AS intraday_auction_volume,
                        sum(CASE WHEN period = 'error' THEN turnover ELSE 0 END)                 AS trade_at_last_volume
                 FROM data.vols_phases
                 WHERE date BETWEEN '{start}' and '{end}'
                   AND symbol IN {tuple(stocks)}
                 GROUP BY symbol, date),
     atc_orders AS (SELECT date,
                           symbol,
                           bids AS preclose_atc_buys,
                           asks AS preclose_atc_sells
                    FROM close.auction_changes
                    WHERE lag = 0
                      AND price = 0
                      AND symbol IN {stocks} AND date BETWEEN '{start}' AND'{end}')
SELECT s.*,
       p.opening_price,
       p.first_price,
       p.last_price,
       cf.preclose_midquote,
       p.close_price  AS effective_close,
       opening_volume,
       continuous_volume,
       closing_volume as closing_volume,
       intraday_auction_volume,
       trade_at_last_volume,
       atc.preclose_atc_buys,
       atc.preclose_atc_sells,
       cf.market_buy_exec,
       cf.market_sell_exec,
       cf.market_buy_not_exec,
       cf.market_sell_not_exec,
       cf.limit_buy_exec,
       cf.limit_sell_exec,
       cf.limit_buy_not_exec,
       cf.limit_sell_not_exec
FROM close.sensitivity s
         LEFT JOIN volumes v
                   ON s.symbol = v.symbol AND s.date = v.date
         LEFT JOIN main.prices p ON p.date = s.date AND p.symbol = s.symbol
         LEFT JOIN close.crossed_final_new cf ON s.date = cf.date AND s.symbol = cf.symbol
         LEFT JOIN atc_orders AS atc ON atc.date = s.date AND atc.symbol = s.symbol

WHERE s.date BETWEEN '{start}' AND '{end}'
  AND s.date != '2020-07-01'
  AND s.symbol IN {tuple(stocks)}
ORDER BY s.symbol, s.date, s.removal

        """

        imp = pd.read_sql(sql, self.connection, parse_dates=['date'], index_col=['symbol', 'date', 'removal'])
        calculate_deviations_bps(imp)
        imp['size_quantile'] = extract_quantiles(imp['closing_volume'], num_quants=self.QUANTILE_NUM, groups='date')
        imp = imp.loc[imp['market_buy_not_exec'] + imp['market_sell_not_exec'] == 0, :]
        imp.to_hdf(f"{self.DATADIR}\\{filename}", key='df', mode='w')
        return imp

    def import_master_local(self) -> Union[pd.DataFrame, object]:
        return pd.read_hdf(f"{self.DATADIR}\\master_import", key='df')

    def import_refined_data(self) -> Union[pd.DataFrame, object]:
        return pd.read_hdf(f"{self.DATADIR}\\refined_data", key='df')

    def set_stock_universe(self, start: str, end: str, top_stocks: int, minimum_days: int) -> tuple:
        sql = f"""
SELECT symbol, date, turnover
FROM data.vols_phases
WHERE period = '03_Closing Auction'
  AND date BETWEEN '{start}' AND '{end}'
ORDER BY symbol, date"""
        volumes = pd.read_sql(sql, self.connection, index_col=['symbol', 'date'], parse_dates=['date'])

        selection = volumes.groupby('symbol').agg([np.mean, np.count_nonzero]).xs('turnover', axis=1)
        selection = selection[selection['count_nonzero'] >= minimum_days].sort_values('mean', ascending=False)
        stocks = selection.reset_index()['symbol'].iloc[:top_stocks]
        return tuple(stocks)

    def manipulate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        def calculate_return(start: pd.Series, end: pd.Series, roll=-1) -> pd.Series:
            def shift_forward(values, shift=roll):
                if shift == 0:
                    return values
                else:
                    output = np.roll(values, shift)
                    output[shift:] = np.nan
                    return output

            shifted = end.groupby('symbol').transform(lambda x: shift_forward(x.to_numpy(), roll))
            return (np.log(shifted / start)) * 100

        filename = 'refined_data'
        removal_choice = 1.0

        df = data.xs(removal_choice, level='removal', drop_level=True).copy()
        # df['closing_volume_share'] = df['closing_volume'] / (df.filter(like='_volume')).sum(axis=1)

        df['preclose_to_close'] = calculate_return(df['preclose_midquote'], df['calculated_close'], roll=0)
        df['open_to_preclose'] = calculate_return(df['opening_price'], df['preclose_midquote'], roll=0)
        df['open_to_close'] = calculate_return(df['opening_price'], df['calculated_close'], roll=0)
        # df['preclose_to_open+1'] = calculate_return(df['preclose_midquote'], df['opening_price'])
        # df['preclose_to_close+1'] = calculate_return(df['preclose_midquote'], df['preclose_midquote'])
        df['close_to_open+1'] = calculate_return(df['calculated_close'], df['opening_price'])
        # df['close_to_preclose+1'] = calculate_return(df['calculated_close'], df['preclose_midquote'])
        df['close_to_close+1'] = calculate_return(df['calculated_close'], df['calculated_close'])
        df['close_to_open+2'] = calculate_return(df['calculated_close'], df['opening_price'], roll=-2)
        df['close_to_close+2'] = calculate_return(df['calculated_close'], df['calculated_close'], roll=-2)

        # df['WPDC_forward'] = calculate_WPDC_measures(df, 'preclose_to_close', 'preclose_to_close+1')
        df['WPDC_backward'] = calculate_WPDC_measures(df, 'preclose_to_close', 'open_to_close')

        df['market_buy_ratio'] = df['market_buy_exec'] / (df['market_buy_exec'] + df['limit_buy_exec'])
        df['market_sell_ratio'] = df['market_sell_exec'] / (df['market_sell_exec'] + df['limit_sell_exec'])
        df['market_ratios_diff'] = (df['market_buy_ratio'] - df['market_sell_ratio'])
        df['market_ratio'] = df[['market_sell_exec', 'market_buy_exec']].sum(1) / df[
            ['market_sell_exec', 'market_buy_exec', 'limit_buy_exec', 'limit_sell_exec']].sum(1)
        df['market_imbalance'] = calculate_imbalance(df['market_buy_exec'].to_numpy(), df['market_sell_exec'].to_numpy())
        df['abs_market_imbalance'] = np.abs(df['market_imbalance'])
        df['market_imbalance_squared'] = np.square(df['market_imbalance'])
        df['preclose_imbalance'] = calculate_imbalance(df['preclose_atc_buys'].to_numpy(), df['preclose_atc_sells'].to_numpy())
        add_expiry_date_columns(df)

        # df = df.loc[df.filter(like='close_to_').abs().max(axis=1) < 10, :]
        df.to_hdf(f"{self.DATADIR}\\{filename}", key='df', mode='w')
        return df
