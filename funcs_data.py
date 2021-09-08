import itertools
import numpy as np
import pandas as pd

from funcs_base import *


def add_absolute_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    check_column_names(frame, columns)
    new_cols = list(map(lambda x: f"abs_{x}", columns))
    frame[new_cols] = np.abs(frame[columns])
    return new_cols


def calculate_deviations_bps(frame: pd.DataFrame) -> None:
    names = [f"{x}_{y}" for x, y in itertools.product(['execution', 'liquidity', 'market'], ['bid', 'ask', 'both'])]
    new_names = list(map(lambda x: f"{x}_dev", names))
    frame['calculated_close'] = frame.groupby(['symbol', 'date'])['execution_both'].transform(lambda values: values[0])

    actuals, others = frame['calculated_close'].to_numpy(), frame[names].to_numpy()
    frame[new_names] = np.true_divide(others - actuals[:, None], actuals[:, None]) * 10 ** 4


def extract_quantiles(series: pd.Series, num_quants: int, groups: Union[list, str] = 'date') -> pd.Series:
    def binning_quantiles(values: np.ndarray, num_quants: int) -> np.ndarray:
        bins = np.nanquantile(values, np.linspace(0, 1, num_quants + 1)[1:])
        quants = np.digitize(values, bins, right=True) + 1.0
        quants[np.isnan(values)] = np.nan
        return quants.astype(int)

    series = series.fillna(0)
    return series.groupby(groups).transform(lambda x: binning_quantiles(x, num_quants))


@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def calculate_imbalance(positive, negative) -> float:
    imbalance = np.nansum(np.array([np.abs(positive), - np.abs(negative)]))
    divisor = np.nansum(np.array([np.abs(positive), np.abs(negative)]))
    if divisor != 0:
        return imbalance / divisor
    else:
        return 0


def add_expiry_date_columns(frame: pd.DataFrame) -> None:
    dates = frame.reset_index()['date']
    weekday = dates.dt.weekday.to_numpy()
    day = dates.dt.day.to_numpy()
    month = dates.dt.month.to_numpy()

    frame['weekly'] = np.where(weekday == 4, 1, 0)
    frame['monthly'] = np.where((weekday == 4) & (15 <= day) & (day <= 21), 1, 0)
    frame['quarterly'] = np.where((weekday == 4) & (15 <= day) & (day <= 21) & np.isin(month, [3, 6, 9, 12]), 1, 0)

def calculate_WPDC_measures(frame: pd.DataFrame, inner_name: str, outer_name: str) -> pd.Series:
    inner_returns, outer_returns = frame[inner_name], frame[outer_name]
    PDC = np.where(outer_returns == 0, np.nan, inner_returns / outer_returns)
    weights = outer_returns.abs() / outer_returns.groupby(['date']).transform(lambda x: np.sum(np.abs(x)))
    return pd.Series(weights * PDC)

def add_lagged_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    check_column_names(frame, columns)

    new_cols = list(map(lambda x: f"{x}_lag", columns))

    grouper = frame[columns].groupby(['symbol', 'date'])
    frame[new_cols] = grouper.transform(vectorized_shift, engine='numba')
    return new_cols

def add_pos_neg_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    if not all(c in frame.columns for c in columns):
        raise Exception(f"Unknown input columns: {columns}")

    for c in columns:
        values = frame.loc[:, c]
        frame[f'{c}_pos'] = np.where(values > 0, values, 0)
        frame[f'{c}_neg'] = np.where(values < 0, values, 0)


def print_stockwise_stats(df: pd.DataFrame):
    relevant_cols = ['continuous_volume','closing_volume','preclose_to_close','size_quantile']
    grp = df[relevant_cols].groupby('symbol')

    table = pd.DataFrame({'Stockdays': grp['continuous_volume'].count().astype(int),
                          'Avg. closing volume': grp['closing_volume'].agg(np.nanmean) / 10**6,
                          'Std. closing volume': grp['closing_volume'].agg(np.nanstd) / 10**6,
                          'Avg. continuous volume': grp['continuous_volume'].agg(np.nanmean) / 10**6,
                          'Std. continuous volume': grp['continuous_volume'].agg(np.nanstd) / 10**6,
                          'Avg. closing return': grp['preclose_to_close'].agg(np.nanmean),
                          'Std. closing return': grp['preclose_to_close'].agg(np.nanstd)
                          })
    for q in sorted(df['size_quantile'].unique()):
        table[f"$Q_{q}$"] = grp['size_quantile'].agg(lambda x: np.count_nonzero(x == q)).astype(int)

    table = table.reset_index().reset_index()
    table.iloc[:,0] = table.iloc[:,0] +1
    print(table.round(2).to_latex(index=False, longtable=True, index_names=False))

