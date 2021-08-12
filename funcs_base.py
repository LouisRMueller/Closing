import pandas as pd
import numpy as np
import os
import time
import cProfile, pstats

from typing import Optional, Sequence, Union
import numba as nb


def profiler(func):
    with cProfile.Profile() as pr:
        func()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    # stats.dump_stats(filename='profiling.prof')


@nb.njit
def calculate_auction_reversals(values: np.ndarray, index=None) -> float:
    copy = np.copy(values)
    copy[np.isnan(values)] = 0
    idx = np.argmax(np.abs(copy))
    return values[-1] - values[idx]


def check_column_names(frame: pd.DataFrame, variable_names: Sequence[str]) -> None:
    if not all(var_check := tuple(map(lambda x: x in frame.columns, variable_names))):
        raise Exception(f"Unknown column names submitted: {variable_names[var_check.index(False)]}")


def data_to_value_string(data):
    data['date'] = data['date'].astype(str)
    data.fillna('NULL', inplace=True)
    valueList = [tuple(x) for x in data.to_numpy()]
    return str(valueList).replace("'NULL'", "NULL")[1:-1]


def add_market_orders_if_none(frame: pd.DataFrame, level: str = 'price') -> pd.DataFrame:
    prices = frame.index.to_frame()[level].to_numpy()
    if prices.min() == 0:
        return frame
    else:
        frame.loc[0, :] = 0
        return frame.sort_index()
