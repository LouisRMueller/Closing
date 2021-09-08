import pandas as pd
import numpy as np
import os
import time
import functools
import cProfile, pstats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tick

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


def add_market_orders_if_none(frame: pd.DataFrame, level: str = 'price') -> np.ndarray:
    prices = frame.index.to_frame()[level].to_numpy()
    if prices.min() == 0:
        return frame.reset_index().to_numpy()
    else:
        frame.loc[0, :] = 0
        return frame.sort_index().reset_index().to_numpy()

def prepare_data_for_latex_plot(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=False)
    frame = frame.replace('_', '\_', regex=True)
    frame.rename(columns=lambda x: x.replace('_', '-'), inplace=True)
    return frame

def draw_gridlines(ax: plt.axis, axis: str = 'y') -> None:
    ax.set_axisbelow(True)
    ax.grid('major', axis=axis)

def show_and_save_graph(filename: str) -> None:
    plt.savefig(f"{os.getcwd()}\\Figures\\{filename}.pdf")
    plt.show()


@nb.njit
def numpy_shift(values: np.ndarray, num: int = 1, fill_value=np.nan) -> np.ndarray:
    result = np.full_like(values, fill_value=fill_value)
    if num > 0:
        result[num:] = values[:-num]
    elif num < 0:
        result[:num] = values[-num:]
    else:
        result[:] = values

    return result

@nb.guvectorize(["(float64[:], int32, float64[:])"], "(n),() -> (n)")
def vectorized_shift(values, num, result):
    if num > 0:
        result[num:] = values[:-num]
        result[:num] = np.nan
    elif num < 0:
        result[:num] = values[-num:]
        result[num:] = np.nan
    else:
        result[:] = values
