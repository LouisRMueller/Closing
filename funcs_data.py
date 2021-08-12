from funcs_base import *


def add_absolute_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    check_column_names(frame, columns)
    new_cols = list(map(lambda x: f"abs_{x}", columns))
    frame[new_cols] = np.abs(frame[columns])
    return new_cols


def create_aggregated_overnight_data(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_return(close: pd.Series, other: pd.Series, roll=-1) -> pd.Series:
        # @nb.njit
        def shift_forward(values, roll=roll):
            output = np.roll(values, roll)
            output[roll:] = np.nan
            return output

        shifted = other.groupby('symbol').transform(lambda x: shift_forward(x.to_numpy(), roll))
        return np.log(shifted / close) * 100

    # grp = frame.groupby(['symbol', 'date'])
    # relevant = ['opening_price', 'preclose_midquote', 'close_price', 'log_volume', 'initial_ret', 'current_volume']
    # df = grp[relevant].aggregate(lambda values, index: values[-1], engine='numba')
    #
    # df['cumul_max'] = grp['cumul_rets'].aggregate(lambda values, index: values[np.argmax(np.abs(values))], engine='numba')
    # df['max_CPDC'] = grp['CPDC'].aggregate(lambda values, index: values[np.argmax(np.abs(values))], engine='numba')
    # df['reversion'] = grp['cumul_rets'].aggregate(calculate_auction_reversals, engine='numba')
    # add_absolute_columns(df, ['max_CPDC', 'reversion', 'cumul_max'])
    #
    # df['max_cumul_secs'] = grp['cumul_rets'].aggregate(lambda values, index: np.argmax(np.abs(values)) * 10, engine='numba')
    # df['std_ival_rets'] = grp['ival_rets'].aggregate(np.nanstd)
    # df['log_close_volume'] = np.log(df['current_volume'])
    starting_point = 'close_price'
    df['close_return'] = np.log(df['close_price'] / df['preclose_midquote']) * 100
    df['ret_to_open'] = calculate_return(df[starting_point], df['opening_price'])
    df['ret_to_preclose'] = calculate_return(df[starting_point], df['preclose_midquote'])
    df['ret_to_close'] = calculate_return(df[starting_point], df['close_price'])
    df['ret_to_open_2'] = calculate_return(df[starting_point], df['opening_price'], roll=-2)

    return df
