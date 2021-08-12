import numpy as np
from collections import deque

from funcs_base import *

@nb.njit
def remove_orders(frame: pd.DataFrame, mode: str, side: str, perc: int = 0) -> dict:
    """
    This function removes a certain percentage of liquidity from the closing auction.
    It is called for a every date-title combination individually
    :param date: Onbook_date
    :param symbol: Name of the stock
    :param perc: Values in decimals, i.e. 5% is handed in as 0.05
    :param side: ['bid','ask','all']. Which side should be included in the removal
    :param market: True if market orders are included and False otherwise
    :return: A dateframe with new bid-ask book based on removing adjustments.
    """
    empty_output = dict(asks=pd.Series(), bids=pd.Series())

    if perc == 0:
        return dict(asks=frame['asks'], bids=frame['bids'])

    bids = frame['bids'].to_numpy().copy()
    asks = frame['asks'].to_numpy().copy()

    if mode == 'market':
        if frame.loc[0, :].sum() == 0:
            return empty_output  # Only considering limit orders for adjustments
        else:
            rem_bid = bids[0] * perc
            rem_ask = asks[0] * perc

    elif mode == 'liquidity':
        rem_bid = min((sum(bids) + sum(asks)) * perc / 2, sum(bids))
        rem_ask = min((sum(bids) + sum(asks)) * perc / 2, sum(asks))
    elif mode == 'execution':
        close_volume = calculate_uncross(bids=frame['bids'], asks=frame['asks'])['trade_vol']
        rem_bid = close_volume * perc
        rem_ask = close_volume * perc

    else:
        raise KeyError(f"Incorrect value for mode submitted : {mode}")

    if side in ['bid', 'both']:
        b = len(bids) - 1
        while rem_bid > 0:
            if bids[0] != 0:
                local_vol = bids[0]
                bids[0] = local_vol - min(local_vol, rem_bid)
                rem_bid -= min(rem_bid, local_vol)
            else:
                local_vol = bids[b]
                bids[b] = local_vol - min(local_vol, rem_bid)
                rem_bid -= min(rem_bid, local_vol)
                b -= 1

    if side in ['ask', 'both']:
        a = 1
        while rem_ask > 0:
            if asks[0] != 0:
                local_vol = asks[0]
                asks[0] = local_vol - min(local_vol, rem_ask)
                rem_ask -= min(rem_ask, local_vol)
            else:
                local_vol = asks[a]
                asks[a] = local_vol - min(local_vol, rem_ask)
                rem_ask -= min(rem_ask, local_vol)
                a += 1

    ret_df = pd.DataFrame([asks, bids], index=['end_close_vol_ask', 'end_close_vol_bid'], columns=frame.index).T
    return dict(asks=ret_df['end_close_vol_ask'], bids=ret_df['end_close_vol_bid'])

def calculate_uncross(bids: pd.Series, asks: pd.Series, exportFull: bool = None) -> dict:
    """
    Function calculates the theoretical uncross price of a closing order book.
    :return: dict() with price/trade_vol/cum_bids/cum_asks/total_bids/total_asks
    """
    output = dict(price=np.nan, trade_vol=np.nan, cum_bids=np.nan, cum_asks=np.nan, total_bids=np.nan, total_asks=np.nan)
    base_df = pd.DataFrame({'bids': bids, 'asks': asks})
    df, mark_buy, mark_sell = extract_market_orders(base_df)
    prices = df.index.to_numpy()
    limit_orders = df.to_numpy()

    if np.min(np.sum(limit_orders[:, -2:], axis=0)) == 0:
        return output

    else:
        cumul = np.zeros((limit_orders.shape[0], 2))
        cumul[:, 0] = np.flip(np.cumsum(np.flip(limit_orders[:, -2]))) + mark_buy
        cumul[:, 1] = np.cumsum(limit_orders[:, -1]) + mark_sell

        if np.max(cumul[:, 0] * cumul[:, 1]) == 0:  # No overlap
            return output

        volumes = np.minimum(cumul[:, 0], cumul[:, 1])
        imbalances = np.abs(cumul[:, 0] - cumul[:, 1])

        maxvol_indices = np.flatnonzero(volumes == volumes.max())
        minimb_index = np.argmin(imbalances[maxvol_indices])
        optimum = maxvol_indices.min() + minimb_index

        price = prices[optimum]
        trade_vol = np.min(cumul[optimum, :])

        sum_bids, sum_asks = cumul[:, 0].max(), cumul[:, 1].max()

        if min(cumul[optimum, 0], cumul[optimum, 1]) == 0:
            output = output
        else:
            output = dict(price=price, trade_vol=trade_vol, cum_bids=cumul[optimum, 0], cum_asks=cumul[optimum, 1],
                          total_bids=sum_bids, total_asks=sum_asks)
        return output

def calculate_preclose_midquote(bids: pd.Series, asks: pd.Series) -> dict:
        """
        This helper function calculates the hypothetical last midquote before closing auctions start.
        This method takes only inputs from the self._remove_liq method.
        """
        base_df = pd.DataFrame({'bids': bids, 'asks': asks})
        try:
            base_df.drop(index=[0], inplace=True)
        except KeyError:
            pass

        n_lim = base_df.shape[0]
        limit_bids, limit_asks = deque(base_df['bids'], n_lim), deque(base_df['asks'], n_lim)
        neg_asks = limit_asks.copy()
        neg_asks.appendleft(0)

        cum_bids = np.cumsum(limit_bids)
        cum_asks = sum(limit_asks) - np.cumsum(neg_asks)

        total = cum_bids + cum_asks

        i_top_bid = np.argmax(total)
        i_top_ask = len(total) - np.argmax(np.flip(total)) - 1
        maxbid, minask = base_df['bids'].index[i_top_bid], base_df['asks'].index[i_top_ask]

        if i_top_bid > i_top_ask:
            raise ValueError("i_top_bid not smaller than i_top_ask (spread overlap)")

        else:
            output = dict(abs_spread=round(minask - maxbid, 4), midquote=round((maxbid + minask) / 2, 4),
                          rel_spread=round((minask - maxbid) / ((maxbid + minask) / 2) * 10 ** 4, 4))
            return output


def extract_market_orders(imp_df: pd.DataFrame) -> tuple:
    """
    Removes market orders from an order book snapshot.
    :param imp_df: Pandas DataFrame
    :return: Pandas DataFrame without the market orders
    """
    try:
        mark_buy = imp_df.loc[0, 'bids']
    except KeyError:
        imp_df.loc[0, :] = 0
        mark_buy = imp_df.loc[0, 'bids']

    try:
        mark_sell = imp_df.loc[0, 'asks']
    except KeyError:
        imp_df.loc[0, :] = 0
        mark_sell = imp_df.loc[0, 'asks']

    df = imp_df.drop(0, axis=0).sort_index()

    return df, mark_buy, mark_sell


# @nb.njit
# def process_uncrossing_numba(group_indices: nb.typed.List, data: np.ndarray, removals: np.ndarray = None) -> tuple:
#     """
#     Function calculates the theoretical uncross price of a closing order book.
#     :return: dict() with price/trade_vol/cum_bids/cum_asks/total_bids/total_asks
#     """
#     preclose: float
#     # names = ('price', 'volume',
#     #          'market_buy_exec', 'market_sell_exec', 'limit_buy_exec', 'limit_sell_exec',
#     #          'market_buy_not_exec', 'market_sell_not_exec', 'limit_buy_not_exec', 'limit_sell_not_exec')
#     collector = np.full((len(group_indices)), np.nan)
#
#     for group, counter in zip(group_indices, np.arange(0, len(group_indices))):
#         uncrossing = calculate_uncrossing(values=data[group])
#
#         for remove in removals:
#             for mode in ('bid', 'ask'):
#                 remove_executed_volume(data[group], remove=remove, mode=mode, executed_vol=uncrossing[1])
#                 # remove_average_volume()
#                 # remove_market_orders()
#
#     # for group, counter in zip(group_indices, np.arange(0, len(group_indices))):
#     #     if data[group[0], 0] == -10:
#     #         result = calculate_midquote(group=data[group], names=names)
#     #     else:
#     #         result = calculate_uncrossing(values=data[group], names=names)
#     #
#     #     collector[counter, :] = result
#     #
#     return collector, names
#
#
# @nb.njit
# def remove_executed_volume(values: np.ndarray, remove: np.float64, mode: str, executed_vol: float) -> tuple:
#     rem_bid = executed_vol * remove
#     rem_ask = executed_vol * remove
#     bids = values[:, -2].copy()
#     asks = values[:, -1].copy()
#
#     if mode == 'bid':
#         b = len(bids) - 1
#         while rem_bid > 0:
#             if bids[0] != 0:
#                 local_vol = bids[0]
#                 bids[0] = local_vol - min(local_vol, rem_bid)
#                 rem_bid -= min(rem_bid, local_vol)
#             else:
#                 local_vol = bids[b]
#                 bids[b] = local_vol - min(local_vol, rem_bid)
#                 rem_bid -= min(rem_bid, local_vol)
#                 b -= 1
#
#     elif mode == 'ask':
#         a = 1
#         while rem_ask > 0:
#             if asks[0] != 0:
#                 local_vol = asks[0]
#                 asks[0] = local_vol - min(local_vol, rem_ask)
#                 rem_ask -= min(rem_ask, local_vol)
#             else:
#                 local_vol = asks[a]
#                 asks[a] = local_vol - min(local_vol, rem_ask)
#                 rem_ask -= min(rem_ask, local_vol)
#                 a += 1
#
#
#     else:
#         pass
#
#     return tuple()
#
#
# @nb.njit
# def remove_average_volume() -> tuple:
#     return tuple()
#
#
# @nb.njit
# def remove_market_orders() -> tuple:
#     return tuple()
#
#
# @nb.njit
# def calculate_uncrossing(values: np.ndarray) -> np.ndarray:
#     output = np.full(10, np.nan)
#
#     if values[0, 0] == 0:
#         market_orders = values[0, -2:]
#         limit_orders = values[1:, :]
#     else:
#         market_orders = np.zeros(2)
#         limit_orders = values[0:, :]
#
#     if np.min(np.sum(limit_orders[:, -2:], axis=0)) == 0:
#         return output
#
#     cumul = np.zeros((limit_orders.shape[0], 2))
#     cumul[:, 0] = np.flip(np.cumsum(np.flip(limit_orders[:, -2]))) + market_orders[0]
#     cumul[:, 1] = np.cumsum(limit_orders[:, -1]) + market_orders[1]
#
#     if np.max(cumul[:, 0] * cumul[:, 1]) == 0:  # No overlap
#         return output
#
#     volumes = np.minimum(cumul[:, 0], cumul[:, 1])
#     imbalances = np.abs(cumul[:, 0] - cumul[:, 1])
#
#     maxvol_indices = np.flatnonzero(volumes == volumes.max())
#     minimb_index = np.argmin(imbalances[maxvol_indices])
#     optimum = maxvol_indices.min() + minimb_index
#
#     price = limit_orders[optimum, 1]
#     trade_vol = np.min(cumul[optimum, :])
#
#     output[-10] = price
#     output[-9] = trade_vol
#
#     output[-8] = min(market_orders[0], trade_vol)  # MBE
#     output[-7] = min(market_orders[1], trade_vol)  # MSE
#     output[-6] = max(0, trade_vol - market_orders[0])  # LBE
#     output[-5] = max(0, trade_vol - market_orders[1])  # LSE
#
#     output[-4] = max(market_orders[0] - trade_vol, 0)  # MBNE
#     output[-3] = max(market_orders[1] - trade_vol, 0)  # MSNE
#     output[-2] = max(0, cumul[optimum, -2] - output[-4])  # LBNE
#     output[-1] = max(0, cumul[optimum, -1] - output[-5])  # LSNE
#
#     return output
#
#
# @nb.njit
# def calculate_midquote(group: np.ndarray, names: tuple) -> np.ndarray:
#     """
#     This helper function calculates the hypothetical last midquote before closing auctions start.
#     This method takes only inputs from the self._remove_liq method.
#     """
#     output = np.full(len(names), np.nan)
#
#     if group[0, 1] == 0:
#         limit_orders = group[1:, :]
#     else:
#         limit_orders = group[0:, :]
#
#     if np.max(np.sum(limit_orders, axis=0)) == 0:  # Empty order book
#         return output
#
#     cumul = np.zeros((limit_orders.shape[0], 2))
#     cumul[:, 0] = np.flip(np.cumsum(np.flip(limit_orders[:, -2])))
#     cumul[:, 1] = -np.cumsum(limit_orders[:, -1])
#
#     total = np.sum(cumul, axis=1)
#     optimum = np.argmin(np.abs(total))
#     if total[optimum] > 0:
#         midquote = (limit_orders[optimum, 1] + limit_orders[optimum + 1, 1]) / 2
#     else:
#         midquote = (limit_orders[optimum - 1, 1] + limit_orders[optimum, 1]) / 2
#     output[-10] = midquote
#     return output
