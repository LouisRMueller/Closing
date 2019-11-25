import pandas as pd
import numpy as np
import copy
import time
from collections import deque
from Class_Research import Research


class PriceDiscovery(Research):
	def __init__(self, file_snapshots, file_close_prices):
		super().__init__(file_snapshots)
		self._closeprices = pd.read_csv(file_close_prices, header=0)
		self._closeprices.set_index(['onbook_date', 'symbol'], drop=True, inplace=True)
		self._closeprices.sort_index(inplace=True)
	
	def _calc_uncross(self, date, title):
		"""
		Function calculates the theoretical uncross price of a closing order book.
		:param date:
		:param title:
		:return: dict()
		"""
		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)
		
		df, mark_buy, mark_sell = Research._extract_market_orders(imp_df)
		
		if 0 in df[['close_vol_ask', 'close_vol_bid']].sum().values:
			return dict(price=np.nan, imbalance=np.nan, trade_vol=np.nan)
		
		else:
			n_lim = df.shape[0]
			limit_bids, limit_asks = deque(df['close_vol_bid'], n_lim), deque(df['close_vol_ask'], n_lim)
			
			neg_asks = limit_asks.copy()
			neg_asks.appendleft(0)
			
			cum_bids = mark_buy + np.cumsum(limit_bids)
			cum_asks = mark_sell + sum(limit_asks) - np.cumsum(neg_asks)
			
			total = cum_bids + cum_asks
			i = np.argmax(abs(total))
			
			output = dict(price=df.index[i], imbalance=total[i], trade_vol=min(cum_bids[i], cum_asks[i]))
			return output
	
	def _calc_preclose_price(self, date, title):
		"""
		This helper function calculates the hypothetical last midquote before closing auctions start.
		This method takes only inputs from the self._remove_liq method.
		"""
		
		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)
		
		try:
			imp_df.drop(index=[0], inplace=True)
		except KeyError:
			pass
		
		n_lim = imp_df.shape[0]
		limit_bids, limit_asks = deque(imp_df['cont_vol_bid'], n_lim), deque(imp_df['cont_vol_ask'], n_lim)
		neg_asks = limit_asks.copy()
		neg_asks.appendleft(0)
		
		cum_bids = np.cumsum(limit_bids)
		cum_asks = sum(limit_asks) - np.cumsum(neg_asks)
		
		total = cum_bids + cum_asks
		
		i_top_bid = np.argmax(total)
		i_top_ask = len(total) - np.argmax(np.flip(total)) - 1
		maxbid, minask = imp_df['cont_vol_bid'].index[i_top_bid], imp_df['cont_vol_ask'].index[i_top_ask]
		
		if i_top_bid > i_top_ask:
			raise ValueError("i_top_bid not smaller than i_top_ask (spread overlap)")
		
		else:
			if (maxbid + minask) == 0:
				print("Symbol: {}".format(title))
			output = dict(abs_spread=round(minask - maxbid, 4),
			              midquote=round((maxbid + minask) / 2, 4),
			              rel_spread=round((minask - maxbid) / ((maxbid + minask) / 2) * 10 ** 4, 4))
			return output
	
	def discovery_analysis(self):
		"""
		This function is supposed to exeucte the required calculations and add it to an appropriate data format.
		It calls other helper functions in order to determine the results of the analysis.
		"""
		
		for date in self._dates:
			t0 = time.time()
			current_symbols = self.get_SB().loc[date, :].index.get_level_values(0).unique()
			
			for symbol in current_symbols:
				# for symbol in ['ALLN']:
				close_uncross = self._calc_uncross(date, symbol)
				preclose_uncross = self._calc_preclose_price(date, symbol)
				
				res = self._result_dict[date, symbol] = {}
				
				res['pre_abs_spread'] = preclose_uncross['abs_spread']
				res['pre_midquote'] = preclose_uncross['midquote']
				res['pre_rel_spread'] = preclose_uncross['rel_spread']
				
				res['close_price_calculated'] = close_uncross['price']
				res['close_vol'] = close_uncross['trade_vol']
				res['close_imbalance'] = close_uncross['imbalance']
				
				try:
					res['actual_close_price'] = self._closeprices.loc[(date, symbol), 'price_org_ccy']
				except KeyError:
					res['actual_close_price'] = np.nan
			# print("--- {} completed".format(symbol))
			
			print(">> {} finished ({} seconds)".format(date, round(time.time() - t0, 2)))
		
