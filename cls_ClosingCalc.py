import pandas as pd
import numpy as np
from time import time
import os
import copy
from collections import deque


class Research:
	def __init__(self, file_snapshots):
		t0 = time()
		self._snapbook = pd.read_csv(file_snapshots, header=0)
		self._symbols = self._snapbook['symbol'].unique()
		self._dates = self._snapbook['onbook_date'].unique()
		self._snapbook.set_index(['onbook_date', 'symbol', 'price'], drop=True, inplace=True)
		self._snapbook.sort_index(inplace=True)
		
		self._result_dict = {}  # Collects all the results
		
		print(">>> Class initiated ({} seconds)".format(round(time() - t0, 2)))
		
	@staticmethod
	def _extract_market_orders(imp_df):
		"""
		Removes market orders from an order book snapshot.
		:param imp_df: Pandas DataFrame
		:return: Pandas DataFrame without the market orders
		"""
		try:
			mark_buy = imp_df.loc[0, 'close_vol_bid']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_buy = imp_df.loc[0, 'close_vol_bid']
		except TypeError:
			print(imp_df.head())
			raise TypeError('FAILO')
		try:
			mark_sell = imp_df.loc[0, 'close_vol_ask']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_sell = imp_df.loc[0, 'close_vol_ask']
		
		df = imp_df.drop(0, axis=0).sort_index()
		
		return df, mark_buy, mark_sell
	
	def get_SB(self):
		return self._snapbook
	
	def export_results(self, filename, filetype):
		df = self.results_to_df()
		if filetype == 'xlsx':
			df.round(4).to_excel(os.getcwd() + "\\Exports\\{}.xlsx".format(filename))
		elif filetype == 'csv':
			df.round(4).to_csv(os.getcwd() + "\\Exports\\{}.csv".format(filename))


class SensitivityAnalysis(Research):
	@staticmethod
	def _calc_uncross(imp_df):
		"""
		Function calculates the theoretical uncross price of a closing order book.
		:return: dict()
		"""
		df, mark_buy, mark_sell = Research._extract_market_orders(imp_df)
		
		if 0 in df[['close_vol_ask', 'close_vol_bid']].sum().values:  # Where one side is empty
			return dict(price=np.nan, imbalance=np.nan, trade_vol=np.nan)
		
		else:
			n_lim = df.shape[0]
			limit_bids, limit_asks = deque(df['close_vol_bid'], n_lim), deque(df['close_vol_ask'], n_lim)
			
			neg_bids = limit_bids.copy()
			neg_bids.appendleft(0)
			
			cum_bids = mark_buy + sum(limit_bids) - np.cumsum(neg_bids)
			cum_asks = mark_sell + np.cumsum(limit_asks)
			
			OIB = cum_bids - cum_asks
			i = np.argmin(abs(OIB))
			trade_vol = min(cum_bids[i], cum_asks[i])

			dump_df = pd.DataFrame({'price': df.index, 'cum. bids': cum_bids,
							    'cum. asks': cum_asks, 'OIB': OIB, 'vol': trade_vol})
			dump_df.set_index('price', inplace=True)

			output = dict(price=df.index[i], imbalance=OIB[i], trade_vol=trade_vol, dump=dump_df)

			return output
	
	def _remove_liq(self, date, title, percentage=0, market=None, side=None):
		"""
		This function removes a certain percentage of liquidity from the closing auction.
		It is called for a every date-title combination individually
		:param date: Onbook_date
		:param title: Name of the stock
		:param percentage: Values in decimals, i.e. 5% is handed in as 0.05
		:param side: ['bid','ask','all']. Which side should be included in the removal
		:param market: True if market orders are included and False otherwise
		:return: A dateframe with new bid-ask book based on removing adjustments.
		"""
		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)
		
		if percentage == 0:
			return imp_df[['close_vol_bid', 'close_vol_ask']]
		
		bids = imp_df['close_vol_bid'].tolist()
		asks = imp_df['close_vol_ask'].tolist()
		
		if market == "remove_all":  # Removes all market orders in closing auction
			ret_df = imp_df.loc[:, ('close_vol_ask', 'close_vol_bid')]
			try:
				ret_df.loc[0, :] = 0
				return ret_df
			except KeyError:
				return ret_df
		
		else:  # Only considering limit orders for adjustments
			removable_bid = sum(bids[1:]) * percentage
			removable_ask = sum(asks[1:]) * percentage
			imp_df.drop(columns=['cont_vol_bid', 'cont_vol_ask'], inplace=True)
		
		# Below is the algorithm
		if side in ['bid', 'all']:
			b = len(bids) - 1
			# remaining_liq = removable_bid / percentage
			while removable_bid > 0:
				local_vol = bids[b]
				bids[b] = local_vol - min(local_vol, removable_bid)
				removable_bid -= min(removable_bid, local_vol)
				b -= 1
		
		if side in ['ask', 'all']:
			a = 1
			# remaining_liq = removable_ask / percentage
			while removable_ask > 0:
				local_vol = asks[a]
				asks[a] = local_vol - min(local_vol, removable_ask)
				removable_ask -= min(removable_ask, local_vol)
				a += 1
		
		
		ret_df = pd.DataFrame([asks, bids], index=imp_df.columns, columns=imp_df.index).T
		return ret_df
	
	def sens_analysis(self, key, percents=tuple([1])):
		"""
		This function is supposed to exeucte the required calculations and add it to an appropriate data format.
		It calls other helper functions in order to determine the results of the analysis.
		:param key: String with mode of calculation ['bid_limit','ask_limit','all_limit','all_market','cont_market']
		:param percents: Iterable object with integers of percentages to be looped through.
		"""
		dump = {}
		
		if key == 'bid_limit':
			side, mkt = 'bid', None
		elif key == 'ask_limit':
			side, mkt = 'ask', None
		elif key == 'all_limit':
			side, mkt = 'all', None
		elif key == 'all_market':
			side, mkt = 'all', 'remove_all'
		# elif key == "cont_market":
		# 	side, mkt = 'all', 'remove_cont'
		
		else:
			raise ValueError("key input not in ['bid_limit','ask_limit','all_limit','all_market','cont_market']")
		
		for date in self._dates:
			t0 = time()
			current_symbols = self.get_SB().loc[date, :].index.get_level_values(0).unique()
			dump.update({date:{}})
			
			for symbol in current_symbols:
				close_df = self._remove_liq(date=date, title=symbol, percentage=0)
				close_uncross = self._calc_uncross(close_df)
				dump[date].update({symbol: {}})

				for p in percents:
					res = self._result_dict[key, date, symbol, p] = {}
					
					remove_df = self._remove_liq(date=date, title=symbol, percentage=p, side=side, market=mkt)
					remove_uncross = self._calc_uncross(remove_df)

					dump[date][symbol].update({p: remove_uncross.pop('dump', None)})
					
					res['close_price'] = close_uncross['price']
					res['close_vol'] = close_uncross['trade_vol']
					res['close_imbalance'] = close_uncross['imbalance']
					
					res['adj_price'] = remove_uncross['price']
					res['adj_vol'] = remove_uncross['trade_vol']
					res['adj_imbalance'] = remove_uncross['imbalance']
			
			print(">> {} finished ({} seconds)".format(date, round(time() - t0, 2)))

		return dump if dump else None
	
	def results_to_df(self):
		"""
		Export such that it can be used further in later stages.
		"""
		df = pd.DataFrame.from_dict(self._result_dict, orient='index')
		df.index.set_names(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		return df
	

class PriceDiscovery(Research):
	def __init__(self, file_snapshots, file_close_prices):
		super().__init__(file_snapshots)
		t0 = time()
		self._closeprices = pd.read_csv(file_close_prices, header=0)
		self._closeprices.set_index(['onbook_date', 'symbol'], drop=True, inplace=True)
		self._closeprices.sort_index(inplace=True)

		self._price_discovery_results = {}  # Collects all the results

		print(">>> Class initiated ({} seconds)".format(round(time() - t0, 2)))

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
			
			neg_bids = limit_bids.copy()
			neg_bids.appendleft(0)
			
			cum_bids = mark_buy + sum(limit_bids) - np.cumsum(neg_bids)
			cum_asks = mark_sell + np.cumsum(limit_asks)
			
			OIB = cum_bids - cum_asks
			i = np.argmin(abs(OIB))
			
			output = dict(price=df.index[i], imbalance=OIB[i], trade_vol=min(cum_bids[i], cum_asks[i]))
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
		limit_bids, limit_asks = deque(imp_df['cont_vol_bid'], n_lim) , deque(imp_df['cont_vol_ask'], n_lim)
		neg_asks = limit_asks.copy() ; neg_asks.appendleft(0)

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
		:param key: String with mode of calculation ['bid_limit','ask_limit','all_limit','all_market','cont_market']
		:param percents: Iterable object with integers of percentages to be looped through.
		"""

		for date in self._dates:
			t0 = time()
			current_symbols = self.get_SB().loc[date, :].index.get_level_values(0).unique()

			for symbol in current_symbols:
				close_uncross = self._calc_uncross(date, symbol)
				preclose_uncross = self._calc_preclose_price(date, symbol)

				res = self._price_discovery_results[date, symbol] = {}

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

			print(">> {} finished ({} seconds)".format(date, round(time() - t0, 2)))

	def results_to_df(self):
		"""
		Export such that it can be used further in later stages.
		"""
		df = pd.DataFrame.from_dict(self._price_discovery_results, orient='index')
		df.index.set_names(['Date', 'Symbol'], inplace=True)
		df.sort_index()
		return df

