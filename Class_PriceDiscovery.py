import pandas as pd
import numpy as np
import copy
import time
import os


class PriceDiscovery:
	def __init__(self, file_snapshots, file_close_prices):
		t0 = time.time()
		self._snapbook = pd.read_csv(file_snapshots, header=0)
		self._closeprices = pd.read_csv(file_close_prices, header=0)

		self._symbols = self._snapbook['symbol'].unique()
		self._dates = self._snapbook['onbook_date'].unique()

		self._snapbook.set_index(['onbook_date', 'symbol', 'price'], drop=True, inplace=True)
		self._snapbook.sort_index(inplace=True)
		self._closeprices.set_index(['onbook_date', 'symbol'], drop=True, inplace=True)
		self._closeprices.sort_index(inplace=True)

		self._price_discovery_results = {}  # Collects all the results

		print(">>> Class initiated ({} seconds)".format(round(time.time() - t0, 2)))

	def _calc_uncross(self, date, title):
		"""
		Function calculates the theoretical uncross price of a closing order book.
		:param date:
		:param title:
		:return: dict()
		"""
		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)

		try:
			mark_buy = imp_df.loc[0, 'close_vol_bid']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_buy = imp_df.loc[0, 'close_vol_bid']

		try:
			mark_sell = imp_df.loc[0, 'close_vol_ask']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_sell = imp_df.loc[0, 'close_vol_ask']

		df = imp_df.drop(0, axis=0).sort_index()
		i = round(df.shape[0] / 2)
		close_dict = dict(price=np.nan, imbalance=np.nan, trade_vol=np.nan)

		bid_vol = df.iloc[i:, :]['close_vol_bid'].sum() + mark_buy
		ask_vol = df.iloc[:i + 1, :]['close_vol_ask'].sum() + mark_sell
		OIB = bid_vol - ask_vol
		prev_OIB = OIB.copy()

		if title == 'PARG':
			print('PARG')

		while OIB != 0:
			if abs(OIB) > abs(prev_OIB):  # Normal case: If OIB is actually getting worse
				break
			elif np.sign(OIB) != np.sign(prev_OIB):
				break
			elif max(bid_vol - mark_buy, ask_vol - mark_sell) == 0:  # No quotes available, hence no price
				break
			elif i in [-1, df.shape[0]]:
				break
			else:
				close_dict['price'] = df.index[i]
				close_dict['imbalance'] = OIB
				close_dict['trade_vol'] = min(bid_vol, ask_vol)
				i = i + 1 if OIB > 0 else i - 1

				prev_OIB = OIB.copy()
				bid_vol = df.iloc[i:, :]['close_vol_bid'].sum() + mark_buy
				ask_vol = df.iloc[:i + 1, :]['close_vol_ask'].sum() + mark_sell
				OIB = bid_vol - ask_vol

		return close_dict

	def _calc_preclose_price(self, date, title):
		"""
		This helper function calculates the hypothetical last midquote before closing auctions start.
		This method takes only inputs from the self._remove_liq method.
		"""
		# df = self._snapbook.loc[(date, title), ['cont_vol_bid', 'cont_vol_ask']]
		# output = {}
		# maxbid = df.index[~df['cont_vol_bid'].isna()].max()
		# minask = df.index[(~df['cont_vol_ask'].isna()) & (df.index > 0)].min()
		#
		# output['abs_spread'] = round(minask - maxbid, 3)
		# output['midquote'] = round((maxbid + minask) / 2, 4)
		# output['rel_spread'] = round(output['abs_spread'] / output['midquote'] * 10000, 3)



		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)

		cum_bids = np.cumsum(imp_df['cont_vol_bid'].tolist())
		cum_asks = np.cumsum(np.flip(imp_df['cont_vol_ask'].tolist()))

		total = cum_bids + np.flip(cum_asks)

		i_top_bid = np.argmax(total)
		i_top_ask = len(total) - np.argmax(np.flip(total)) - 1
		maxbid, minask = imp_df['cont_vol_bid'].index[i_top_bid], imp_df['cont_vol_ask'].index[i_top_ask]


		if i_top_bid > i_top_ask:
			raise ValueError("i_top_bid not smaller than i_top_ask (spread overlap)")

		else:
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

		# for date in self._dates[:2]:
		for date in ['2019-03-01']:
			t0 = time.time()
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

				print("--- {} completed".format(symbol))

			print(">> {} finished ({} seconds)".format(date, round(time.time() - t0, 2)))

	def results_to_df(self):
		"""
		Export such that it can be used further in later stages.
		"""
		df = pd.DataFrame.from_dict(self._price_discovery_results, orient='index')
		df.index.set_names(['Date', 'Symbol'], inplace=True)
		df.sort_index()
		return df

	def export_results(self, filename, filetype):
		if filetype == 'xlsx':
			df = self.results_to_df()
			df.round(4).to_excel(os.getcwd() + "\\Data\\{}.xlsx".format(filename))
		elif filetype == 'csv':
			df = self.results_to_df()
			df.round(4).to_excel(os.getcwd() + "\\Data\\{}.xlsx".format(filename))

	def get_SB(self):
		return self._snapbook
