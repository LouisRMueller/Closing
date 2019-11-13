import pandas as pd
import numpy as np
import copy
import time


class SnapBook:
	def __init__(self, file):
		t0 = time.time()
		self._snapbook = pd.read_csv(file, header=0)
		self._symbols = self._snapbook['symbol'].unique()
		self._dates = self._snapbook['onbook_date'].unique()
		
		self._snapbook.set_index(['onbook_date', 'symbol', 'price'], drop=True, inplace=True)
		self._snapbook.sort_index(inplace=True)
		
		self._result_dict = {}  # Collects all the results
		
		print(">>> Class initiated ({} seconds)".format(round(time.time() - t0, 2)))
	
	def _calc_uncross(self, imp_df):
		"""
		Function calculates the theoretical uncross price of a closing order book.
		:param date:
		:param title:
		:return: dict()
		"""
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
		close_dict = dict(price=np.nan, imbalance=imp_df.sum().sum(), trade_vol=np.nan)
		OIB = 1
		
		while OIB != 0:
			bid_vol = df.iloc[i:, :]['close_vol_bid'].sum() + mark_buy
			ask_vol = df.iloc[:i + 1, :]['close_vol_ask'].sum() + mark_sell

			OIB = bid_vol - ask_vol
			
			if abs(OIB) >= close_dict['imbalance']:  # Normal case: If OIB is actually getting worse
				break
			elif max(bid_vol - mark_buy, ask_vol - mark_sell) == 0:   # No quotes available, hence no price
				break
			else:
				close_dict['price'] = df.index[i]
				close_dict['imbalance'] = abs(OIB)
				close_dict['trade_vol'] = min(bid_vol, ask_vol)
				
				i = i + 1 if OIB > 0 else i - 1
		
		return close_dict
	
	def _calc_preclose_price(self, date, title):
		"""
		This helper function calculates the hypothetical last midquote before closing auctions start.
		This method takes only inputs from the self._remove_liq method.
		"""
		df = self._snapbook.loc[(date, title), ['close_vol_ask', 'close_vol_bid']]
		output = {}
		maxbid = df.index[~df['cont_vol_bid'].isna()].max()
		minask = df.index[(~df['cont_vol_ask'].isna()) & (df.index > 0)].min()
		
		output['abs_spread'] = round(minask - maxbid, 3)
		output['midquote'] = round((maxbid + minask) / 2, 4)
		output['rel_spread'] = round(output['spread'] / output['midquote'] * 10000, 3)
		
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
			return imp_df
		
		bids = imp_df['close_vol_bid'].tolist()
		asks = imp_df['close_vol_ask'].tolist()
		
		if market == "remove_all":  # Removes all market orders in closing auction
			ret_df = imp_df.loc[:, ('close_vol_ask', 'close_vol_bid')]
			try:
				ret_df.loc[0, :] = 0
				return ret_df
			except KeyError:
				return ret_df
		
		# elif market == "remove_cont":  # Removes all preceeding market orders that have already been in the book before close
		# 	imp_df.loc[0, 'close_vol_bid'] = max(0, imp_df.loc[0, 'close_vol_bid'] - imp_df.loc[0, 'cont_vol_bid'])
		# 	imp_df.loc[0, 'close_vol_ask'] = max(0, imp_df.loc[0, 'close_vol_ask'] - imp_df.loc[0, 'cont_vol_ask'])
		#
		# 	ret_df = imp_df.loc[:, ('close_bid_vol', 'close_ask_vol')]
		# 	return ret_df
			
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
	
	def process_analysis(self, key, percents):
		"""
		This function is supposed to exeucte the required calculations and add it to an appropriate data format.
		It calls other helper functions in order to determine the results of the analysis.
		:param key: String with mode of calculation ['bid_limit','ask_limit','all_limit','all_market','cont_market']
		:param percents: Iterable object with integers of percentages to be looped through.
		"""
		res = self._result_dict

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
		
		res[key] = {}
		
		for date in self._dates[:2]:
			t0 = time.time()
			res[key][date] = {}
			current_symbols = self.get_SB().loc[date, :].index.get_level_values(0).unique()
			
			for symbol in current_symbols:
				# t1 = time.time()
				res[key][date][symbol] = {}
				
				for p in percents:
					res[key][date][symbol][p] = {}
					tmp_dict = res[key][date][symbol][p]
					
					orig_out = self._remove_liq(date=date, title=symbol, percentage=0)
					orig_uncross = self._calc_uncross(orig_out)
					
					remove_out = self._remove_liq(date=date, title=symbol, percentage=p, side=side, market=mkt)
					remove_uncross = self._calc_uncross(remove_out)
					
					tmp_dict['orig_price'] = orig_uncross['price']
					tmp_dict['orig_vol'] = orig_uncross['trade_vol']
					tmp_dict['orig_imbalance'] = orig_uncross['imbalance']
					
					tmp_dict['adj_price'] = remove_uncross['price']
					tmp_dict['adj_vol'] = remove_uncross['trade_vol']
					tmp_dict['adj_imbalance'] = remove_uncross['imbalance']
					
				# print("{} finished ({} seconds)".format(symbol, round(time.time() - t1, 2)))
			print(">> {} finished ({} seconds)".format(date, round(time.time() - t0, 2)))
	
	def results_to_df(self):
		"""
		Export such that it can be used further in later stages.
		"""
		pass
			
	def get_SB(self):
		return self._snapbook
	
	def get_output(self):
		return self._result_dict